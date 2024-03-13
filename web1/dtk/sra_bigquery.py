

from patsearch.patent_search import ApiInfo
from django.db import transaction
import logging
logger = logging.getLogger(__name__)

TABLE_NAME = 'nih-sra-datastore.sra.metadata'


class SraBigQuery:
    def __init__(self):
        self.client = ApiInfo.bq_client()


    def get_srx_srr_mapping(self, srx_list):
        assert srx_list, "Empty srx_list, would have failed the query below"
        quoted_srx_list = [f'"{srx}"' for srx in srx_list]
        query = f"""SELECT experiment, acc
        FROM `{TABLE_NAME}`
        WHERE experiment IN ({','.join(quoted_srx_list)})
        """
        logger.info(f"SRA query is {query}")

        query_job = self.client.query(query)

        from collections import defaultdict
        srx_srr = []
        for row in query_job.result():
            if not row:
                # This happens in patent search, not sure if it will happen here
                # but better to be safe.
                logger.warning("Skipping empty row: %s", row)
                continue

            content = dict(list(row.items()))
            srx_srr.append((content['experiment'], content['acc']))

        from dtk.data import MultiMap
        return MultiMap(srx_srr)

    @transaction.atomic
    def search_projects(self, project_accs, species='human'):
        """Searches for bioproject data in the SRA bigquery table.

        Data is saved to the local SraRun table.
        """
        from browse.models import AeSearch

        if not project_accs:
            logger.info("No accs to search for")
            return

        human_filter_q = ''
        if species != AeSearch.species_vals.any:
            latin = AeSearch.latin_of_species(species)
            human_filter_q = f'AND organism = "{latin}"'

        quoted_accs = [f'"{acc}"' for acc in project_accs]

        query = f"""SELECT experiment, acc, biosample, bioproject, attributes
        FROM `{TABLE_NAME}`
        WHERE bioproject IN ({','.join(quoted_accs)})
        AND consent = "public"
        {human_filter_q}
        """
        logger.info(f"SRA query is {query}")

        query_job = self.client.query(query)

        from collections import defaultdict
        logger.info("Importing SraRun data")
        for row in query_job.result():
            if not row:
                # This happens in patent search, not sure if it will happen here
                # but better to be safe.
                logger.warning("Skipping empty row: %s", row)
                continue
            content = dict(list(row.items()))

            from ge.models import SraRun
            sra, new = SraRun.objects.get_or_create(
                    experiment=content['experiment'],
                    bioproject=content['bioproject'],
                    )
            logger.info(f"Found SRA {sra}, {new}")
            if new:
                sra.biosample = content['biosample']
                sra.accession = content['acc']

                # We use the attrs data directly even though they have a jattr
                # already json'ified because jattr is missing data from repeated
                # keys, which happens to contain the GEO data we need.
                attr_list = []
                for attribute in content['attributes']:
                    k, v = attribute['k'], attribute['v']
                    attr_list.append([k,v])
                    if k == 'primary_search' and v.startswith('GSE'):
                        sra.geo_id = v
                import json
                sra.attrs_json = json.dumps(attr_list)
                sra.save()

        logger.info("SraRun data imported")
