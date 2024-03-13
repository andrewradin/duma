# XXX It may be largely historical accident what code is in dtk.ae_parser
# XXX and what code is here. This could probably be cleaned up some day.

import logging

from requests.models import HTTPError
logger = logging.getLogger(__name__)


# Warn if more than this many results.
GEO_RESULTS_WARN_NUM = 500

def refresh_ae_search(aesearch, add_new_results):
    # Doing a matching ae search will simply refresh the existing one.
    do_ae_search(
        term=aesearch.term,
        mode=aesearch.mode,
        species=aesearch.species,
        ws=aesearch.ws,
        add_new_results=add_new_results,
        )


def do_ae_search(term, mode, species, ws, add_new_results=True):
    """
    If not add_new_results, only existing results will be refreshed, new
    accessions will not be added, the timestamp will not be adjusted, and
    scores will not be updated.
    """
    from browse.models import AeSearch
    from dtk.ae_parser import parser
    run = parser(
            ws = ws,
            disease = term,
            tr_flag = (mode == AeSearch.mode_vals.TR),
            species = species,
            )
    run.run(add_new_results=add_new_results)

    msgs = []
    msgs.extend(add_geo_to_run(run, add_new_results))
    result = run.results
    from browse.models import AeSearch,AeAccession
    from django.utils import timezone
    if add_new_results:
        srch,new = AeSearch.objects.update_or_create(
                ws=ws,
                term=term,
                mode=mode,
                species=species,
                defaults={
                    'when':timezone.now(),
                    },
                )
    else:
        new = False
        srch = AeSearch.objects.get(
                ws=ws,
                term=term,
                mode=mode,
                species=species,
        )
    from browse.models import Tissue
    src_choices = Tissue.method_choices()
    src_norm = src_choices[0][0]
    src_seq = [x[0] for x in src_choices if x[1]=='RNAseq'][0]

    accs = []
    from django.db import transaction
    with transaction.atomic():
        logger.info(f"Updating {len(result)} results")
        for item in result.values():
            src_default = src_norm
            if 'RNA-seq of coding RNA' in item.experiment_type or 'Expression profiling by high throughput sequencing' in item.experiment_type:
                src_default = src_seq
            if item.accession.startswith("PRJ"):
                # Everything we pull from bioproj is RNASeq.
                src_default = src_seq
            alt_ids = ','.join(item.alt_ids)
            if add_new_results:
                acc,new = AeAccession.objects.update_or_create(
                        ws=ws,
                        geoID=item.accession,
                        defaults={
                                'desc':item.orig_description,
                                'title':item.orig_title,
                                'src_default':src_default,
                                'pub_ref':item.pub_ref,
                                },
                        )
            else:
                new = False
                try:
                    acc = AeAccession.objects.get(ws=ws, geoID=item.accession)
                except AeAccession.DoesNotExist:
                    continue

            from ge.models import GESamples
            if not GESamples.objects.filter(geoID=acc.geoID):
                gesamples = GESamples.objects.create(geoID=acc.geoID)
                gesamples.attrs_json_gz = GESamples.compress_sample_attrs(item.samples)
                gesamples.save()

            changed = False
            if acc.alt_ids != alt_ids:
                acc.alt_ids = alt_ids
                changed = True

            # update searches from before pub_ref existed
            if acc.pub_ref != item.pub_ref:
                acc.pub_ref = item.pub_ref
                changed = True
            
            if acc.num_samples != item.sample_n:
                acc.num_samples = item.sample_n
                changed = True

            if acc.experiment_type != item.experiment_type:
                acc.experiment_type = item.experiment_type
                changed = True

            if changed:
                acc.save()
            accs.append(acc)
        srch.version = AeSearch.LATEST_VERSION
        srch.save()
    
    if add_new_results:
        from browse.default_settings import OmicsSearchModel
        from runner.process_info import JobInfo
        jid = OmicsSearchModel.value(ws=ws)
        if jid != '':
            bji = JobInfo.get_bound(ws, jid)
            score_model = bji.load_trained_model()
        else:
            from scripts.gesearch_model import PreviousModel
            score_model = PreviousModel({})
        score_search(srch, accs, score_model)

    logger.info("AESearch completed")
    return msgs

def score_search(srch, accs, score_model):
    logger.info(f"Scoring results for {srch}")
    from browse.models import AeSearch,AeAccession,AeScore
    from scripts.gesearch_model import make_entry
    # clear out any stale scores, in case the search
    # doesn't include all the same accessions
    AeScore.objects.filter(search=srch).delete()
    
    entries = [(make_entry(accession, srch, srch.mode), 0) for accession in accs]
    scores = score_model.run_predict(entries)

    for acc, score in zip(accs, scores):
        AeScore.objects.update_or_create(
                search=srch,
                accession=acc,
                defaults={'score':score},
                )

def parse_gds_samples(fileobj):
    samples = []
    from collections import defaultdict
    cur_sample = None
    def flush_sample():
        if not cur_sample:
            return
        
        sample = {}
        for k, v in cur_sample.items():
            if k.startswith("characteristics_"):
                # Characteristics is a special case, with : separated key-values.
                # However, some have unexpected formatting here, so track those as leftovers
                # and write them separately.
                leftovers = []
                for entry in v:
                    parts = [x.strip() for x in entry.split(':', maxsplit=1)]
                    if len(parts) == 2:
                        char_name, char_val = parts
                        sample[f'ch_{char_name}'] = char_val
                    else:
                        leftovers.append(entry)
                v = leftovers
            
            if v:
                sample[k] = ', '.join(v)

        samples.append(sample)

    s1 = ord('^')
    s2 = ord('!')
    for line in fileobj:
        # These files often contain summarized results, which can be huge.
        # Instead of decoding every line, first check that we're looking at our 
        # sample metadata.
        # NOTE: This parser is optimized for the soft family files, which were often filled
        # with huge results datasets. We now hit their CGI endpoint that shows just the sample data,
        # so some of the optimizations are probably no longer necessary.
        s = line[0]
        if s != s1 and s != s2:
            continue
        line = line.decode('utf8', errors='ignore')
        if line.startswith('^SAMPLE'):
            parts = [x.strip() for x in line.split('=')]
            sample_id = parts[1]
            flush_sample()
            cur_sample = defaultdict(list)
            if len(samples) >= 1000:
                logger.warn("WARNING: parsed 1000 samples, ending parse")
                break
        elif line.startswith('!Sample_'):
            parts = [x.strip() for x in line.split('=')]
            key = parts[0][len('!Sample_'):]
            cur_sample[key].append(parts[1])

    flush_sample()

    return samples

def find_existing_samples(acc_ids):
    from collections import defaultdict
    from ge.models import GESamples
    import json
    results = defaultdict(list)
    remaining = []

    all_samples = GESamples.objects.filter(geoID__in=acc_ids)
    results = {s.geoID: s.get_sample_attrs() for s in all_samples}
    
    for acc_id, v in results.items():
        logger.info(f"Using {len(v)} cached samples for {acc_id}")
    
    remaining = [x for x in acc_ids if x not in results]
    return results, remaining

def fetch_gds_samples_for_accs(acc_ids):
    from dtk.parallel import pmap
    out, remaining = find_existing_samples(acc_ids)
    # We're mostly just waiting for GEO to respond, so we can exceed our cores here,
    # but don't want to go crazy and overload GEO.
    results = pmap(fetch_gds_samples, remaining, num_cores=10)
    out.update(dict(zip(remaining, results)))
    return out

def fetch_gds_samples(acc_id):
    logger.info(f"Pulling samples for {acc_id}")
    short_acc = acc_id[:-3] + 'nnn'
    # In rare cases, the soft file in massive (multiple GB).
    # Instead, we'll hit the CGI endpoint and request just the sample data.
    #url = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{short_acc}/{acc_id}/soft/{acc_id}_family.soft.gz'
    url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc_id}&targ=gsm&form=text&view=brief'
    import requests
    with requests.get(url, stream=True, timeout=60) as req:
        try:
            req.raise_for_status()
            out = parse_gds_samples(req.raw)
            logger.info(f"Received {acc_id}")
            return out
        except requests.exceptions.HTTPError as e:
            import traceback as tb
            tb.print_exc()
            logger.error(f"ERROR: HTTP error {e} fetching {acc_id} at {url}")
            return []
        except (OSError, EOFError):
            import traceback as tb
            tb.print_exc()
            logger.error(f"ERROR: Failed to parse gds acc {acc_id}, ignoring")
            return []
        except:
            logger.error(f"ERROR: Failed while parsing gds acc {acc_id}")
            raise

def add_geo_to_run(run, add_new_results):
    msgs = []
    logger.info("Doing geo search")
    # conduct a backup search against GEO
    from dtk.entrez_utils import GeoSearch
    gs = GeoSearch(term=run.disease.search_term, species=run.species)
    if len(gs.results) > GEO_RESULTS_WARN_NUM:
        msg = 'Search matched %d accessions on GEO' % gs.full_count
        msg += '; consider refining your search'
        if len(gs.results) < gs.full_count:
            msg += '; only %d returned (randomly selected)'%len(gs.results)
        msgs.append(msg)

    geo_set = set(gs.results.keys())
    ae_geo_prefix = 'E-GEOD-'
    from dtk.ae_parser import Experiment
    geo_to_ae = {}

    new_geo_set = set()
    for geoID in geo_set:
        # Some of these were already found via AE.
        # Skip them, but track their alternative ID.
        ae_id = ae_geo_prefix + geoID[3:]
        if ae_id in run.results:
            geo_to_ae[geoID] = ae_id
            run.results[ae_id].alt_ids.append(geoID)
            continue
        else:
            # If we're not adding new results, check that this already exists.
            if not add_new_results:
                from browse.models import AeAccession
                if not AeAccession.objects.filter(ws=run.disease.ws, geoID=geoID):
                    continue
            new_geo_set.add(geoID)

    logger.info(f"Fetching gds samples for {len(new_geo_set)} geos")
    acc_to_samples = fetch_gds_samples_for_accs(new_geo_set)
    logger.info("Done Fetching gds samples")

    for geoID in new_geo_set:
        # Otherwise, this is a new result, add it.
        data = gs.results[geoID]
        e = Experiment(geoID,run.disease,run.tr_flag)
        run.results[geoID] = e
        e.experiment_type = data.get('experiment_type', '')
        e.orig_title = data['title']
        e.title = run._prep_text(data['title'])
        e.orig_description = data['summary']
        e.description = run._prep_text(data['summary'])
        e.table_headers = ''
        e.table_vals = ''
        e.pub_ref = data.get('pmid','')
        e.sample_n = data['sample_n']
        e.samples = acc_to_samples[geoID]

    logger.info("Doing SRA search")
    sras = query_sra(
            term=run.disease.search_term,
            species=run.species,
            )
    for bioID, data in sras.items():
        geo_id = data['geo_id']
        geo_id = geo_to_ae.get(geo_id, geo_id)
        if geo_id and geo_id in run.results:
            run.results[geo_id].alt_ids.append(bioID)
            continue


        e = Experiment(bioID,run.disease,run.tr_flag)
        run.results[bioID] = e
        e.experiment_type = data.get('data_type', '')
        e.orig_title = data['title']
        e.title = run._prep_text(data['title'])
        e.orig_description = data['summary']
        e.description = run._prep_text(data['summary'])
        e.table_headers = ''
        e.table_vals = ''
        e.pub_ref = data.get('pmid','')
        e.sample_n = data['sample_n']
    return msgs

def query_sra(term, species):
    from dtk.entrez_utils import SraSearch
    sras = SraSearch(term=term, species=species)
    results = sras.results

    # Ask bigquery for more information about these.
    from dtk.sra_bigquery import SraBigQuery
    sbq = SraBigQuery()
    accs = [x['id'] for x in results.values()]
    sbq.search_projects(accs, species=species)
    # The results get inserted into SraRun.
    from ge.models import SraRun

    for bio_id, data in results.items():
        samples = SraRun.objects.filter(bioproject=bio_id).values_list('experiment', flat=True)
        data['sample_n'] = len(samples)
        data['samples'] = samples
        if samples:
            # Could be null, that is fine
            geo_id = SraRun.objects.get(experiment=samples[0]).geo_id
            data['geo_id'] = geo_id

    # Filter out anything with no samples.
    results = {k:v for k,v in results.items()
                if v['sample_n'] > 0}

    return results
