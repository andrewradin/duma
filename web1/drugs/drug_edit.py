

from .models import DrugProposal
from dtk.affinity import c50_to_ev, ki_to_ev

import logging
logger = logging.getLogger(__name__)

def make_attrs(proposal):
    import json
    data = json.loads(proposal.data)
    attrs_data = data['attrs']
    # Add a proposal ID so we can track if the imported version
    # of a drug is out-of-date.
    attrs_data.append({
        'name': 'duma_proposal_id',
        'value': str(proposal.id)
        })
    id = proposal.collection_drug_id
    for entry in attrs_data:
        name = entry['name']
        value = entry['value']
        if isinstance(value, list):
            for val_inst in value:
                if val_inst.strip() != '':
                    yield (id, name, val_inst)
        else:
            if value != '':
                yield (id, name, value)


def get_computed_evidence(ev, c50, ki):
    if ev:
        return float(ev)
    elif c50:
        return c50_to_ev(float(c50))
    elif ki:
        return ki_to_ev(float(ki))
    else:
        return None


def make_dpis(proposal):
    import json
    data = json.loads(proposal.data)
    dpi_data = data['dpi']
    id = proposal.collection_drug_id
    for entry in dpi_data:
        if not entry['keep']:
            continue
        uniprot = entry['uniprot'].strip()
        direction = entry['direction']
        ev = get_computed_evidence(entry.get('evidence'), entry.get('c50'), entry.get('ki'))
        if not ev:
            # No evidence or measurement values at all, skip.
            continue

        ev = '%.1f' % float(ev)


        yield (id, uniprot, ev, direction)

def collection_integrity_check():
    published_keys = set()
    proposals = DrugProposal.objects.filter(state=DrugProposal.states.ACCEPTED)
    dups = set()
    for proposal in proposals:
        key = proposal.collection_drug_id
        if key in published_keys:
            dups.add(key)
        published_keys.add(key)
    if dups:
        # If this happens a lot, we can work out a better error reporting
        # path back to the Review Drug Edits page.
        raise RuntimeError(
                'multiple accepted proposals for: ' + ', '.join(sorted(dups))
                )

def collection_attrs():
    proposals = DrugProposal.objects.filter(state=DrugProposal.states.ACCEPTED)
    attrs = [('duma_id', 'attribute', 'value')]
    for proposal in proposals:
        attrs.extend(make_attrs(proposal))
    return attrs


def collection_dpi():
    proposals = DrugProposal.objects.filter(state=DrugProposal.states.ACCEPTED)
    attrs = [('duma_id', 'uniprot_id', 'evidence', 'direction')]
    for proposal in proposals:
        attrs.extend(make_dpis(proposal))
    return attrs

def validate_smiles(smiles):
    from dtk.standardize_mol import standardize_mol
    from rdkit import Chem
    if not smiles or not smiles.strip():
        raise Exception(f"Empty smiles")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise Exception(f"Failed to parse SMILES code: {smiles}")
    mol = Chem.MolFromInchi(Chem.MolToInchi(mol))
    clean_mol = standardize_mol(mol, max_tautomers=1000)

    std_sm = Chem.MolToSmiles(clean_mol, isomericSmiles=False)

    from drugs.models import Drug
    matched = Drug.objects.filter(blob__value=std_sm).filter(tag__prop__name='canonical').values('id', 'collection__name', 'tag__value')

    return {
        'matches': list(matched)
    }


class DumaCollection:
    attr_s3_fn = 'raw.create.duma.full.tsv'
    dpi_s3_fn = 'raw.dpi.duma.default.tsv'
    processed_attr_s3_fn = 'create.duma.full.tsv'

    def __init__(self):
        from dtk.aws_api import AwsBoto3Base
        aws = AwsBoto3Base()
        self.aws = aws
        self.bucket = aws.s3.Bucket('duma-datasets')

    def publish(self):
        collection_integrity_check()
        from path_helper import PathHelper
        if not PathHelper.cfg('can_publish_drugsets'):
            print("Publishing drugsets not enabled")
            return
        import io
        logger.info("Publishing new DPI and attr data for the duma collection")
        attr_bytes = io.BytesIO(self.format_attr_data().encode('utf8'))
        dpi_bytes = io.BytesIO(self.format_dpi_data().encode('utf8'))
        self.bucket.upload_fileobj(attr_bytes, self.attr_s3_fn)
        self.bucket.upload_fileobj(dpi_bytes, self.dpi_s3_fn)

    def remote_as_str(self, fn):
        try:
            import io
            buf = io.BytesIO()
            self.bucket.download_fileobj(fn, buf)
            buf.seek(0)
            return buf.read().decode('utf8')
        except:
            return ""

    def remote_timestamp(self, fn, bucket=None):
        bucket = bucket or self.bucket
        try:
            return bucket.Object(fn).last_modified
        except:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(0).replace(tzinfo=timezone.utc)

    def remote_attr_data(self):
        return self.remote_as_str(self.attr_s3_fn)

    def remote_dpi_data(self):
        return self.remote_as_str(self.dpi_s3_fn)

    def remote_attr_timestamp(self):
        return self.remote_timestamp(self.attr_s3_fn)

    def remote_processed_attr_timestamp(self):
        proc_bucket = self.aws.s3.Bucket('2xar-duma-drugsets')
        return self.remote_timestamp(self.processed_attr_s3_fn, proc_bucket)

    def remote_dpi_timestamp(self):
        return self.remote_timestamp(self.dpi_s3_fn)

    def duma_import_timestamp(self):
        from drugs.models import UploadAudit
        latest_upload = UploadAudit.objects.filter(filename=self.processed_attr_s3_fn).order_by('-timestamp')

        if len(latest_upload) == 0:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(0).replace(tzinfo=timezone.utc)

        return latest_upload[0].timestamp

    def out_of_date_info(self):
        # There are 4'ish out-of-date states.
        # 1) Latest generated doesn't match published (i.e. hit publish)
        # 2) Published doesn't match merged (i.e. re-run ETL)
        # 3) Merged doesn't match DB (i.e. reimport to platform)
        # 4) DB doesn't match workspace (i.e. reimport to workspace)
        # We're not going to track #4 here, though, because it's ws-specific.


        published = self.format_attr_data() == self.remote_attr_data() and \
                    self.format_dpi_data() == self.remote_dpi_data()

        generated = self.remote_attr_timestamp() <= self.remote_processed_attr_timestamp()

        imported = self.remote_processed_attr_timestamp() <= self.duma_import_timestamp()

        return dict(published=published, generated=generated, imported=imported)

    def format_attr_data(self):
        attr_data = '\n'.join('\t'.join(x) for x in collection_attrs())
        return attr_data

    def format_dpi_data(self):
        dpi_data = '\n'.join('\t'.join(x) for x in collection_dpi())
        return dpi_data
