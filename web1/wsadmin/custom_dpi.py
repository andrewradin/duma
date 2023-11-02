
from django.db import transaction
import os
import logging
from tools import Enum

logger = logging.getLogger(__name__)

# Normally we'd put this in the model, but ideally the workflow
# here for workers doesn't involve importing the model, which means
# this enum needs to exist outside of it.
# (Otherwise, need to do an 'import django_setup' at the start of
# any worker entrypoints)
CustomDpiModes = Enum([], [
    ('SUBTRACT',),
    ('EXCLUSIVE',),
    ])


class S3Remote:
    """
    NOTE:
    This gets mocked out in unit tests.
    If you're changing how the remote works, take a look at the unit tests and try to avoid
    hitting s3 during them.
    """
    def __init__(self):
        from dtk.s3_cache import S3Bucket
        self.s3b = S3Bucket('customdpi')

    def upload_fileobj(self, fileobj, path):
        b = self.s3b.bucket.aws_bucket
        logger.info(f"Uploading to s3 at {path}")
        b.upload_fileobj(fileobj, path)
    
    def fetch(self, fileobj, path):
        from botocore.exceptions import ClientError
        try:
            b = self.s3b.bucket.aws_bucket
            logger.info(f"Fetching from s3 {path}")
            b.download_fileobj(path, fileobj)
        except ClientError as e:
            raise FileNotFoundError(e)


class CustomDpiRemote:
    def __init__(self):
        self._remote = S3Remote()
    
    def path(self, uid):
        return f'{uid}.json'

    def push(self, data, uid):
        import json
        import io
        path = self.path(uid)
        data_bin = json.dumps(data).encode('utf8')
        logger.info(f"Uploading {uid} to {path}")
        self._remote.upload_fileobj(io.BytesIO(data_bin), path)

    def fetch(self, uid):
        import io
        import json
        bin_data = io.BytesIO()
        self._remote.fetch(bin_data, self.path(uid))
        bin_data.seek(0)
        return json.loads(bin_data.read().decode('utf8'))



def custom_dpi_path(uid):
    from path_helper import PathHelper 

    path = os.path.join(PathHelper.customdpi, uid + '.tsv')

    if not os.path.exists(path):
        logger.info(f"Couldn't find custom dpi {uid} at {path}, generating it")
        os.makedirs(PathHelper.customdpi, exist_ok=True)
        # Fetch details from S3.
        remote = CustomDpiRemote()

        try:
            details = remote.fetch(uid)
        except FileNotFoundError:
            # If we're working with MoAs, we still need the molecule custom variant for DEFUS
            # This is a bit of a hacky workaroud for it, which allows us to generate the
            # non-moa customdpi from the moa customdpi.
            prefix, ver = uid.split('.')
            if not prefix.endswith('-moa'):
                moa_uid = f'{prefix}-moa.{ver}'
                details = remote.fetch(moa_uid)
                assert '-moa' in details['base_dpi']
                details['base_dpi'] = details['base_dpi'].replace('-moa', '')
            else:
                raise

        # Generate.
        from atomicwrites import atomic_write
        with atomic_write(path, overwrite=True) as f:
            logger.info(f"Writing custom dpi {uid} at {path}")
            for rec in generate_custom_dpi_entries(**details):
                f.write('\t'.join(rec) + '\n')
    
    return path


def generate_custom_dpi_entries(base_dpi, uniprots, mode):
    from dtk.prot_map import DpiMapping
    dpi = DpiMapping(base_dpi)
    prots = set(uniprots)
    mode = int(mode)

    from dtk.files import get_file_records
    is_header = True
    for rec in get_file_records(dpi.get_path(), keep_header=True):
        if is_header:
            keep = True
            is_header = False
        elif mode == CustomDpiModes.SUBTRACT:
            keep = rec[1] not in prots
        elif mode == CustomDpiModes.EXCLUSIVE:
            keep = rec[1] in prots
        else:
            raise Exception(f"Unhandled mode {mode}")
        
        if keep:
            yield rec
        
        
def get_env():
    # We associate with the lts branch so that these files match LTS in terms of
    # branching & cleanup.
    from path_helper import PathHelper
    # The .'s in the standard dev lts branches cause problems with parsing, so remove.
    env = PathHelper.cfg('lts_branch').replace('.', '')
    from dtk.tests import is_pytest
    if is_pytest():
        # Ensures that pytests aren't going to overwrite real data on s3.
        env = f'pytest_env_{env}'
    return env

    
def custom_dpi_entries(ws):
    from .models import CustomDpi
    return list(CustomDpi.objects.filter(ws=ws).values_list('uid', 'name', 'deprecated'))

def create_custom_dpi(base_dpi, protset, mode, name, descr, ws, user):
    from .models import CustomDpi
    from dtk.prot_map import DpiMapping
    # If this fails on S3, don't want a DB entry lying around.
    with transaction.atomic():

        # Pull the protset.
        uniprots = ws.get_uniprot_set(protset)
        customdpi_data = {
            "uniprots": list(sorted(uniprots)),
            "base_dpi": base_dpi,
            "mode": mode,
        }

        base_dpi_obj = DpiMapping(base_dpi)

        # Create the database objs
        custom_dpi = CustomDpi.objects.create(
            ws=ws,
            base_dpi=base_dpi,
            prot_set=protset,
            mode=mode,
            descr=descr,
            created_by=user,
            name=name,
            uid='', # To be filled later.

            protset_prots=len(uniprots),
            base_prots=base_dpi_obj.get_uniq_target_cnt(),
            base_molecules=base_dpi_obj.get_uniq_mol_cnt(),
            # These get filled in momentarily.
            final_prots=-1,
            final_molecules=-1,
        )

        # The env needs to get baked into the uid, otherwise this won't be findable on the worker.
        # Otherwise we would have ideally separated the envs into different subdirs.
        env = get_env()

        # This should be unique, but shouldn't be programmatically parsed (except the cstm_ to id the type).
        # It's used for s3 filename and dpi entry.
        # The DPI entry must be parseable by the DpiMapping.versioned*RE so that it recognizes which cluster version to use.
        # Some fields aren't required, but helps for human parsing.
        from dtk.prot_map import DpiMapping
        prefix = DpiMapping.custom_dpi_prefix
        uid = f'{prefix}{env}_{protset}_m{mode}_cd{custom_dpi.id}_{base_dpi}'
        logger.info(f"Created DPI with uid {uid}")
        custom_dpi.uid = uid

        # Save to s3
        remote = CustomDpiRemote()
        remote.push(customdpi_data, uid)

        # Let's make sure this uid works, and fill in some stats while we're at it.
        custom_dpi_obj = DpiMapping(uid)
        custom_dpi.final_prots = custom_dpi_obj.get_uniq_target_cnt()
        custom_dpi.final_molecules = custom_dpi_obj.get_uniq_mol_cnt()
        custom_dpi.save()




