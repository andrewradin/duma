#!/usr/bin/env python

class prev_resc_cmpr:
    def __init__(self, v):
        # all of these versions need to be the same
        self.params_to_cmp = ['VEP_RELEASE_VERSION', 'VCF_VERSION','VEP_UCSC_HG_VER']
        self.current_v = v
    def is_same(self):
        from versions import versions
        current_settings = versions[current_v]
        # only retreiving the last version presuming that is the most likely to be the same
        prev_settings = versions[current_v-1]
        for p in self.params_to_cmp:
            if current_settings[p] != prev_settings[p]:
                print(f'At least {p} differs between {current_v} and {current_v-1}')
                return False
        return True

if __name__=='__main__':
    import argparse,sys
    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="A wrapper to safely retreive cached VEP results")
    arguments.add_argument("v", help="Current DUMA_GWAS version")
    arguments.add_argument("f", help="file name")

    args = arguments.parse_args()
    current_v = int(args.v.lstrip('v'))
    prc = prev_resc_cmpr(current_v)
    try:
        same = prc.is_same()
    except KeyError:
        same = False

    if same:
        print(f'Using previously rescued SNPs from duma_gwas version {current_v-1}')
        fn = f'duma_gwas.v{current_v-1}.prev_rscd.tsv.gz'
        bucket='duma_gwas'
        import sys
        sys.path.insert(1, (sys.path[0] or '.')+"/../")
        from matching.move_s3_files import move_s3_files
        move_s3_files(bucket,fn)
        import subprocess
        from path_helper import PathHelper
        subprocess.check_call(['cp', f'{PathHelper.storage}{bucket}/{fn}', args.f])
    else:
        # This results in their being no previously saved SNPs, so start from scratch
        print("WARNING: no previously saved SNPs. This can be normal, but will slow the ETL, possibly considerably")
        if args.f.endswith('.gz'):
            fn = args.f[:-3]
            needs_gz=True
        else:
            fn = args.f
            needs_gz=False
        with open(fn, 'w') as f:
            pass
        if needs_gz:
            import subprocess
            subprocess.check_call(['gzip', fn])

