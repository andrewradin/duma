#!/usr/bin/env python3

from dtk.lazy_loader import LazyLoader

# This class checks the clusterer output against UniChem.
#
# Because we cluster more aggressively than UniChem, we expect clusterer
# matches that don't show up in UniChem. So to construct a metric, we
# only look for UniChem matches that don't show up in the clusterer, and
# report that as a fraction.
#
# UniChem data is stored in pairwise files, so in any one run we only
# compare two sources (drugbank and pubchem, for example). Comparing
# 
# Since the unichem ETL doesn't currently extract any of the pair files
# we need, this downloads them directly.

class UniChemChecker(LazyLoader):
    def check(self,src1,src2,show_errors=0):
        uchem = self.load_uchem(src1,src2)
        match = self.load_match(src1,src2)
        all_keys = set(uchem.keys()) | set(match.keys())
        matches = 0
        errors = 0
        for key in all_keys:
            v_uchem = uchem.get(key,set())
            v_match = match.get(key,set())
            # Note that since we do a one-sided comparison, a possible
            # silent failure is a uchem that always returns an empty
            # set, which will appear as a 100% match.
            # XXX is there a good way to guard against this?
            if v_uchem <= v_match:
                matches += 1
            else:
                errors += 1
                if errors <= show_errors:
                    # XXX add more info for tracking mismatches?
                    print(
                        key,
                        'both',v_uchem & v_match,
                        'uchem only',v_uchem - v_match,
                        'match only',v_match - v_uchem,
                        )
        found = matches/len(all_keys)
        print(matches,'matches out of',len(all_keys),src1,'keys',
                f'({100*found:0.4f}%)'
                )
        return found
    def load_match(self, src1, src2):
        src1 = src1+'_id'
        src2 = src2+'_id'
        result = {}
        from dtk.files import get_file_records
        from dtk.drug_clusters import assemble_pairs
        for rec in get_file_records(self.cluster_fn,keep_header=None):
            pairs = assemble_pairs(rec)
            src1_keys = [x[1] for x in pairs if x[0] == src1]
            src2_keys = [x[1] for x in pairs if x[0] == src2]
            if src1_keys and src2_keys:
                for key1 in src1_keys:
                    result[key1] = set(src2_keys)
        return result
    def load_uchem(self,src1,src2):
        # make sure file is present
        inds = self.uc._get_order(src1,src2)
        names = [src1,src2]
        role = self.uc._get_file_role(names[inds[0]], names[inds[1]])
        fn = role + '.tsv.gz'
        import os
        if not os.path.exists(fn):
            import subprocess
            # XXX note assumption about relative location of unichem.py
            subprocess.run([
                    '../unichem/unichem.py',
                    '--base_source',src1,
                    '--other_sources',src2,
                    '--refresh',
                    ],
                    check=True
                    )
        filt1 = self.get_filter(src1)
        filt2 = self.get_filter(src2)
        return {
                k:set(v) if filt2 is None else set(v) & filt2
                for k,v in self.uc.get_converter_dict_from_file(fn,inds).items()
                if filt1 is None or k in filt1
                }
    def get_filter(self,src):
        if src == 'pubchem':
            return self.pubchem_keys
        if src == 'chembl':
            return self.chembl_keys
        if src == 'bindingdb':
            return self.bindingdb_keys
        # this doesn't matter; we extract all drugbank keys
        #if src == 'drugbank':
        #    return self.drugbank_keys
        return None
    def _uc_loader(self):
        from dtk.unichem import UniChem
        return UniChem()
    def get_keys_for_source(self,src):
        from dtk.etl import get_last_published_version
        ver = get_last_published_version(src)
        print(src,'at version',ver)
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(src,f'full.v{ver}','attributes')
        s3f.fetch()
        from dtk.files import get_file_records
        return set(
                rec[0]
                for rec in get_file_records(s3f.path(),keep_header=False)
                )
    def _pubchem_keys_loader(self):
        return self.get_keys_for_source('pubchem')
    def _chembl_keys_loader(self):
        return self.get_keys_for_source('chembl')
    def _bindingdb_keys_loader(self):
        return self.get_keys_for_source('bindingdb')
    def _drugbank_keys_loader(self):
        return self.get_keys_for_source('drugbank')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='compare clustering with unichem'
            )
    parser.add_argument('--show-errors',type=int,default=0),
    parser.add_argument('cluster_file'),
    parser.add_argument('unichem_src1'),
    parser.add_argument('unichem_src2'),
    args = parser.parse_args()

    uchk = UniChemChecker()
    uchk.cluster_fn = args.cluster_file
    if args.unichem_src1 == '-':
        # these are all our data sources that overlap with
        # unichem, not including selleck
        srcs = ['bindingdb','chembl','pubchem','drugbank']
        results = []
        for i in range(len(srcs)):
            for j in range(i+1,len(srcs)):
                label = f'{srcs[i]} vs. {srcs[j]}'
                print(label)
                found = uchk.check(srcs[i],srcs[j],args.show_errors)
                results.append((
                        label,
                        f'({100*found:0.4f}%)',
                        ))
        from dtk.text import print_table
        print_table(results)
    else:
        uchk.check(args.unichem_src1,args.unichem_src2,args.show_errors)

