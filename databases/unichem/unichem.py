#!/usr/bin/env python

class UniChem:
    rewrites={
            'bindingdb':lambda x:'BDBM'+x,
            'pubchem':lambda x:'PUBCHEM'+x,
            }
    def __init__(self,base,osources,pattern=None):
        import os
        try:
            from path_helper import PathHelper
        except ImportError:
            import sys
            sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
            from path_helper import PathHelper
        self.map_file = os.path.join(
                               PathHelper.website_root,
                               'browse',
                               'static',
                               'unichem_src_map.tsv'
                               )
        self.base_source_name = base
        self.other_source_names = osources
        self.pattern = pattern or "%.tsv.gz"
        self._get_source_ids()
        self.ftp=None
    def _get_source_ids(self):
        from dtk.files import get_file_records
        self.source_map={}
        header = None
        for frs in get_file_records(self.map_file):
            if header is None:
                header = frs
                continue
            self.source_map[frs[header.index('name')]] = int(frs[header.index('id')])
        assert self.base_source_name in self.source_map
        assert all([x in self.source_map for x in self.other_source_names])
    def filenames(self,src1,src2):
        mapping = "%s_to_%s" % (src1,src2)
        return ("src%dsrc%d.txt.gz" % (self.source_map[src1],self.source_map[src2]),
                self.pattern.replace('%',mapping))
    def fetch(self,src1,src2):
        if self.ftp is None:
            from ftplib import FTP
            self.ftp = FTP('ftp.ebi.ac.uk')
            self.ftp.login()
        root="pub/databases/chembl/UniChem/data/wholeSourceMapping"
        wd="/%s/src_id%d" % (root,self.source_map[src1])
        self.ftp.cwd(wd)
        ufn,ofn = self.filenames(src1,src2)
        need_rewrite = src1 in self.rewrites or src2 in self.rewrites
        if need_rewrite:
            ffn = ofn
            ofn = 'tmp1.'+ffn
            o2fn = 'tmp2.'+ffn
        print(ufn,'now in',ofn)
        with open(ofn,'wb') as f:
            self.ftp.retrbinary("RETR %s" % ufn, f.write)
        if need_rewrite:
            print(ufn,'now in',o2fn)
            map1 = self.rewrites.get(src1,lambda x:x)
            map2 = self.rewrites.get(src2,lambda x:x)
            from dtk.files import get_file_records,FileDestination
            seen_header=False
            with FileDestination(o2fn,gzip=True) as fd:
                for rec in get_file_records(ofn):
                    if not seen_header:
                        seen_header = True
                    else:
                        rec[0] = map1(rec[0])
                        rec[1] = map2(rec[1])
                    fd.append(rec)
            import os
            os.rename(o2fn,ffn)
            print(ufn,'now in',ffn)
    def old_refresh(self):
        for src1 in self.sources:
            for src2 in self.sources:
                if src2 > src1:
                    self.fetch(src1,src2)
    def refresh(self):
        for osrc in self.other_source_names:
            if self.source_map[osrc] < self.source_map[self.base_source_name]:
                self.fetch(osrc,self.base_source_name)
            else:
                self.fetch(self.base_source_name,osrc)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='unichem utility')
    parser.add_argument('--base_source'
        ,default='chembl'
        ,help="The reference source to which all other sources are mapped"
        )
    parser.add_argument('--other_sources'
        ,default='drugbank,surechembl,pubchem,lincs,bindingdb,zinc'
        ,help="comma-separated list of source names to compare to the base_source"
        )
    parser.add_argument('--pattern'
        ,help="output filename pattern (%% will become name1_to_name2)"
        )
    parser.add_argument('--mapping'
        ,help="specify sources as name1_to_name2 (alternative to base and other)"
        )
    parser.add_argument('--refresh'
        ,action="store_true"
        ,help="pull down latest versions of relevant unichem files"
        )

    args = parser.parse_args()
    if args.mapping:
        args.base_source,args.other_sources = args.mapping.split('_to_')
    osources = args.other_sources.split(',')
    assert osources
    uc = UniChem(args.base_source,osources,pattern=args.pattern)
    if args.refresh:
        uc.refresh()

