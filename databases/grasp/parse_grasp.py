#!/usr/bin/env python

import sys

class parse_gwas_DBs(object):
    def __init__(self,**kwargs):
        self.db = kwargs.get('db',None)
        assert self.db
        self.failed_snps_file = open(kwargs.get('failed_snps_file',None), 'w')
        chain_file = kwargs.get('chain_file',None)
        self.k_sep = "|"
        ### if a study reports more than this many phenos,
        ### we don't report it b/c it's excedingly
        ### unlikely to be useful for us
        self.max_phenos = 100
        self._set_fields()
        if chain_file:
            try:
                from our_CrossMap import CrossMap
            except ImportError:
                sys.path.insert(1, (sys.path[0] or '.')+"/../CrossMap")
                from our_CrossMap import CrossMap
            self.CrossMap = CrossMap(chain_file)
            self.unmapped = set()
        else:
            self.CrossMap = None
        self._init_filter()
    def _init_filter(self):
        import os
        try:
            from filter_gwas import gwas_filter
        except ImportError:
            sys.path.insert(1, (sys.path[0] or '.')+"/../duma_gwas")
            from filter_gwas import gwas_filter
        self.filter = gwas_filter(log=
            os.path.abspath(os.path.dirname(sys.argv[0]))+'/filter.log'
                                 )
    def run(self, encoding='cp1252'):
        self.studies = {}
        header = None
        # A few phenotype fields are wrapped in double quotes; remove those
        # to simplify life downstream. Also, most phenotype fields are
        # straight ASCII, but the non-ascii characters 0x92, 0xb5, 0xe7
        # and 0xfb appear. The first three are cp1252; the last is less
        # clear, but decoding as cp1252 prevents unicode errors downstream.
        with open(self.db, 'r', encoding=encoding) as f:
            for l in f:
                frs = l.rstrip("\n").split("\t")
                if not header:
                    self._set_inds(frs)
                    header = True
                    continue
                k = self._clean_k(frs)
                self._process_study_data(k,frs)
                if not self._process_snp_data(k,frs):
                    if self.snp_data[1] != '':
                        self.failed_snps_file.write(self._format_out(k)+"\n")
        self.failed_snps_file.close()
        self.filter.report()
    def _process_snp_data(self,k,frs):
        self._format_snp_data(frs)
        return self._finish_process_snp_data(k)
    def _finish_process_snp_data(self,k):
        nsamples=min(float(s) for s in self.studies[k][0]) #as errors in data source may cause multiple entries
        if self.CrossMap and not self.crossmap():
            return
        if not self.filter.qc_snp(self.snp_data,nsamples,k):
            return
        print(self._format_out(k))
        return True
    def _format_snp_data(self, frs):
        self.snp_data = []
        for i in self.snp_inds:
            try:
                self.snp_data.append(frs[i])
            except IndexError:
                self.snp_data.append('')
    def _format_out(self, k):
        return k + "\t" + "\t".join(self.snp_data)
        return k.encode('utf-8') + "\t" + "\t".join(self.snp_data)
    def crossmap(self):
        input = ['chr' + self.snp_data[self.chr_ind],
                 int(self.snp_data[self.pos_ind]),
                 int(self.snp_data[self.pos_ind]),
                 '+'
                ]
        new_locs = self.CrossMap.crossMap(input)
        if new_locs:
            self.snp_data[self.chr_ind] = new_locs[0].lstrip('chr')
            self.snp_data[self.pos_ind] = str(new_locs[1])
            return True
        self.unmapped.add(' '.join([str(x) for x in input]))
    def _process_study_data(self, k,frs):
        study_data = self._prep_study_data(frs)
        self._finish_study_data(k,study_data)
    def _finish_study_data(self,k,study_data):
        if k in self.studies:
            for i in range(len(self.studies[k])):
                self.studies[k][i].add(study_data[i])
        else:
            self.studies[k] = [set([x]) for x in study_data]
    def _prep_study_data(self, frs):
        study_data = []
        for i in self.study_inds:
            try:
                study_data.append(frs[i])
            except IndexError:
                study_data.append('')
        return study_data
    def report(self, study_out, max_phenos_out):
        phenos_per_pmid = count_pmids(self.studies.keys())
        with open(study_out, 'w') as f:
            f.write("\t".join([self.k_sep.join(self.key_fields)
                              ]+ self.study_fields
                             )
                    + "\n"
                   )
            for k in self.studies:
                if phenos_per_pmid[pmid_from_k(k)] <= self.max_phenos:
                    f.write("\t".join([k] +
                                [",".join([x for x in s]) for s in self.studies[k]]
                              ) + "\n")
        with open(max_phenos_out, 'w') as f:
            f.write("\n".join([pmid for pmid,v
                                    in phenos_per_pmid.items()
                                    if v > self.max_phenos
                                    ])
                         +"\n")
        if self.CrossMap:
            sys.stderr.write("SNPs that were unmapped in crossMapping:%d\n" % len(self.unmapped))
            with open('unmapped.txt', 'w') as f:
                f.write("\n".join(self.unmapped)+'\n')

class parse_grasp(parse_gwas_DBs):
    def __init__(self,**kwargs):
        super(parse_grasp,self).__init__(**kwargs)
        self.key_fields = kwargs.get('keys', ['Phenotype','PMID'])
    def _set_fields(self):
        self.study_fields = [
                    'TotalSamples(discovery+replication)',
                    'GWASancestryDescription',
                    'Platform [SNPs passing QC]',
                    'DatePub',
                    'IncludesMale/Female Only Analyses',
                    'Exclusively Male/Female',
                    'European Discovery',
                    'African Discovery',
                    'East Asian Discovery',
                    'European Replication',
                    'African Replication',
                    'East Asian Replication'
                   ]
        self.snp_fields = ['SNPid(dbSNP134)',
                  'chr(hg19)', 'pos(hg19)',
                  'Pvalue',
                  'dbSNPMAF',
                  'dbSNPvalidation'
                 ]
    def _set_inds(self, header):
        self.study_inds = [header.index(x) for x in self.study_fields]
        self.snp_inds = [header.index(x) for x in self.snp_fields]
        self.key_inds = [header.index(x) for x in self.key_fields]
        self.chr_ind = self.snp_fields.index('chr(hg19)')
        self.pos_ind = self.snp_fields.index('pos(hg19)')
    def _clean_k(self, frs):
        k = self.k_sep.join([frs[i].strip('"')
                      for i in self.key_inds
                     ])
        k = k.replace(" ", "_").lower()
        # there are also some left and right quotes that are causing issues
        # we'll remove those
        return k.replace(u"\u2018", "").replace(u"\u2019", "")

def count_pmids(snp_studies):
    from collections import Counter
    return Counter([pmid_from_k(k)
                    for k in snp_studies
                   ])
def pmid_from_k(k, k_sep = "|"):
    return k.split(k_sep)[1]

if __name__=='__main__':
    import argparse

    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse full GRASP DB into our format")
    arguments.add_argument("db", help="Full DB")
    arguments.add_argument("o", help="studies output file")
    arguments.add_argument("f", help="failed SNPs file")
    arguments.add_argument("p", help="too many phenotypes pmid output file")
    arguments.add_argument('--chain_file', help='If the genome locations need to be mapped to a different build, please provide the appropriate Chain file')
    args = arguments.parse_args()

    pg = parse_grasp(db=args.db, failed_snps_file=args.f, chain_file = args.chain_file)
    pg.run()
    pg.report(args.o, args.p)

