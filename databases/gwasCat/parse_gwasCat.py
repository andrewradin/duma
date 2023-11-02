#!/usr/bin/env python3

import sys
try:
    from dtk.files import get_file_records
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import get_file_records
try:
    from parse_grasp import parse_gwas_DBs
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../grasp")
    from parse_grasp import parse_gwas_DBs

class parse_gwascat(parse_gwas_DBs):
    def __init__(self,**kwargs):
        super(parse_gwascat,self).__init__(**kwargs)
        self.key_fields = kwargs.get('keys', ['DISEASE/TRAIT','PUBMEDID'])
        self._set_samples_string_re()
    def _set_samples_string_re(self):
        import re
        self.sample_splitter = re.compile('controls, |cases, |individuals ')
    def _set_fields(self):
        from dtk.hgChrms import get_chrom_sizes
        from dtk.etl import latest_ucsc_hg_in_gwas
        self.chrom_sizes=get_chrom_sizes(latest_ucsc_hg_in_gwas())
        self.study_fields = [
                    'INITIAL SAMPLE SIZE',
                    'REPLICATION SAMPLE SIZE',
                    'PLATFORM [SNPS PASSING QC]',
                    'DATE'
                    ]
### These come from GRASP, and we want to stay in line with that
        self.study_header=[
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
        self.snp_fields = [
                  'SNP_ID_CURRENT', # may need to filter out CSV
                  'CHR_ID',
                  'CHR_POS',
                  'P-VALUE',
                 ]
        self.allele_freq = 'RISK ALLELE FREQUENCY'
        self.allele_nuc = 'STRONGEST SNP-RISK ALLELE'
    def _process_snp_data(self,k,frs):
        self._format_snp_data(frs)
        self.snp_data += [';'.join([
                                 frs[self.allele_nuc_ind][-1],
                                 frs[self.allele_freq_ind]
                                 ]),
                      'NA'
                     ]
        return self._finish_process_snp_data(k)
    def _set_inds(self, header):
        self.study_inds = [header.index(x) for x in self.study_fields]
        self.snp_inds = [header.index(x) for x in self.snp_fields]
        self.key_inds = [header.index(x) for x in self.key_fields]
        self.allele_freq_ind = header.index(self.allele_freq)
        self.allele_nuc_ind = header.index(self.allele_nuc)
        self.chr_ind = header.index('CHR_ID')
        self.pos_ind = header.index('CHR_POS')
    def _clean_k(self, frs):
        # This doesn't seem to have the same issues as GRASP:
        # - encoding is consistently utf8
        # - phenotypes aren't wrapped in quotes
        # - there are no odd characters to strip
        # So, just decode, replace spaces with underbars, and lowercase
        k = "|".join([frs[i]
                      for i in self.key_inds
                     ])
        return k.replace(" ", "_").lower()
    def _process_study_data(self, k,frs):
        study_data =  self._prep_study_data(frs)
        total, eths, ethnicity_breakdown = self._parse_sample_sizes(study_data[0:2])
        # the NAs are for fields present only in GRASP, but no here
        final_data = [total, eths] + study_data[2:4] + ['NA']*2 + ethnicity_breakdown
        self._finish_study_data(k,final_data)
#    def report(self, study_out):
#        with open(study_out, 'w') as f:
#            f.write("\t".join(["|".join(['Phenotype','PMID'])]+ self.study_header) + "\n")
#            for k in self.snp_studies:
#                f.write("\t".join([k.encode('utf8')] +
#                                [";".join(s) for s in self.studies[k]]
#                              ) + "\n")
    def _parse_sample_sizes(self, l):
        discov_count, discov_eths, discov_dets = self._parse_a_sample_size(l[0])
        valid_count, valid_eths, valid_dets = self._parse_a_sample_size(l[1])
        total = str(discov_count+valid_count)
        all_eths =  ", ".join(list(discov_eths & valid_eths))
        all_dets = [str(x) for x in discov_dets + valid_dets]
        return total, all_eths, all_dets
    def _parse_a_sample_size(self, sample_str):
        total = 0
        eths = set()
        dets = [0]*3
        samples = self.sample_splitter.split(sample_str)
        for s in samples:
            count, eth = self._parse_count_from_eth(s)
            eths.add(eth)
            total += count
            det_ind = self._get_det_ind(eth)
            if det_ind is not None:
                dets[det_ind] += count
        return total, eths, dets
    def _parse_count_from_eth(self, s):
        relevant = s.split('ancestry')[0].split(" ")
        try:
            cnt = int(relevant[0].replace(",", ""))
        except ValueError:
            return 0, ''
        eth = " ".join(relevant[1:]).lower()
        return cnt, eth
    def _get_det_ind(self, eth):
        # this may not catch everything, but this isn't vital info
        if any(x in eth for x in ['european', 'british', 'dutch']):
            return 0
        elif any(x in eth for x in ['african']):
            return 1
        elif any(x in eth for x in ['chinese', 'korean', 'japanese', 'east asian']):
            return 2
        return None

if __name__=='__main__':
    import argparse

    #=================================================
    # Read in the arguments/define options
    #=================================================
    arguments = argparse.ArgumentParser(description="Parse full GRASP DB into our format")
    arguments.add_argument("db", help="Full DB")
    arguments.add_argument("o", help="studies output file")
    arguments.add_argument("f", help="File to capture the failed SNPs")
    arguments.add_argument("p", help="too many phenotypes pmid output file")
    args = arguments.parse_args()
    pg = parse_gwascat(db=args.db, failed_snps_file=args.f)
    pg.run(encoding='utf8')
    pg.report(args.o, args.p)
