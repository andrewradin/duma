#!/usr/bin/env python

def get_chrom_sizes(ver):
    return _get_hg_file('chrom', ver)
def get_prot_txn_sizes(ver):
    return _get_hg_file('prot_txn', ver)

def _get_hg_file(name, ver):
    import os
    from path_helper import PathHelper
    from dtk.s3_cache import S3File
    s3_file = S3File.get_versioned(
                'ucsc_hg',
                'v'+str(ver),
                role=name+'_sizes'
                )
    s3_file.fetch()
    ifile = s3_file.path()
    from dtk.files import get_file_records
    return {frs[0]:float(frs[1]) for frs in get_file_records(ifile)}

class linear_hgChrms(object):
    def __init__(self,ver):
        self.letter_map={'X':'23',
                         'Y':'24',
                         'M':'25'
                        }
        chrom_sizes = get_chrom_sizes(ver)
        self.chrom_sizes = {self.get_chrm_index(k):v
                            for k,v in chrom_sizes.items()
                           }
        self._setup_linear_genome()
    def _setup_linear_genome(self):
        self.cumulative_chrm_sizes = {}
        running_total = 0
        for chr in sorted(self.chrom_sizes):
            self.cumulative_chrm_sizes[chr] = running_total
            running_total += self.chrom_sizes[chr]
    def get_linear_pos(self, tup):
        chr,pos=tup
        return self.cumulative_chrm_sizes[self.get_chrm_index(chr)]+int(pos)
    def get_chrm_index(self, chr):
        if chr in self.letter_map:
            return int(self.letter_map[chr])
        return int(chr)
    def check_position(self, chr, pos):
        chr = self.get_chrm_index(chr)
        if chr in list(self.chrom_sizes.keys()):
            try:
                if (int(pos)
                    <= self.chrom_sizes[chr]):
                    return True
            except ValueError:
                pass
