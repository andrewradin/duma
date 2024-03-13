
import io
import sys
from dtk.files import get_file_records

def pmid_from_k(k, k_sep = "|"):
    return k.split(k_sep)[1]

def k_is_good(k):
# for now the only thing that makes a k bad is a non-numeric PMID
# exception: 'ukbb_gwas_imputedv3'
# this is originally set in the ukbb Makefile -study flag for parse_ukbb.py

    # This is most of our data, get rid of this first.
    try:
        if 'ukbb' in k or 'finngen' in k:
            return True
    except TypeError:
        pass

    try:
        temp = int(pmid_from_k(k))
        return True
    except ValueError:
        return False

class gwas_filter(object):
    def __init__(self,log=None):
        from dtk.hgChrms import get_chrom_sizes
        from dtk.etl import latest_ucsc_hg_in_gwas
        self.chrom_sizes=get_chrom_sizes(latest_ucsc_hg_in_gwas())
        self.bad_maf=set()
        if log:
            self.log_fn = log
        else:
            self.log_fn = False
        self.log = []
    def write_snps(self, snp_file, out_file, study_file):
        nsample_dict = get_nsamples(study_file)
        with io.open(out_file, 'w',encoding='utf-8',errors='replace') as f:
            for frs in get_file_records(snp_file,parse_type='tsv'):
	    	#frs[1:].decode("utf8")
                if self.qc_snp(frs[1:],nsample_dict[frs[0]],frs[0]):
                    s ="\t".join(frs)+'\n'
                    f.write(s.decode('utf8', errors='replace'))
    def report(self):
        if self.log_fn:
            with io.open(self.log_fn,'w',encoding='utf-8',errors='replace') as f:
                s = '\n'.join(self.log) + '\n'
                f.write(s)
        else:
            sys.stderr.write('\n'.join(self.log)+'\n')
    def qc_snp(self, snp_data, nsamples, k):
        if (
### I disabled this so we can run QC with all of the SNPs
### But only those with associated Uniprots will be used for predictions
            # snp_data[self.snp_fields.index('InGene')] != '' and
            self._check_position(snp_data, k) and
            self._check_snp_stats(snp_data, nsamples, k)
           ):
            return True
### This collectively filtered only 4 records (seemingly from chrMT),
    def _check_position(self, snp_data, k):
        chrom = snp_data[1]
        if chrom in self.chrom_sizes.keys():
            try:
                if not (int(snp_data[2]) <= self.chrom_sizes[chrom]):
                    self._append_to_log(k,snp_data,'failed the size-position check: too big',snp_data[2])
                    return False
                else:
                    return True
            except ValueError:
                self._append_to_log(k,snp_data,'failed the int-position check: non integer position',snp_data[2])
                return False
        else:
            self._append_to_log(k,snp_data,'failed the chr-position check: chromosome name',chrom)
            return False
    def _check_snp_stats(self, snp_data, nsamples, k, mac=25, minMAF=0.0):
        if self._check_p(snp_data,k) and self._check_maf(snp_data, nsamples, k, minMAF, mac):
            return True
### TODO add a maximal p-val filter here
    def _check_p(self,snp_data,k):
        try:
            float(snp_data[3])
            return True
        except ValueError:
            pass
        self._append_to_log(k,snp_data,'failed the p-value check',snp_data[3])
        return False
    def _check_maf(self, snp_data, nsamples, k, minMAF, mac):
        amaf_col=4
        try:
            allele,maf = snp_data[amaf_col].split(";")
            n = float(nsamples)
            if allele.lower() not in ['a', 'c', 't', 'g']:
                self._append_to_log(k,snp_data,'failed the allele check:',allele)
                return False
            if float(maf) < minMAF:
                self._append_to_log(k,snp_data,'failed the minMAF check:',maf)
                return False
            if float(maf)*n/2 <= mac:
                # mac is minor allele count. Ideally, we'd know how many cases and controls there were
                # then we'd check that the (number cases & number controls) * MAF >= some minimum
                # The Neale lab (ukbb) recommends 25
                self._append_to_log(k,snp_data,'failed the minMAC check:',n)
                return False
            return True
        except (ValueError, IndexError):
            self._append_to_log(k,snp_data,'failed the format-MAF check:',snp_data[amaf_col])
            return False
        self._append_to_log(k,snp_data,'failed the MAF check',"ERROR")
    def _append_to_log(self,k,snp_data,message,info):
        toappend="\t".join([k,"\t".join(snp_data),message,str(info)])
        self.log.append(toappend)
def get_nsamples(ifile):
    nsamples=dict()
    for frs in get_file_records(ifile):
        nsamples[frs[0]]=frs[1]
    return nsamples