#!/usr/bin/env python
import logging
logger = logging.getLogger(__name__)

from dtk.files import get_file_records

from do_vcf_subset import rschr_to_chrnum


def parse_info(info):
    """

    Sample info looks something like:
        RS=12138618;dbSNPBuildID=120;SSR=0;VC=SNV;GNO;FREQ=GnomAD:0.9633,0.03673,.|GoNL:0.9639,0.03607,.|KOREAN:0.9456,0.05441,0;COMMON
    
    We're just parsing out the freq data for now.

    Freq data will have a frequency value for each allele, starting with the reference allele.
    """
    try:
        info_parts = info.split(';')
        info_map = {}
        for info_part in info_parts:
            if '=' not in info_part:
                continue
            key, val = info_part.split('=', maxsplit=1)
            info_map[key] = val
        
        if 'FREQ' not in info_map:
            return None

        freq = info_map['FREQ']
        freq_parts = freq.split('|')

        out = None
        import numpy as np

        # NOTE: You can get some weird allele frequencies out of this, sometimes even 1.0
        # There are apparently many known cases of the "reference" genome actualy being a variant,
        # and so the "effect" allele actually is way more common.
        # e.g. https://www.ncbi.nlm.nih.gov/snp/rs215
        # Also see https://www.biostars.org/p/282029/#282033 for similar discussion.
        for freq_part in freq_parts:
            src, alfs = freq_part.split(':')
            alf_parts = alfs.split(',')
            alf_parts = [float(x) if x != '.' else 0 for x in alf_parts]
            
            if out is None:
                out = alf_parts
            else:
                # For each allele, take the maximum reported value.
                # This will add up to >100, but that's fine, looking for any reason to keep.
                out = np.maximum(out, alf_parts)
    except:
        print("Failed on", info) 
        raise
    return out


def make_vcf_maps(vcf_file):
    """
    Builds maps of vcf data, by rsid and chr:pos
    """
    out_rs = {}
    out_chrpos = {}
    for rec in get_file_records(vcf_file, progress=True, parse_type='tsv'):
        if rec[0][0] == '#':
            continue
        
        rschrm, pos, rsid, ref, alt, qual, filter, info = rec

        assert rsid[:2] == 'rs'
        rsid = rsid[2:]

        if rschrm not in rschr_to_chrnum:
            continue
        chrm = rschr_to_chrnum[rschrm]

        alt_freqs = parse_info(info)
        if alt_freqs is None:
            # No freq data.
            continue

        alts = alt.split(',')

        assert len(alts) + 1 == len(alt_freqs), f"Mismatched alts with {alts} {alt_freqs} - {info}"
        if len(alts) == 0:
            # Nothing we can do with this one, no allele or maf.
            continue
        best_freq_and_alt = max(zip(alt_freqs[1:], alts))

        chrpos = chrm + ':' + pos

        data = (chrpos, rsid, ref, best_freq_and_alt)

        if rsid:
            out_rs[rsid] = data
        
        out_chrpos[chrpos] = data
    
    return out_rs, out_chrpos
        
def run(output, fail_files, vcf_file, studies_file, filter_file):
    rs_map, chrpos_map = make_vcf_maps(vcf_file)
    
    from dtk.gwas_filter import gwas_filter
    filter = gwas_filter(log=filter_file)

    from dtk.gwas_filter import get_nsamples
    nsamples = get_nsamples(studies_file)

    from collections import defaultdict
    stats = defaultdict(int)

    def process_fail_file(out, fn):
        for rec in get_file_records(fn, progress=True):
            study, rs_id, chr_num, chr_pos, p_val = rec[:5]

            chrnumpos = chr_num + ':' + chr_pos

            if rs_id in rs_map:
                data = rs_map[rs_id]
            elif chrnumpos in chrpos_map:
                data = chrpos_map[chrnumpos]
            else:
                stats['not_found'] += 1
                continue
            
            # TODO: THis part isn't too bad, but then have to figure out where to store ref
            # If we don't store ref, then we can't link to otarg.
            vcf_chrpos, vcf_rsid, vcf_ref, best_freq_and_alt = data
            maf, allele = best_freq_and_alt

            allele_maf = allele + ';' + str(maf)

            rs_id = vcf_rsid
            
            qc_check = [rs_id, chr_num, chr_pos, p_val, allele_maf]
            if not study in nsamples:
                stats['missing_study'] += 1
                continue
            
            if not filter.qc_snp(qc_check, nsamples[study], study):
                stats['filtered'] += 1
                continue
                
            stats['rescued'] += 1
            out.write('\t'.join([study, rs_id, chr_num, chr_pos, p_val, allele_maf] + rec[6:]) + '\n')

    from atomicwrites import atomic_write
    with atomic_write(output, overwrite=True) as out:
        for fail_file in fail_files:
            process_fail_file(out, fail_file)

    filter.report()

    print(dict(stats))
            
if __name__=='__main__':
    import argparse
    arguments = argparse.ArgumentParser(description="")
    arguments.add_argument("-o", '--output', help="Output SNP data filename(s)")
    arguments.add_argument("-v", '--vcf-file', help="Input vcf file")
    arguments.add_argument("-f", '--filter-file', help="Output filtered data to this file")
    arguments.add_argument("-s", '--studies-file', help="Studies file (for n samples)")
    arguments.add_argument("fail_files", nargs='*', help="Input variant fail file(s)")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))
