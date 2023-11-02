#!/usr/bin/env python
import logging
logger = logging.getLogger(__name__)

# Sourced from https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/latest_assembly_versions/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_assembly_report.txt
# Also available at the bottom of https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.39
# This github repo also seems to keep track - https://github.com/dpryan79/ChromosomeMappings/blob/master/GRCh38_RefSeq2UCSC.txt
mapping = [
    ("NC_000001.11", "1"),
    ("NC_000002.12", "2"),
    ("NC_000003.12", "3"),
    ("NC_000004.12", "4"),
    ("NC_000005.10", "5"),
    ("NC_000006.12", "6"),
    ("NC_000007.14", "7"),
    ("NC_000008.11", "8"),
    ("NC_000009.12", "9"),
    ("NC_000010.11", "10"),
    ("NC_000011.10", "11"),
    ("NC_000012.12", "12"),
    ("NC_000013.11", "13"),
    ("NC_000014.9","14"),
    ("NC_000015.10", "15"),
    ("NC_000016.10", "16"),
    ("NC_000017.11", "17"),
    ("NC_000018.10", "18"),
    ("NC_000019.10", "19"),
    ("NC_000020.11", "20"),
    ("NC_000021.9", "21"),
    ("NC_000022.11", "22"),
    ("NC_000023.11", "X"),
    ("NC_000024.10", "Y"),
]

rschr_to_chrnum = {a:b for a,b in mapping}
chrnum_to_rschr = {b:a for a,b in mapping}

def make_coords(variants_fn):
    from dtk.files import get_file_records
    out = set()
    missing = set()
    for rec in get_file_records(variants_fn, keep_header=None):
        study, rsid, chrnum, chrpos = rec[:4]

        if chrnum not in chrnum_to_rschr:
            missing.add(chrnum)
            continue
        rschr = chrnum_to_rschr[chrnum]
        chrpos = int(chrpos)
        out.add(f'{rschr}\t{chrpos}')
        if rsid:
            out.add(f'rs{rsid}')
    
    if missing:
        logger.info(f"Missing lookups for: {len(missing)}: (samples) {list(missing)[:10]}")
    logger.info(f"Searching for {len(out)} tokens from {variants_fn}")
    return out

def run(output, variant_files, vcf_file):
    from atomicwrites import atomic_write
    coords = set()
    for f in variant_files:
        coords.update(make_coords(f))

    import tempfile
    with tempfile.NamedTemporaryFile() as temp:
        temp_fn = temp.name
        with atomic_write(temp_fn, overwrite=True) as out:
            for coord in sorted(coords):
                out.write(coord + '\n')
        

        import subprocess
        import time
        done = False
        def watch():
            while not done:
                time.sleep(5)
                subprocess.call(['./rs_to_vcf_monitor.sh'])


        from threading import Thread
        t = Thread(target=watch)
        t.daemon = True
        t.start()

        subprocess.check_call(['./rs_to_vcf.sh', temp_fn, vcf_file, output])
            
if __name__=='__main__':
    import argparse
    arguments = argparse.ArgumentParser(description="")
    arguments.add_argument("-o", '--output', help="Output SNP data filename(s)")
    arguments.add_argument("-v", '--vcf-file', help="Input vcf file")

    arguments.add_argument("variant_files", nargs='*', help="Input variant file(s)")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()

    run(**vars(args))
