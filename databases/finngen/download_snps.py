#!/usr/bin/env python
import logging
logger = logging.getLogger(__name__)

"""
Downloads the finngen data with some minimal pval filtering to make it more manageable.

On the UKBB side we do this with grep and parallel, but the p-values aren't as nicely structured here.
Could do it with awk (awk -F'\t'  '{if ($7<0.01)print $0}'), but doesn't end up being meaningfully faster, and you lose the progress bar.
"""

def download(url, out_dir, threshold):
    import requests
    import io
    import isal.igzip as gzip
    import os
    from contextlib import closing
    r = requests.get(url, stream=True) 

    fn = os.path.basename(url)
    out_fn = os.path.join(out_dir, fn)

    pval_idx = None
    header = None
    with gzip.open(out_fn, 'wt') as out, closing(r):
        #for line in io.TextIOWrapper(r.raw):
        for line in io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw)):
            parts = line.split('\t')
            if header is None:
                header = parts
                pval_idx = header.index('pval')
                out.write(line)
                continue
            
            if float(parts[pval_idx]) <= threshold:
                out.write(line)

def run(input, output, pval):
    from dtk.files import get_file_records
    header = None
    url_idx = None
    name_idx = None

    datas = []
    logger.info("Processing manifest")
    for rec in get_file_records(input, keep_header=True, progress=True):
        if header is None:
            header = rec
            url_idx = rec.index('path_https')
            name_idx = rec.index('name')
            continue

        datas.append((rec[url_idx], rec[name_idx]))

    urls = [x[0] for x in datas]
    logger.info(f"Downloading {len(urls)} urls")
    from dtk.parallel import pmap
    list(pmap(download, urls, static_args=dict(out_dir=output, threshold=pval), progress=True))


if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="")
    arguments.add_argument("-i", "--input", required=True, help='Manifest file')
    arguments.add_argument("-o", "--output", required=True, help="Output directory")
    arguments.add_argument("-p", "--pval", type=float, required=True, help="Filter pvalue")
    args = arguments.parse_args()
    from dtk.log_setup import setupLogging
    setupLogging()
    run(**vars(args))
