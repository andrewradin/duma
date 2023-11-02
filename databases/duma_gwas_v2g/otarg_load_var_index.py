#!/usr/bin/env python

from pyarrow import parquet as pq
from tqdm import tqdm
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--downloaded-data-dir", help="Input variants file with chr# and positions.")
    args = parser.parse_args()
    all_rsids = set()
    for fn in tqdm(os.listdir(args.downloaded_data_dir)):
        fn = os.path.join(args.downloaded_data_dir, fn)
        if not fn.endswith('.parquet'):
            continue
        data = pq.read_table(fn, columns=["rs_id"]).to_pydict()
        all_rsids.update([x for x in data['rs_id'] if x])
    
    print('\n'.join(all_rsids))