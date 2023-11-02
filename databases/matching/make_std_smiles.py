#!/usr/bin/env python3

from dtk.standardize_mol import standardize_mol
import os
import sys
import dtk.rdkit_2019
from rdkit import Chem
import traceback
import tqdm
from dtk.tsv_alt import SqliteSv
from path_helper import PathHelper

# We cache standard smiles codes, so make sure that if you change how they are
# computed, you update the version so we don't return obsolete cached values.
STDSM_VERSION = 2

def gen_std_smiles(id_and_mol_data, max_tautomers,isomeric):
    id, mol_data = id_and_mol_data

    # We use inchi as the starting point instead of SMILES, where available.
    # This tends to be more consistent and works around a class of tautomerization
    # issues.
    try:
        if 'inchi' in mol_data:
            mol = Chem.MolFromInchi(mol_data['inchi'])
            if mol:
                clean_mol = standardize_mol(mol, max_tautomers=max_tautomers)
                clean_smiles = Chem.MolToSmiles(
                        clean_mol,
                        isomericSmiles = isomeric,
                        )
                return (id, clean_smiles)
    except Exception as e:
        print("FAILED to standardize smiles via inchi for %s" % id)
        traceback.print_exc()

    try:
        if 'smiles_code' in mol_data:
            mol = Chem.MolFromSmiles(mol_data['smiles_code'])
            # Bouncing through inchi makes things more consistent.
            mol = Chem.MolFromInchi(Chem.MolToInchi(mol))
            if mol:
                clean_mol = standardize_mol(mol, max_tautomers=max_tautomers)
                clean_smiles = Chem.MolToSmiles(
                        clean_mol,
                        isomericSmiles = isomeric,
                        )
                return (id, clean_smiles)
    except Exception as e:
        print("FAILED to standardize smiles via smiles for %s" % id)
        traceback.print_exc()

    # Ran out of options, return None.
    return (None, None)

class Cache:
    def __init__(self, fn):
        if not os.path.exists(fn):
            header = ['smiles_code', 'inchi_code', 'stdsmiles']
            types = [str, str, str]
            SqliteSv.write_from_data(fn, [], types, header)
        self.cache = SqliteSv(fn)

    def update_cache(self, id_to_clean_smiles, moldata):
        records = []
        for id, stdsmiles in id_to_clean_smiles.items():
            orig_smiles = moldata[id].get('smiles_code', None)
            orig_inchi = moldata[id].get('inchi', None)
            records.append((orig_smiles, orig_inchi, stdsmiles))
        self.cache.insert(records)

    def check_cache(self, moldata):
        smiles = moldata.get('smiles_code', None)
        if smiles:
            records = list(self.cache.get_records(
                        columns=['stdsmiles'],
                        filter__smiles_code__eq=smiles,
                        ))
            if records:
                return records[0][0]

        inchi = moldata.get('inchi_code', None)
        if inchi:
            records = self.cache.get_records(
                        columns=['inchi'],
                        filter__inchi_code__eq=inchi,
                        )
            if records:
                return records[0][0]
        return None

def main(in_fn, out_fn, max_tautomers, cores, isomeric):
    from collections import defaultdict
    mols = defaultdict(dict)

    print("Collecting existing smiles")
    with open(in_fn) as f:
        for line in f:
            id, attr, val = line.strip().split('\t')
            if attr == 'smiles_code':
                mols[id]['smiles_code'] = val
            if attr == 'inchi':
                mols[id]['inchi'] = val

    id_to_smiles = {}

    cache_fn = os.path.join(PathHelper.storage, f'stdsmiles_cache.{STDSM_VERSION}-{isomeric}-{max_tautomers}.sqlsv')
    print(f"Using cache at {cache_fn}")
    cache = Cache(cache_fn)

    # Check if we have cached smiles for any of these.
    for mol_id, moldata in list(mols.items()):
        cached_smiles = cache.check_cache(moldata)
        if cached_smiles:
            del mols[mol_id]
            id_to_smiles[mol_id] = cached_smiles
    print(f"Found {len(id_to_smiles)} from the cache")
    print("Cleaning smiles")
    from multiprocessing import Pool
    from functools import partial
    pool = Pool(cores)
    func = partial(gen_std_smiles, max_tautomers=max_tautomers, isomeric=isomeric)
    id_and_clean_smiles = tqdm.tqdm(
            pool.imap_unordered(func, mols.items(), chunksize=1),
            total=len(mols),
            smoothing=0,
            )

    new_id_to_smiles = dict(id_and_clean_smiles)

    print(f"Updating the cache with {len(new_id_to_smiles)} items")
    cache.update_cache(new_id_to_smiles, mols)

    id_to_smiles.update(new_id_to_smiles)

    print("Merging and outputting create file")
    prev_id = None
    from atomicwrites import atomic_write
    with atomic_write(out_fn, overwrite=True) as f:
        with open(in_fn) as in_f:
            for line in in_f:
                id, attr, val = line.strip('\n').split('\t')
                if id != prev_id and prev_id and prev_id in id_to_smiles:
                    std_smiles = id_to_smiles[prev_id]
                    f.write('%s\t%s\t%s\n' % (prev_id, 'std_smiles', std_smiles))
                prev_id = id
                f.write(line)
                if not line.endswith('\n'):
                    f.write('\n')

        if prev_id and prev_id in id_to_smiles:
            std_smiles = id_to_smiles[prev_id]
            f.write('%s\t%s\t%s\n' % (prev_id, 'std_smiles', std_smiles))


if __name__ == "__main__":
    import argparse, time
    import multiprocessing
    parser = argparse.ArgumentParser(description='Inserts standardized smiles into a create file')
    parser.add_argument("-i", "--input", help="Input create file")
    parser.add_argument("-o", "--output", help="Output create file")
    parser.add_argument("-m", "--max-tautomers", type=int, default=1000, help="Max # of tautomers to enumerate for each drug")
    parser.add_argument("--isomeric", help="Preserve isomers",action='store_true')
    parser.add_argument("-r", "--remote", help="Run on remote machine")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Number of cores alloted")
    args=parser.parse_args()

    if args.remote:
        import aws_op
        import time
        mch = aws_op.Machine.name_index[args.remote]
        remote_fn = f'/tmp/std_smiles.{time.time()}.in'
        remote_out_fn = f'/tmp/std_smiles.{time.time()}.out'
        mch.copy_to(args.input, remote_fn)
        iso_flag = '--isomeric' if args.isomeric else ''
        mch.run_remote_cmd(f"2xar/twoxar-demo/databases/matching/make_std_smiles.py -i {remote_fn} -o {remote_out_fn} -m {args.max_tautomers} {iso_flag}")
        local_out_fn = './tmp.out.tsv'
        mch.copy_from(remote_out_fn, local_out_fn)
        import os
        os.rename(local_out_fn, args.output)
    else:
        main(args.input, args.output, args.max_tautomers, args.cores, args.isomeric)
