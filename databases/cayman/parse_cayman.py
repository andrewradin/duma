#!/usr/bin/env python3


from __future__ import print_function
from atomicwrites import atomic_write
import dtk.rdkit_2019
from rdkit import Chem


def parse(input_file):
    print(("Opening ", input_file))
    out = [['cayman_id', 'attribute', 'value']]

    seen = set()

    # This file appears to be encoded via ISO-8859-1, but rdkit will by
    # default try to load it as utf8.  So let's convert for it.
    import gzip
    opener = gzip.open if input_file.endswith('.gz') else open
    missing_itm_ns = 0
    with opener(input_file, 'rt', encoding='ISO-8859-1') as f:
        file_data = f.read()
        import io
        utf8_f = io.BytesIO(file_data.encode('utf8'))

        for mol in Chem.ForwardSDMolSupplier(utf8_f, sanitize=False):
            if not mol:
                continue
            data = mol.GetPropsAsDict()
            if 'Item number' not in data:
                missing_itm_ns+=1
                continue
            id = 'CAY' + str(data['Item number'])
            if id in seen:
                # There are duplicates in the file, skip them.
                continue
            seen.add(id)

            if not "Item name" in data:
                # We can't possibly match these drugs because none of them
                # have useful information right now, just skip them.
                print(("Skipping nameless data", data))
                continue

            name = str(data['Item name'])

            if not name.strip():
                continue

            entry = [
                ('canonical', name),
            ]
            # RDKit gets angry if you try to canonicalize some of these,
            # probably because we didn't sanitize them.
            smiles = Chem.MolToSmiles(mol, canonical=False)
            if smiles:
                entry.append(('smiles_code', smiles))

            # Apparently some mols have valid smiles without valid inchi.
            try:
                inchi = Chem.MolToInchi(mol)
                if inchi:
                    entry.append(('inchi', inchi))
                    entry.append(('inchi_key', Chem.MolToInchiKey(mol)))
            except ValueError:
                # This usually happens because sanitization fails, can't turn
                # it off for inchi.
                pass

            if 'CAS Number' in data and data['CAS Number'].strip():
                entry.append(('cas', data['CAS Number']))


            for attr, attrval in entry:
                out.append((id, attr, attrval.strip()))
        print(f'{missing_itm_ns} entries were skipped for lacking an "Item number"')
        return out

def run(input_file, output_file):
    data = parse(input_file)
    with atomic_write(output_file, overwrite=True) as f:
        for row in data:
            if len(row) != 3:
                raise Exception(f'Length of {row} was {len(row)}')
            f.write('\t'.join(row) + "\n")

if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="Parses out a cayman sdf file.")
    
    arguments.add_argument('-o', '--output', help="Where to write the output")
    arguments.add_argument('input', help="Input file")
    
    args = arguments.parse_args()

    run(args.input, args.output)

