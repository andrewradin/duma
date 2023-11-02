#!/usr/bin/env python
import os

def make_moa_key_parts(key, uni2gene):
    def dir_char(x):
        x = int(x)
        if x == 1:
            return '+'
        elif x == 0:
            return '?'
        elif x == -1:
            return '-'
        raise Exception(f"Unexpected val {x}")
    
    def prot_to_gene(prot):
        return uni2gene.get(prot, prot)

    if isinstance(key[0], tuple):
        key = [prot_to_gene(x[0]) + dir_char(x[1]) for x in key]
    else:
        key = [prot_to_gene(x) for x in key]
    
    key = sorted(key)
    return key
    
def make_moa_key(key, uni2gene, max_parts):
    def hashify(tail_keys):
        N = len(tail_keys)
        import hashlib
        import base64
        hash = hashlib.sha1(str(tail_keys).encode('utf8'))
        hashval = hash.hexdigest().upper()[:6]
        return f'+{N}_{hashval}'

    key = make_moa_key_parts(key, uni2gene)
    if max_parts and len(key) > max_parts:
        cutoff = max_parts - 2
        key = key[:cutoff] + [hashify(key[cutoff:])]
    return 'MOA_' + '_'.join(key)

def make_u2g(uni_fn):
    out = {}
    from dtk.files import get_file_records
    for prot, _, gene in get_file_records(uni_fn, progress=True, select=(['Gene_Name'],1), keep_header=False):
        out[prot] = gene
    return out


def make_attrs(dpikey, moa, u2g):
    canonical = ' '.join(make_moa_key_parts(moa, u2g))
    if len(canonical) > 150:
        canonical = f'{canonical[:120]}... ({len(moa)} parts)'
    return [
        (dpikey, 'canonical', canonical),
    ]

def transform(in_dpi, collapse_dir, rekey, u2g, merge=None):
    from dtk.data import assemble_attribute_records

    seen = set()
    seen_moakey = {}
    from tqdm import tqdm

    DPI_THRESH=0.5

    out_dpi = []
    out_attrs = []
    for molkey, prot_ints in tqdm(list(assemble_attribute_records(in_dpi))):
        # We should have only 1 record per prot, collapse the set.
        prot_ints = {k:next(iter(v)) for k,v in prot_ints.items()}

        if collapse_dir:
            # Key is just the interacting prots
            key = tuple(prot for prot, (_, _, ev, direction) in sorted(prot_ints.items()) if float(ev) >= DPI_THRESH)
        else:
            key = tuple((prot, direction) for prot, (_, _, ev, direction) in sorted(prot_ints.items()) if float(ev) >= DPI_THRESH)

        if key in seen or not key:
            continue
        seen.add(key)

        if rekey:
            molkey = make_moa_key(key, u2g, max_parts=6)
            assert molkey not in seen_moakey, f"Collision, no good, seen {molkey} for {key} vs {seen_moakey[molkey]}"
            seen_moakey[molkey] = key

        for prot, (_, _, ev, direction) in prot_ints.items():
            if float(ev) >= DPI_THRESH:
                if collapse_dir:
                    out = [molkey, prot, '1', '0']
                else:
                    out = [molkey, prot, '1', direction]
                out_dpi.append(out)
        
        if rekey:
            out_attrs.extend(make_attrs(molkey, key, u2g))
    
    if merge:
        # Merge in anything from old dpi file that isn't in the new one.
        merged_moakey = set()
        from dtk.files import get_file_records
        for rec in get_file_records(merge, keep_header=False, progress=True):
            merged_moakey.add(rec[0])
            if rec[0] not in seen_moakey:
                out_dpi.append(rec)
        
        print(f"Previous had {len(merged_moakey)} moas, new has {len(seen_moakey)}")
        print(f"Merge adds in {len(merged_moakey - seen_moakey.keys())} that would have otherwise dropped")
        print(f"Ends with {len(merged_moakey | seen_moakey.keys())} moas")
    
    if rekey:
        return out_dpi, out_attrs
    else:
        return out_dpi

def transform_file(dpi_fn, out_fn, collapse_dir, rekey, u2g, attrs_out, merge=None):
    from dtk.files import get_file_records
    rows = get_file_records(dpi_fn, keep_header=True, progress=True)
    header = next(rows)

    results = transform(rows, collapse_dir, rekey, u2g, merge)
    if rekey:
        dpi, attrs = results
    else:
        dpi, attrs = results, None

    from atomicwrites import atomic_write
    if dpi:
        with atomic_write(out_fn, overwrite=True) as out_f:
            out_f.write('\t'.join(header) + '\n')
            for out in dpi:
                out_f.write('\t'.join(out) + '\n')
    
    if attrs:
        attr_header = ['moa_id', 'attribute', 'value']
        with atomic_write(attrs_out, overwrite=True) as out_f:
            out_f.write('\t'.join(attr_header) + '\n')
            for out in attrs:
                out_f.write('\t'.join(out) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='create transformed dpi file')
    parser.add_argument('--collapse-dir', action='store_true', help="If true, removes directionality from MoAs")
    parser.add_argument('-i', '--input', nargs='+', help="input filenames")
    parser.add_argument('-o', '--output', nargs='+', help="Output file names")
    parser.add_argument('-u', '--uniprot', help="Uniprot data/gene name file")
    parser.add_argument('-v', '--version', help='vXX, the version string')
    parser.add_argument('-a', '--attrs', help="attrs output")
    parser.add_argument('--rekey', action='store_true', help="If true, generates MoA-based keys instead of exemplars")
    # We have the option to merge in previous moa entries.
    # 
    parser.add_argument('--merge-prev', action='store_true', help="Includes DPI entries from specified version in addition to this one")
    args = parser.parse_args()


    u2g = make_u2g(args.uniprot)
    ver = f'.{args.version}'
    for input in args.input:
        if input.endswith('.sqlsv'):
            continue

        
        out = input.replace(ver, f'-moa{ver}')

        merge_fn = None
        if args.merge_prev:
            prev_ver = f'{ver[:2]}{int(ver[2:]) - 1}'
            from dtk.s3_cache import S3File, S3Bucket
            s3f = S3File(S3Bucket('matching'), os.path.basename(out.replace(ver, prev_ver)))
            s3f.fetch()
            merge_fn = s3f.path()
            print(f"Merging in previous version {prev_ver} at {merge_fn} to create {out}")
        print(f"Writing {out} from {input}")

        assert out in args.output, f"Missing {out} from {args.output}"
        transform_file(input, out, args.collapse_dir, args.rekey, u2g=u2g, attrs_out=args.attrs, merge=merge_fn)

        from dtk.tsv_alt import SqliteSv
        assert out.endswith('.tsv'), f"Unexpected output without .tsv {out}"
        sql_outf = out.replace('.tsv', '.sqlsv')
        SqliteSv.write_from_tsv(sql_outf, out, [str, str, float, int])
