#!/usr/bin/env python3

from dtk.tsv_alt import SqliteSv

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='Convert a tsv file to an alternative format'
            )
    parser.add_argument('-i', '--input', help="Input .tsv file")
    parser.add_argument('-o', '--output', help="Output new file")
    parser.add_argument('--header', help="Space-separated column names; don't specify if first line is header")
    parser.add_argument('--index', help="Space-separated column names to index; if not specified, all are indexed")
    parser.add_argument('coltypes', choices=('int', 'float', 'str'), nargs='+', help="Column types")
    args=parser.parse_args()
    from pydoc import locate
    types = [locate(type) for type in args.coltypes]
    header = args.header.split(' ') if args.header is not None else None
    index = args.index.split(' ') if args.index is not None else None

    SqliteSv.write_from_tsv(args.output, args.input, types, header, index)
