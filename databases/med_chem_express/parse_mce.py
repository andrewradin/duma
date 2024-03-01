#!/usr/bin/env python3

from __future__ import unicode_literals
from atomicwrites import atomic_write
from openpyxl import Workbook, load_workbook
import logging
logger = logging.getLogger(__name__)


def sheet_values(sheet):
    out = []
    for row in sheet.rows:
        row_values = [cell.value for cell in row]
        out.append(row_values)
    return out


def parse(input_file, seen):
    wb = load_workbook(input_file)
    info_sheet = wb['Compound Information']
    data = sheet_values(info_sheet)
    id_col = data[0].index('Catalog Number')
    col_map = [
            ('Product Name', 'canonical'),
            ('CAS Number', 'cas'),
            ('Synonyms', lambda x: [('synonym', v) for v in x.split(';') if v]),
            ]

    out = []
    for row in data[1:]:
        id = row[id_col]
        if id in seen:
            continue
        seen.add(id)

        for col_name, out_type in col_map:
            idx = data[0].index(col_name)
            assert idx != -1, "Couldn't find %s in %s" % (col_name, data[0])
            if isinstance(out_type, str):
                out_fn = lambda x: [(out_type, x)]
            else:
                out_fn = out_type

            val = row[idx]
            for attr, attrval in out_fn(val):
                if attrval.strip() != '':
                    out.append((id, attr, attrval.strip()))
    return out

def run(input_files, output_file):
    all_data = [['med_chem_express_id', 'attribute', 'value']]
    seen = set()
    for input_file in input_files:
        new_data = parse(input_file, seen)
        logger.info("Adding %d rows from %s", len(new_data), input_file)
        all_data.extend(new_data)
    with atomic_write(output_file, overwrite=True) as f:
        for row in all_data:
            f.write('\t'.join(row) + "\n")

if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    setupLogging()
    import argparse
    arguments = argparse.ArgumentParser(description="Parses out a medChemExpress xlsx file.")
    
    arguments.add_argument('-o', '--output', help="Where to write the output")
    arguments.add_argument('inputs', nargs='+', help="Input files")
    
    args = arguments.parse_args()

    run(args.inputs, args.output)

