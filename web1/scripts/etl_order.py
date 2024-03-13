#!/usr/bin/env python3

program_description='''\
Provide a rough equivalent of the ETL Order page on the command line.
'''

def show_order():
    from dtk.etl import get_all_versioned_etl, order_etl
    all_etl = get_all_versioned_etl()
    etl_groups=[
            sorted(subset,key=lambda x:x.name)
            for subset in order_etl(all_etl.values())
            ]
    rows = [[
            'Wave',
            'Source',
            'Last update',
            'Source version',
            'Update every...',
            'Exceptions',
            ]]
    for i,wave in enumerate(etl_groups):
        for j,src in enumerate(wave):
            row = [str(i+1) if j == 0 else '']
            row.append(src.name)
            row.append(src.published)
            row.append(str(src.source_version))
            if src.months_between_updates is None:
                row.append('Never')
            else:
                row.append(f'{src.months_between_updates} months')
            if src.last_check:
                row.append(f'{src.last_check[0]}:\n{src.last_check[1]}')
            else:
                row.append('')
            rows.append(row)
    from dtk.text import wrap,split_multi_lines,print_table
    wrap(rows,5,50)
    print_table(split_multi_lines(rows),header_delim='-')

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=program_description,
            )
    args = parser.parse_args()
    show_order()
