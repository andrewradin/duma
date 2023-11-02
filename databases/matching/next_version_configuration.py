#!/usr/bin/env python3

from dtk.text import indent

default_sources=[
        'chembl',
        'drugbank',
        'duma',
        'ncats',
        'bindingdb',
        'med_chem_express',
        'selleckchem',
        'cayman',
        'pubchem',
        'lincs',
        'globaldata',
        ]

def matching_config(
        next_ver,
        source_versions,
        dpimerge_outputs,
        combo_dpi,
        ):
    dpimerge_info = []
    for name,srcs in dpimerge_outputs:
        dpimerge_info.append(f"{name}=(")
        dpimerge_info+=indent(2*'\t',[
                f"'{x}'," for x in srcs
                ]+['),'])
    return [
            f"{next_ver}:dict(",
            ]+indent(2*'\t',[
                "description='PUT VERSION DESCRIPTION HERE',",
                "UNIPROT_VER='PUT UNIPROT VERSION HERE',",
                "**dpimerge_config(",
                ]+indent(2*'\t',[
                    "match_inputs=["
                    ]+indent(2*'\t',[
                        f"('{x}','v{y}')," for x,y in source_versions
                        ]+["],"]
                    )+dpimerge_info+[
                    "combo_dpi=(",
                    ]+indent(2*'\t',[
                        f"'{x}'," for x in combo_dpi
                        ]+["),"]
                    )+["),"]
                )+["),"]
            )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='''
    Show a proposed versions.py file entry for the next version.
    ''')
    parser.add_argument('--list_source_defaults',action='store_true')
    parser.add_argument('source',nargs='*')
    args = parser.parse_args()

    if args.list_source_defaults:
        print(' '.join(default_sources))
        import sys
        sys.exit(0)
    if args.source:
        source_list = args.source
    else:
        source_list = default_sources
    # pre-load source version info
    from dtk.etl import get_last_published_version,get_versions_namespace
    source_versions = []
    for name in source_list:
        ns = get_versions_namespace(name)
        max_ver = max(ns['versions'].keys())
        source_versions.append((name,max_ver))
    # extract other config from previous version of matching
    matching_ver = get_last_published_version('matching')
    last_config = get_versions_namespace('matching')['versions'][matching_ver]
    dpimerge_outputs = []
    inputs = None
    for item in last_config['DPIMERGE_ARG_STEMS'].split():
        if item == '+':
            inputs = None
        elif inputs is None:
            # this is an output name
            inputs = []
            name = item.split('.')[1]
            dpimerge_outputs.append((name,inputs))
        else:
            # this is a source for the previous output name
            name = '.'.join(item.split('/')[-1].split('.')[:2])
            inputs.append(name)
    combo_dpi = list(set([
            item.split('.')[1]
            for item in last_config['COMBO_OUT_STEMS'].split()
            ]))
    # output configuration
    print('\n'.join(indent(2*'\t',matching_config(
            next_ver=matching_ver+1,
            source_versions=source_versions,
            dpimerge_outputs = dpimerge_outputs,
            combo_dpi = combo_dpi,
            ))))
