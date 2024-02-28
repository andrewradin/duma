#!/usr/bin/env python3

def make_datas(ev_fn, attr_fn):
    from dtk.files import get_file_records
    ev_mols = set()
    targets = set()
    assocs = set()

    attr_mols = set()
    attrs = set()
    for mol, targ, ev, dr in get_file_records(ev_fn, keep_header=False, progress=True):
        ev_mols.add(mol)
        targets.add(targ)
        assocs.add((mol, targ))

    for mol, attr_name, attr_val in get_file_records(attr_fn, keep_header=False, progress=True):
        attr_mols.add(mol)
        attrs.add((mol, attr_name, attr_val))

    return dict(ev_mols=ev_mols, targets=targets, assocs=assocs, attr_mols=attr_mols, attrs=attrs)

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=['drugbank.default.v{version}.evidence.tsv', 'drugbank.full.v{version}.attributes.tsv'])