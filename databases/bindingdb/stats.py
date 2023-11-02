#!/usr/bin/env python3

def make_datas(c50_ev_fn, ki_ev_fn, full_attr_fn):
    from dtk.files import get_file_records

    full_attr_mols = set()
    for mol, attr_name, attr_val in get_file_records(full_attr_fn, keep_header=False, progress=True):
        full_attr_mols.add(mol)

    full_targs = set()
    full_moas_accum = []
    for ev_fn in [c50_ev_fn, ki_ev_fn]:
        for mol, targ, ev, dr in get_file_records(ev_fn, keep_header=False, progress=True):
            # Ignore low-conf Ki that we don't use.
            from dtk.prot_map import DpiMapping
            if float(ev) < DpiMapping.default_evidence:
                continue

            full_targs.add(targ)
            full_moas_accum.append([mol, targ])


    from dtk.data import MultiMap
    full_moas = set(frozenset(x) for x in MultiMap(full_moas_accum).fwd_map().values())

    return dict(
        full_attr_mols=full_attr_mols,
        full_targs=full_targs,
        full_moas=full_moas,
    )

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'bindingdb.c50.v{version}.evidence.tsv',
        'bindingdb.ki.v{version}.evidence.tsv',
        'bindingdb.full_condensed.v{version}.attributes.tsv',
        ])
