#!/usr/bin/env python3

def make_datas(c50_ev_fn, ki_ev_fn, cond_attr_fn, full_attr_fn):
    from dtk.files import get_file_records

    full_attr_mols = set()
    for mol, attr_name, attr_val in get_file_records(full_attr_fn, keep_header=False, progress=True):
        full_attr_mols.add(mol)

    cond_attr_mols = set()
    cond_attrs = set()
    for mol, attr_name, attr_val in get_file_records(cond_attr_fn, keep_header=False, progress=True):
        cond_attr_mols.add(mol)
        cond_attrs.add((mol, attr_name, attr_val))


    full_targs = set()
    full_moas_accum = []
    cond_targs = set()
    cond_moas_accum = []
    for ev_fn in [c50_ev_fn, ki_ev_fn]:
        for mol, targ, ev, dr in get_file_records(ev_fn, keep_header=False, progress=True):
            # Ignore low-conf Ki that we don't use.
            from dtk.prot_map import DpiMapping
            if float(ev) < DpiMapping.default_evidence:
                continue

            full_targs.add(targ)
            full_moas_accum.append([mol, targ])

            if mol in cond_attr_mols:
                cond_targs.add(targ)
                cond_moas_accum.append([mol, targ])
    

    from dtk.data import MultiMap
    full_moas = set(frozenset(x) for x in MultiMap(full_moas_accum).fwd_map().values())
    cond_moas = set(frozenset(x) for x in MultiMap(cond_moas_accum).fwd_map().values())


    return dict(
        full_attr_mols=full_attr_mols,
        # This one uses up a lot of memory and isn't particularly interesting.
        # full_attrs=full_attrs,
        cond_attr_mols=cond_attr_mols,
        cond_attrs=cond_attrs,
        full_targs=full_targs,
        cond_targs=cond_targs,
        full_moas=full_moas,
        cond_moas=cond_moas,
    )

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'chembl.c50.v{version}.evidence.tsv',
        'chembl.ki.v{version}.evidence.tsv',
        'chembl.adme_condensed.v{version}.attributes.tsv',
        'chembl.full.v{version}.attributes.tsv',
        ])
