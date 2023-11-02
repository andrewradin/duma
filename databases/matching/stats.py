#!/usr/bin/env python3

def make_datas(dpi_fn, moa_dpi_fn, clusters_fn):
    from dtk.files import get_file_records
    ev_mols = set()
    targets = set()
    assocs = set()
    for mol, targ, ev, dr in get_file_records(dpi_fn, keep_header=False, progress=True):
        ev_mols.add(mol)
        targets.add(targ)
        assocs.add((mol, targ))

    ev_moas = set()
    moa_targets = set()
    moa_assocs = set()
    for mol, targ, ev, dr in get_file_records(moa_dpi_fn, keep_header=False, progress=True):
        ev_moas.add(mol)
        moa_targets.add(targ)
        moa_assocs.add((mol, targ))


    num_clusters = 0
    clustered_mols = set()
    for rec in get_file_records(clusters_fn, keep_header=None, progress=True):
        num_clusters += 1
        clustered_mols.update(rec)

    return dict(
            ev_mols=ev_mols,
            targets=targets,
            assocs=assocs,
            ev_moas=ev_moas,
            moa_targets=moa_targets,
            moa_assocs=moa_assocs,
            num_clusters=num_clusters,
            clustered_mol=clustered_mols,
            )

if __name__ == "__main__":
    from dtk.etl import run_stats_main
    run_stats_main(make_datas, files=[
        'matching.DNChBX_ki.v{version}.dpimerge.tsv',
        'matching.DNChBX_ki-moa.v{version}.dpimerge.tsv',
        'matching.full.v{version}.clusters.tsv',
        ])
