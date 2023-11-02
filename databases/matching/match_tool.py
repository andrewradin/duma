#!/usr/bin/env python3

import sys

import dtk.drug_clusters as dc

# XXX currently:
# XXX - the 'archive' file (base_drug_clusters.tsv) doesn't hold the final
# XXX   clustering; instead, it holds all the attribute data used for
# XXX   clustering, in the form:
# XXX   attr_name attr_val {collection drugkey}+
# XXX   (although maybe all problematic attributes have been removed by
# XXX   this point, so a naive clustering approach might work on this data)
# XXX - cluster_dump.out lists all native keys in a cluster in sorted
# XXX   order, so it's basically a canonical cluster definition, but
# XXX   doesn't select which key is the dpimerge_id
# XXX The MoleculeKeySet class takes lines from cluster_dump.out as
# XXX input. It can be used in the dpimerge map uploader. The m. files
# XXX themselves are no longer needed, and can be removed.

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='match drugs')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--trim',action='store_true')
    parser.add_argument('--save-archive',action='store_true')
    parser.add_argument('--write-cluster-dump',action='store_true')
    parser.add_argument('--falsehoods')
    parser.add_argument('--logdir')
    parser.add_argument('collection',nargs='+')
    args = parser.parse_args()

    if args.self_test:
        # run unit tests
        suite = dc.unittest.TestLoader().loadTestsFromTestCase(dc.ClusterTest)
        dc.unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(1)

    clr = dc.Clusterer()
    if args.logdir:
        clr.logdir = args.logdir
    if args.falsehoods:
        clr.load_fact_filter(args.falsehoods)

    file_collections = []
    for coll in args.collection:
        if coll == 'archive':
            clr.load_archive()
        else:
            file_collections.append(coll)

    # pull in data from drug attributes files;
    # the use of direct_cols, name_cols, and name_remapper makes
    # sure this applies the same filtering as RebuiltCluster
    for coll in file_collections:
        print("loading",coll)
        clr.load_from_file(coll,dc.RebuiltCluster.direct_cols)
        print("loading names from",coll)
        clr.load_from_file(coll,dc.RebuiltCluster.name_cols,
                map_function=dc.RebuiltCluster.name_remapper,
                )
    print(clr.collection_stats())

    if args.trim:
        # This code removes any props that match too many drugs.
        # This was originally used fairly aggressively,
        # but now just handles extreme exceptions.
        print("building links (first cut)")
        clr.build_links()
        print(clr.link_stats())

        print("trimming props")
        # As of 1/2021, this only trims 2 props. The suffix exclusion
        # in name_remapper removes most of the promiscuous names, and
        # promiscuous std_smiles codes are preserved.
        clr.trim_props(max_size=50,keep_types=set(['std_smiles']))
        # we no longer use trim_props2, because std_smiles matching can
        # result in legitimate clusters with a dozen or more drugs in
        # the same collection
        print(clr.trim_disconnected_drugs())

    print("building links")
    clr.build_links()
    print(clr.collection_stats())
    print(clr.link_stats())

    print("building clusters")
    clr.build_clusters()
    print(clr.cluster_stats())

    if True:
        # This code examines the clusters built above for clusters that are
        # 'suspicious' -- i.e. they appear to contain multiple molecules
        # that we'd like to distinguish. These then get broken into smaller
        # clusters, possibly dropping drugs that can't be clearly assigned
        # to one of the new clusters.
        print("rebuilding clusters")
        clr.rebuild_clusters()
        print(clr.cluster_stats())
        print("clustered drugs",sum(len(x) for x in clr.clusters))
        # The above value should match the total printed by collection_stats
        # below.
        # By subtracting the per-collection counts printed below with the
        # original counts at the top, we can determine the number of unmatched
        # (unique) drugs per collection.
        print(clr.collection_stats())

    if args.save_archive:
        print("saving archive")
        clr.save_archive()

    if args.write_cluster_dump:
        clr.write_cluster_dump()

    print(clr.stat_overview())

