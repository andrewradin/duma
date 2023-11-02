#!/usr/bin/env python3

from dtk.lazy_loader import LazyLoader

class ClusterStats(LazyLoader):
    _kwargs=['path']
    def _clusters_loader(self):
        # read cluster file, discarding keys and keeping collection names
        from dtk.files import get_file_records
        return [
                rec[::2]
                for rec in get_file_records(self.path)
                ]
    def _cluster_count_loader(self):
        return len(self.clusters)
    def _molecule_count_loader(self):
        return sum([len(x) for x in self.clusters])
    def _size_histogram_loader(self):
        from collections import Counter
        return Counter([len(x) for x in self.clusters])
    def _struct_histogram_loader(self):
        from collections import Counter
        return Counter([' '.join(x) for x in self.clusters])
    def _src_histogram_loader(self):
        from collections import Counter
        ctr = Counter()
        for row in self.clusters:
            ctr.update(row)
        return ctr
    def _srcs_loader(self):
        return sorted(self.src_histogram.keys())
    def _src_participation_loader(self):
        from collections import Counter
        ctr = Counter()
        for src in self.srcs:
            ctr[src] = sum(src in x for x in self.clusters)
        return ctr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='show clustering stats'
            )
    parser.add_argument('--size-hist',action='store_true'),
    parser.add_argument('--most-common',type=int),
    parser.add_argument('cluster_file'),
    args = parser.parse_args()

    cs = ClusterStats(path=args.cluster_file)
    print('Total number of clusters:',cs.cluster_count)
    print('Total number of clustered molecules:',cs.molecule_count)
    print('Average cluster size:',cs.molecule_count/cs.cluster_count)
    indent = '    '
    print('Molecule source histogram:')
    for key,count in cs.src_histogram.most_common():
        print(indent,key,count)
    print('Molecule source participation:')
    rows = [('source','clusters with','clusters without')]+[
            (key,str(count),str(cs.cluster_count-count))
            for key,count in cs.src_participation.most_common()
            ]
    from dtk.text import print_table
    print_table(rows)
    if args.size_hist:
        print('Cluster size histogram:')
        for key in sorted(cs.size_histogram):
            print(indent,key,cs.size_histogram[key])
    if args.most_common:
        print('Cluster structure histogram:')
        for key,count in cs.struct_histogram.most_common(args.most_common):
            print(indent,count,key)
        print(indent,'...')

# XXX Overall stats:
# XXX - total number of drugs not in clusters (requires total number in
# XXX   each collection)
# XXX Per-collection stats:
# XXX - fraction of collection in clusters (requires total number in cluster)
# XXX - histogram of collection drugs per cluster

