#!/usr/bin/env python3

class RandomAccessSimsets:
    delim = '#\n'
    def __init__(self,fn):
        self.fh = open(fn)
        self.build_index()
    def build_index(self):
        self.index = {}
        pos = 0
        self.fh.seek(pos)
        cur_keyset=set()
        base = pos
        for line in self.fh:
            pos += len(line)
            if line == self.delim:
                for key in cur_keyset:
                    self.index[key] = base
                cur_keyset=set()
                base = pos
                continue
            rec = line.rstrip('\n').split('\t')
            cur_keyset.add(rec[1])
            cur_keyset.add(rec[2])
    def get(self,key):
        start = self.index[key]
        self.fh.seek(start)
        for line in self.fh:
            if line == self.delim:
                return
            rec = line.strip('\n').split('\t')
            yield rec

def list_convert(l):
    return [
            [float(row[0])]+row[1:]
            for row in l
            ]

def all_keys(l):
    s = set()
    for row in l:
        s.add(row[1])
        s.add(row[2])
    return s

def canonical_ordering(c):
    l = [sorted(x) for x in c]
    return sorted(l)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='compare tracefiles')
    args = parser.parse_args()

    ras5 = RandomAccessSimsets('tracefile.try5')
    ras6 = RandomAccessSimsets('tracefile.try6')
    import tqdm
    from dtk.coll_condense import cluster_by_similarity
    done = set()
    for key in tqdm.tqdm(ras5.index):
        if key in done:
            continue
        ras5l = list_convert(ras5.get(key))
        ras6l = list_convert(ras6.get(key))
        ras5s = all_keys(ras5l)
        ras6s = all_keys(ras6l)
        if ras5s != ras6s:
            raise RuntimeError(f"set difference")
        done |= ras5s
        ras5c = canonical_ordering(cluster_by_similarity(ras5l))
        ras6c = canonical_ordering(cluster_by_similarity(ras6l))
        if ras5c != ras6c:
            raise RuntimeError(f"cluster difference\n{ras5c}\n{ras6c}")


