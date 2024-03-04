#!/usr/bin/env python
import sys, os

class combine(object):
    def __init__(self, **kwargs):
        self.dpis = kwargs.get('dpi', [])
        self.cols = kwargs.get('collection', [])
        collection_name = 'ncats'
        self.out_file = '.'.join(['create', collection_name, 'full', 'tsv'])
        self.dpi_file = '.'.join(['dpi', collection_name, 'default', 'tsv'])
    def run(self):
        self._load()
        self._dedup()
    def _dedup(self):
        # Always keep the latest year, and use the key corresponding to
        # that year (so a link to the NCATS website shows the latest pdf)
        ks_to_delete = []
        for base_key,d in self.create_data.iteritems():
            if len(d) == 1:
                continue # no duplicates
            ordered = sorted(d.keys())
            selected = ordered[-1]
            rest = ordered[:-1]
            status = '(identical)'
            for other in rest:
                if not self._identical_entries(base_key,selected,other):
                    status = '(different)'
                    break
            print base_key,'selected',selected,'over',' '.join(rest),status
            for other in rest:
                del d[other]
    def _identical_entries(self, base_key, yr1, yr2):
        d1 = self.create_data[base_key][yr1]
        d2 = self.create_data[base_key][yr2]
        d1 = {k:v for k,v in d1.iteritems() if k != 'canonical'}
        d2 = {k:v for k,v in d2.iteritems() if k != 'canonical'}
        if d1 != d2:
            return False
        d1 = self.dpi_data[base_key][yr1]
        d2 = self.dpi_data[base_key][yr2]
        return d1 == d2
    key_suffix_map = {
            '2017':'-2016',
            }
    def _parse_filename(self,fn):
        yr = fn.split('/')[-1].split('.')[0]
        key_suffix = self.key_suffix_map.get(yr,'')
        return yr,key_suffix
    def _base_key(self,native_key,key_suffix):
        if key_suffix:
            assert native_key.endswith(key_suffix)
            return native_key[:-len(key_suffix)]
        return native_key
    def _load(self):
        try:
            from dtk.files import get_file_records
        except ImportError:
            sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
            from dtk.files import get_file_records
        self.col_header = ''
        # create a dictionary like:
        # {base_key:{yr:{attr:[val,...],...},...},...}
        # this simplifies finding duplicates, and choosing the latest year;
        # the base key has any year suffix removed; the full key is stashed
        # as an attribute _key_, and then used as the actual key when the
        # output records are written.
        self.create_data = {}
        for f in self.cols:
            header = None
            yr,key_suffix = self._parse_filename(f)
            for frs in get_file_records(f):
                if header is None:
                    self.col_header = "\t".join(frs)
                    header = True
                    continue
                native_key = frs[0]
                base_key = self._base_key(native_key,key_suffix)
                d = self.create_data.setdefault(base_key,{})
                if yr not in d:
                    d2 = {'_key_':frs[0]}
                    d[yr] = d2
                d2.setdefault(frs[1],[]).append(frs[2])
        # create a dictionary like:
        # {base_key:{yr:{uniprot:[ev,dir],...},...},...}
        # the matching entry to the year selected from the dictionary above
        # can be easily retrieved
        self.dpi_data = {}
        self.dpi_header = ''
        for f in self.dpis:
            yr,key_suffix = self._parse_filename(f)
            header = None
            for frs in get_file_records(f):
                if header is None:
                    self.dpi_header = "\t".join(frs)
                    header = True
                    continue
                native_key = frs[0]
                base_key = self._base_key(native_key,key_suffix)
                d = self.dpi_data.setdefault(base_key,{})
                d2 = d.setdefault(yr,{})
                d2[frs[1]] = [frs[2], frs[3]]
    def dump(self):
        with open(self.out_file, 'w') as f:
            with open(self.dpi_file, 'w') as f2:
                f.write(self.col_header + "\n")
                f2.write(self.dpi_header + "\n")
                for k in sorted(self.create_data):
                    d = self.create_data[k]
                    assert len(d) == 1
                    yr = d.keys()[0]
                    d = d[yr]
                    native_key = d['_key_']
                    del d['_key_']
                    # luckily canonical is first alphabetically,
                    # so we can just sort and still get it first
                    # (as is required)
                    for a in sorted(d):
                        for v in d[a]:
                            f.write("\t".join([native_key, a, v]) + "\n")
                    if k != native_key:
                        f.write("\t".join([native_key, 'synonym', k]) + "\n")
                    if k not in self.dpi_data:
                        continue
                    for uni in sorted(self.dpi_data[k][yr]):
                        l = self.dpi_data[k][yr][uni]
                        f2.write("\t".join([native_key, uni] + l) + "\n")

if __name__ == '__main__':
    import argparse

    arguments = argparse.ArgumentParser(description="Combines multiple years of NCATS data while eliminating duplicates")
    arguments.add_argument("collection_files", nargs='+', help="e.g. 2012.collection.tsv")
    # To easily allow for an undefined number of files, I'm using the '--' denotation,
    # but I want to keep it required. Not pretty, but it gets the job done
    arguments.add_argument('--dpi_files', nargs='+', required=True, help="e.g. 2012.dpi.tsv")
    args = arguments.parse_args()

    c = combine(dpi = args.dpi_files, collection = args.collection_files)
    c.run()
    c.dump()

