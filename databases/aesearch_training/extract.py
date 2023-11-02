#!/usr/bin/env python

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
django.setup()

#-------------------------------------------------------------------------------
# Extractor
#-------------------------------------------------------------------------------
class Extractor:
    '''Identify training data for GE dataset classifier.

    Positive cases are all imported tissues that aren't excluded.
    Negative cases are all AeDispostion records that don't match a positive
    case.
    '''
    def __init__(self):
        # set up filter for only CC or TR tissue sets
        self.ts2mode = {}
        from browse.models import TissueSet
        for ts in TissueSet.objects.all():
            if ts.name in ('default','Case/Control'):
                self.ts2mode[ts.id] = 'CC'
            elif ts.name in ('Treatment/Response'):
                self.ts2mode[ts.id] = 'TR'
        # initialize other variables
        self.ignore_sources = ['ext','comb']
        self.outf = sys.stdout
        self.pos = {}
        self.neg = set()
    def ok(self,key):
        # allow filtering by components of key
        # this isn't currently used; it can be done with post-processing
        ws,geoID,mode = key
        #if ws == 78:
        #    return False # has multiple tissues per geoID
        return True
    def geo2ae(self,geoID):
        # strip any user-supplied suffix
        geoID = geoID.split(':')[0]
        # convert to AE format
        if geoID.startswith('GSE'):
            return 'E-GEOD-'+geoID[3:]
        return geoID
    def write(self,*args):
        self.outf.write('\t'.join([str(x) for x in args])+'\n')
    def write_tissue(self,t,mode):
        # output positive cases, given a tissue record
        if t.source in self.ignore_sources:
            return
        # handle combined datasets
        for geoID in t.geoID.split(','):
            # construct key, validate, and record source
            key = (t.ws_id,self.geo2ae(geoID),mode)
            if self.ok(key):
                self.pos.setdefault(key,set()).add(t.source)
    def expand_comb(self,t,mode):
        # for comb tissues, record all components
        for t_id in [int(x) for x in t.geoID.split(',')]:
            from browse.models import Tissue
            self.write_tissue(Tissue.objects.get(pk=t_id),mode)
    def run(self):
        from browse.models import Tissue,AeDisposition,AeSearch
        # scan all tissues and record positive cases
        for t in Tissue.objects.all():
            try:
                mode = self.ts2mode[t.tissue_set_id]
            except KeyError:
                continue # skip if not a good tissue set
            if t.source == 'comb':
                self.expand_comb(t,mode)
            else:
                self.write_tissue(t,mode)
        # scan all dispositions and record negative cases
        enum = AeSearch.mode_vals
        for ws,geoID,mode in AeDisposition.objects.values_list(
                    'accession__ws_id',
                    'accession__geoID',
                    'mode',
                    ):
            key = (ws,geoID,enum.get('symbol',mode))
            if self.ok(key) and key not in self.pos:
                self.neg.add(key)
        # output results
        for key,s in self.pos.items():
            self.write(*(list(key)+[','.join(s)]))
        for key in self.neg:
            self.write(*(list(key)+['reject']))

#-------------------------------------------------------------------------------
# Driver
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description='extract AE SEARCH training data',
                )
    args = parser.parse_args()

    ex = Extractor()
    ex.run()
