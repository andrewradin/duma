#!/usr/bin/env python

from __future__ import print_function
import sys
if 'django.core' not in sys.modules:
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
    import django
    django.setup()

from browse.models import Workspace
import runner.data_catalog as dc

class FeatureSet:
    def _wsaset_fetcher_base(self,s,keyset,keysort=False):
        from browse.models import WsAnnotation
        qs = WsAnnotation.objects.filter(ws=self.ws).order_by('id')
        if keyset:
            qs = qs.filter(id__in=keyset)
        for wsa in qs:
            yield (wsa.id, (wsa.id in s,))
    def __init__(self,ws_id,features):
        self.ws = Workspace.objects.get(pk=ws_id)
        self.cat = dc.Catalog()
        self.features = features
        self.with_key = False
        self.key_file = None
        self.plug_unknowns = False
        self.hf_labels = False
        self.exclude_hidden = False
        self.aliases = {}
        # get all the wsa_id_set codes; if one is used as a feature,
        # we'll generate a CodeGroup on the fly
        wsa_set_names = set(x[0] for x in self.ws.get_wsa_id_set_choices())
        # strip out all the "features" that are really job map entries
        keep = []
        for code in self.features:
            if ':' in code:
                # this is actually a job alias
                job_id,label = code.split(':')
                self.aliases[int(job_id)] = label
            else:
                keep.append(code)
        self.features = keep
        # find all the jobs supplying features (assume that all codes
        # are either wsa set names, or in the format handled by parse_stdcode)
        job_ids = set()
        for code in self.features:
            if code in wsa_set_names:
                s = self.ws.get_wsa_id_set(code)
                cg = dc.CodeGroup('wsa',
                        lambda *a,**kwa:self._wsaset_fetcher_base(s,*a,**kwa),
                        dc.Code(code,valtype='bool'),
                        )
                self.cat.add_group('',cg)
            else:
                job_id,subcode = dc.parse_stdcode(code)
                job_ids.add(job_id)
        # add all the code groups from each of those jobs to our catalog
        from runner.process_info import JobInfo
        for job_id in job_ids:
            bji = JobInfo.get_bound(self.ws,job_id)
            for cg in bji.get_data_code_groups():
                self.cat.add_group(
                            dc.stdprefix(job_id),
                            cg,
                            self.aliases.get(job_id,''),
                            )
    def _write_data(self,f,cols,data):
        if self.plug_unknowns:
            default_map = {
                    'float':'0.0',
                    'bool':'False',
                    }
            self.defaults = [
                    default_map[self.cat.get_valtype(x)]
                    for x in cols
                    ]
        else:
            self.defaults = ['?'] * len(cols)
        # saved scores may include wsa_ids that are no longer in the
        # workspace, so set up a whitelist filter; if we're excluding
        # hidden drugs, don't include them in the filter
        from browse.models import WsAnnotation
        qs = WsAnnotation.objects.filter(ws=self.ws)
        if self.exclude_hidden:
            qs = qs.exclude(agent__hide=True)
        whitelist = set(qs.values_list('id',flat=True))
        for key,buf in data:
            if key not in whitelist:
                continue
            # don't output rows that are entirely unknowns
            valset = set(buf)
            if valset == set([None]):
                continue
            fmt = [
                    self.defaults[i] if v is None else str(v)
                    for i,v in enumerate(buf)
                    ]
            if self.key_file:
                self.key_file.write(str(key)+'\n')
            if self.with_key:
                fmt=[str(key)]+fmt
            f.write(','.join(fmt)+'\n')
    def to_arff(self,fn):
        with open(fn,'w') as f:
            cols,data = self.cat.get_feature_vectors(*self.features)
            if self.hf_labels:
                seen = set()
                any_dups = False
                labels = [self.cat.get_arff_label(col) for col in cols]
                for x in labels:
                    if x in seen:
                        any_dups = True
                        print(x,'is used for multiple features')
                    seen.add(x)
                if any_dups:
                    raise Exception('labels not distinct')
            else:
                labels = cols
            f.write('@RELATION "%s"\n' % self.ws.name)
            f.write('\n')
            if self.with_key:
                f.write('@ATTRIBUTE key STRING\n')
            for label,col in zip(labels,cols):
                f.write('@ATTRIBUTE "%s" %s\n' % (
                            label,
                            self.cat.get_arff_type(col),
                            ))
            f.write('\n')
            f.write('@DATA\n')
            self._write_data(f,cols,data)
    def to_csv(self,fn):
        with open(fn,'w') as f:
            cols,data = self.cat.get_feature_vectors(*self.features)
            if self.with_key:
                f.write('key,')
            f.write(','.join(cols)+'\n')
            self._write_data(f,cols,data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='extract feature vectors')
    parser.add_argument('-o','--outfile'
        ,type=str
        ,help='output file name (default %(default)s)'
        ,default = '/tmp/results.arff'
        )
    parser.add_argument('-c','--csv'
        ,action='store_true'
        ,help='output csv format (default is arff)'
        )
    parser.add_argument('--with-key'
        ,action='store_true'
        ,help='include key as first feature'
        )
    parser.add_argument('--key-file'
        ,help='store key in separate file (in same order)'
        )
    parser.add_argument('--plug-unknowns'
        ,action='store_true'
        ,help='substitute defaults for unknowns'
        )
    parser.add_argument('--hf-labels'
        ,action='store_true'
        ,help='generate human-friendly ARFF labels'
        )
    parser.add_argument('--exclude-hidden'
        ,action='store_true'
        ,help='exclude drugs marked for hiding'
        )
    parser.add_argument('ws_id'
        ,type=int
        ,help='id of workspace to extract from'
        )
    parser.add_argument('feature',nargs='+'
        ,type=str
        ,help='feature name (wsa_id_set name or j<job_id>_<code>)'
        )
    args=parser.parse_args()
    s = FeatureSet(args.ws_id,args.feature)
    if args.key_file:
        s.key_file = open(args.key_file,'w')
    if args.with_key:
        s.with_key = True
    if args.plug_unknowns:
        s.plug_unknowns = True
    if args.hf_labels:
        s.hf_labels = True
    if args.exclude_hidden:
        s.exclude_hidden = True
    if args.csv:
        s.to_csv(args.outfile)
    else:
        s.to_arff(args.outfile)

