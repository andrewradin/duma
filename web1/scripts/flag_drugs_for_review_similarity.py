#!/usr/bin/env python3

from __future__ import print_function
import sys
import six
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/..")
    import path_helper

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
django.setup()

from flagging.utils import FlaggerBase

def letternumber(k):
    k = int(k)
    letter = chr((k % 26) + ord('A'))
    repeat = k//26
    name = letter
    if repeat == 0:
        return letter
    else:
        return f'{letter}{repeat+1}'


### TODO
### Develop some UI for showing the meta/outer clusters
### currently the link generated just shows the inner/DPI-generated clusters,
### but the name of the cluster is the outerClusterKey_innerClusterKey

class Flagger(FlaggerBase):
    def __init__(self,**kwargs):
        super(Flagger,self).__init__(kwargs)
        self.dpi = kwargs.pop('dpi')
        self.ppi = kwargs.pop('ppi')
        self.dpi_threshold = kwargs.pop('dpi_threshold')
        self.ppi_threshold = kwargs.pop('ppi_threshold')
        self.repulsion = kwargs.pop('repulsion')
        # These additional parameters are exposed for the rvw_clusters tuning
        # page, but defaults are supplied because they're not yet changeable
        # from the command line.
        self.damping = kwargs.pop('damping',0.8)
        self.max_iter = kwargs.pop('max_iter',1000)
        self.method = kwargs.pop('method','ST')
        self.st_dir_thresh = kwargs.pop('st_dir_thresh',0.7)
        self.st_ind_thresh = kwargs.pop('st_ind_thresh',0.3)
        assert not kwargs
    def flag_drugs(self):
        self.setup()
        self.create_flag_set('ReviewSimilarity')
        self.build_clusters()
        self.do_report()
    def setup(self):
        from dtk.prot_map import DpiMapping, PpiMapping
        self.dm = DpiMapping(self.dpi)
        self.pm = PpiMapping(self.ppi)
        self.build_native2wsa_map(self.get_target_wsa_ids())
    def build_clusters(self):
        self.build_wsa2dpi_map()
        self.build_similarity_matrix()
        self.group2members = self.do_cluster(self.raw_clusters(self.sm,self.st_dir_thresh))
        self.build_uniprot2ppi_map()
        self.build_meta_similarity_matrix()
        self.meta_group2members = self.do_cluster(self.raw_clusters(self.meta_sm,self.st_ind_thresh))
        self.process_metaclusters()
    def process_metaclusters(self):
        self._change_meta_ids_to_letters()
        clustered = set().union(*list(self.meta_group2members.values()))
        all = set([str(k) for k in self.group2members.keys()])
        self.meta_group2members['none'] = all - clustered
    def _change_meta_ids_to_letters(self):
        current_keys = list(self.meta_group2members.keys())
        for k in current_keys:
            name = letternumber(k - 1)

            self.meta_group2members[name] = self.meta_group2members[k]
            del self.meta_group2members[k]
    def build_wsa2dpi_map(self):
        # scan DPI file
        self.wsa2dpi = {}
        from dtk.files import get_file_records
        for row in get_file_records(self.dm.get_path(),
                            keep_header=False,
                            ):
            native,uniprot,ev = row[:3]
            if float(ev) < self.dpi_threshold:
                continue
            # build a set of uniprots for each WSA
            for wsa_id in self.native2wsa.get(native,[]):
                self.wsa2dpi.setdefault(wsa_id,set()).add(uniprot)
    def build_uniprot2ppi_map(self):
        # scan PPI file
        self.uniprot2ppi = {}
        from dtk.files import get_file_records
        for row in self.pm.get_data_records(min_evid=self.ppi_threshold):
            u1,u2,ev = tuple(row)[:3]
            # build a set of uniprots for each uniprot
            if u1 not in self.uniprot2ppi:
                self.uniprot2ppi[u1] = set()
            self.uniprot2ppi[u1].add(u2)
    def build_similarity_matrix(self):
        # now create similarity matrix using jaccard
        from dtk.similarity import build_mol_prot_sim_matrix
        self.sm = build_mol_prot_sim_matrix(self.wsa2dpi)
    def raw_clusters(self, sm, st_thresh=None):
        if self.method == 'AP':
            return sm.clusters(
                    self.repulsion,
                    damping=self.damping,
                    max_iter=self.max_iter,
                    )
        if self.method == 'ST':
            assert st_thresh
            return sm.clusters2(st_thresh)
        raise NotImplementedError("unrecognized method '%s'"%self.method)
    def do_cluster(self, clusters):
        print(sum([len(x) for x in clusters]),'keys returned')
        print(sum([len(x) for x in clusters if len(x) > 1]),'keys grouped')
        group2members = {
                (i+1):set([str(y) for y in sorted(x)])
                for i,x in enumerate(clusters)
                if len(x) > 1
                }
        # report
        for i in sorted(group2members.keys()):
            print('cluster %d: %s'%(i,' '.join(group2members[i])))
        return group2members
    def build_meta_similarity_matrix(self):
        # now create meta similarity matrix using indirect jaccard
        self.extract_clusters()
        self.integrate_ppi()
        from dtk.similarity import build_mol_prot_sim_matrix
        self.meta_sm = build_mol_prot_sim_matrix(self.cluster_indirect_dpi)
    def extract_clusters(self, min_por = 0.5):
        # I want each cluster to have a representative DPI signature
        # where the key will be the cluster identifier
        # and the value will be a set of uniprot IDs.
        # the uniprot IDs will come from the cluters
        # my current thinking is any protein that
        # shows up in half or more of the drugs in the cluster
        from collections import Counter
        self.cluster_dpi = {}
        for key, s in six.iteritems(self.group2members):
            self.cluster_dpi[key] = []
            min_cnt = int(len(s) * min_por)
            cnts = Counter([u for wsa_id in s
                              for u in self.wsa2dpi[int(wsa_id)]
                           ])
            for u,c in cnts.most_common():
                if c < min_cnt:
                    break
                self.cluster_dpi[key].append(u)
    def integrate_ppi(self):
        self.cluster_indirect_dpi={}
        for k,s in six.iteritems(self.cluster_dpi):
            self.cluster_indirect_dpi[k] = set()
            for u in s:
                if u in self.uniprot2ppi:
                    self.cluster_indirect_dpi[k].update(self.uniprot2ppi[u])
    def do_report(self, sep = '_'):
        # build reverse index
        wsa2group={}
        # self.meta_group2members will be keyed by the meta group IDs
        # the values will be a set of initial cluster IDs
        # those can then be used as the keys to group2members
        # the returned value will be the original WSAs
        for m_key,s in six.iteritems(self.meta_group2members):
            for key in s:
                for wsa_id in self.group2members[int(key)]:
                    wsa2group[wsa_id] = sep.join([str(m_key), key])

        # write results
        kts=self.ws.get_wsa_id_set(self.ws.eval_drugset)
        any_class=self.ws.get_wsa_id_set('classified')
        from dtk.duma_view import qstr
        from django.urls import reverse
        for wsa_id, group in six.iteritems(wsa2group):
            mk, k = group.split(sep)
            members = self.group2members[int(k)]
            # link_ids can get modified below, so we need to take a
            # copy here to avoid modifying the original group2members.
            link_ids = members.copy()
            if kts & set([int(x) for x in members]):
                category='Similar DPI w/KT'
            elif any_class & set([int(x) for x in members]):
                category='Similar DPI w/Any'
            else:
# only flag drugs that have DPI with indirect overlap of a KT
                outer_members = [int(x)
                              for k_set in self.meta_group2members[mk]
                              for k in k_set
                              for x in self.group2members.get(int(k), [])
                             ]
                if kts & set(outer_members):
                    category='Indirectly sim. DPI w/KT'
                    link_ids.update(set([x for x in outer_members if x in kts]))
                else:
                    category='Similar DPI'
### for the time being I'm leaving the page as just the inner cluster,
### but we need some way to show the outer cluster
            href = reverse('clust_screen',args=(self.ws_id,))+qstr({},
                            ids=','.join([str(s) for s in link_ids]),
                            dpi=self.dpi,
                            dpi_t=self.dpi_threshold,
                            )
            self.create_flag(
                    wsa_id=wsa_id,
                    category=category,
                    detail='cluster %s'%group,
                    href=href,
                    )

    def build_native2wsa_map(self,wsa_ids):
        # native2wsa maps from a native collection key to a set of wsa_ids
        self.native2wsa = {
                k:set([x for x in v if x in wsa_ids])
                for k,v in self.dm.get_wsa_id_map(self.ws).items()
                }
        print('got',len(wsa_ids),'wsa ids;',len(self.native2wsa),'dpi keys')

if __name__ == '__main__':
    import argparse
    from dtk.prot_map import DpiMapping, PpiMapping
    parser = argparse.ArgumentParser(
            description="flag drugs for Review Similarity",
            )
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--count',type=int,default=200)
    parser.add_argument('ws_id',type=int)
    parser.add_argument('job_id',type=int)
    parser.add_argument('score')
    parser.add_argument('--dpi',)
    parser.add_argument('--dpi-threshold',type=float,
            default=DpiMapping.default_evidence,
            )
    parser.add_argument('--ppi',)
    parser.add_argument('--ppi-threshold',type=float,
            default=PpiMapping.default_evidence,
            )
    parser.add_argument('--repulsion',type=float,default=0.5)
    args = parser.parse_args()

    from runner.models import Process
    err = Process.prod_user_error()
    if err:
        raise RuntimeError(err)

    flagger = Flagger(
                ws_id=args.ws_id,
                job_id=args.job_id,
                score=args.score,
                start=args.start,
                count=args.count,
                dpi=args.dpi,
                dpi_threshold=args.dpi_threshold,
                ppi=args.ppi,
                ppi_threshold=args.ppi_threshold,
                repulsion=args.repulsion,
                )
    flagger.flag_drugs()
