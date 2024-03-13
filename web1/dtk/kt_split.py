# XXX The following line is here to allow this file to be run as a script,
# XXX but it doesn't work because django.http.request imports cgi, which
# XXX attempts to import html, but picks up dtk.html instead of the python
# XXX standard module. The same thing happens with a few other dtk files.
# XXX we should either get rid of this idiom and provide wrappers in the
# XXX scripts directory, or we should make sure none of the dtk sub-modules
# XXX conflict with standard python module names.
import django_setup
from django.db import transaction

import logging
logger = logging.getLogger(__name__)

# This module creates test/train splits for a drugset.
#
# The test and train splits aren't stored until the first time they're used,
# since most drugsets won't need test/train splits. They are stored as a
# pair of DrugSplit records, each of which has a m2m relationship with
# WsAnnotation.
#
# Overall architecture is:
# - Cluster class holds a set of drugs along with DPI information
# - make_cluster returns a Cluster object based on a starting drug and
#   DPI information; the cluster contains the transitive closure of all
#   drugs which overlap DPI of any other drug in the cluster
# - KtSplit class assembles a set of drugs into n (==2) clusters by first
#   making all the natural clusters with make_cluster(), and then merging
#   them to get down to n. If there are fewer than n natural clusters,
#   it pads out to n with empty clusters.
# - EditableKtSplit class uses DrugSplit records to record splits in the
#   database, and supports manual editing of splits, and automatic
#   adjustment of splits if the underlying drugset changes.
#
# A CLI allows testing and inspecting the function of the KtSplit class.

class Cluster(object):
    def __init__(self, drugs, drug2dpis):
        self._data = {d: drug2dpis[d] for d in drugs}

    def id(self):
        return next(iter(sorted(self.drugs())))

    def drugs(self):
        return list(self._data.keys())

    def all_dpis(self):
        return set.union(*list(self._data.values()))

    def dpi_table(self):
        out = []
        for drug, dpis in self._data.items():
            row = ["%8s" % drug]
            row += ['X' if dpi in dpis else ' ' for dpi in self.all_dpis()]
            out.append(row)
        return out

    def dpi_table_str(self):
        return '\n'.join(['    '.join(row) for row in self.dpi_table()])

    @staticmethod
    def merge(clusters):
        out = Cluster([], None)
        for cluster in clusters:
            out._data.update(cluster._data)
        return out
        

def make_cluster(drug, drug2dpis, dpi2drugs):
    dpis = drug2dpis[drug]
    clustered_drugs = set([drug])
    seen_dpis = set()
    new_dpis = set(dpis)
    while new_dpis:
        seen_dpis.update(new_dpis)
        dpis_to_add = new_dpis
        new_dpis = set()
        for dpi in dpis_to_add:
            drugs = dpi2drugs[dpi]
            new_drugs = set(drugs) - clustered_drugs
            clustered_drugs.update(drugs)
            for drug in new_drugs:
                new_dpis.update(drug2dpis[drug])
        new_dpis -= seen_dpis
    return Cluster(clustered_drugs, drug2dpis)


class EditableKtSplit(object):
    @transaction.atomic
    def __init__(self, ws, input_drugset_name):
        self.ws = ws
        assert not is_split_drugset(input_drugset_name)
        self.input_drugset_name = input_drugset_name
        self.test_name = 'split-test-' + input_drugset_name
        self.train_name = 'split-train-' + input_drugset_name
        # Get pre-existing splits, if they exist.
        # If they exist, check if we have any new drugs to add
        # If they don't exist, do a clustering.
        from browse.models import DrugSplit
        try:
            self.get_clusters()
        except DrugSplit.DoesNotExist as e:
            self._autosplit()
            self.get_clusters()

    @transaction.atomic
    def modify_split(self, wsas_to_test, wsas_to_train):
        wsas_to_test = [int(x) for x in wsas_to_test]
        wsas_to_train = [int(x) for x in wsas_to_train]
        from browse.models import DrugSplit
        new_test_wsa_ids = [x for x in self.test_wsa_ids
                            if x not in wsas_to_train] + wsas_to_test

        new_train_wsa_ids = [x for x in self.train_wsa_ids
                             if x not in wsas_to_test] + wsas_to_train

        test_split = DrugSplit.objects.get(ws=self.ws, name=self.test_name)
        test_split.drugs.clear()
        test_split.drugs.add(*new_test_wsa_ids)
        test_split.manual_edits = True
        test_split.save()

        train_split = DrugSplit.objects.get(ws=self.ws, name=self.train_name)
        train_split.drugs.clear()
        train_split.drugs.add(*new_train_wsa_ids)
        train_split.manual_edits = True
        train_split.save()

    def get_misaligned(self, base_split):
        '''Return the tuple (to_test,to_train).

        to_test - list of train wsas that appear in test in the base
        to_train - list of test wsas that appear in train in the base
        Note that drugs not appearing in the base are never returned.
        '''
        to_test = [
                x for x in self.train_wsa_ids
                if x in base_split.test_wsa_ids
                ]
        to_train = [
                x for x in self.test_wsa_ids
                if x in base_split.train_wsa_ids
                ]
        return (to_test,to_train)
    def align_with_base(self, base_drugset_name):
        base_split = EditableKtSplit(self.ws,base_drugset_name)
        # re-assign every drug in self that is on the opposite side
        # of the split in base
        to_test,to_train = self.get_misaligned(base_split)
        self.modify_split(to_test, to_train)

    @transaction.atomic
    def redo_autosplit(self):
        from browse.models import DrugSplit, WsAnnotation
        test_split = DrugSplit.objects.get(ws=self.ws, name=self.test_name)
        train_split = DrugSplit.objects.get(ws=self.ws, name=self.train_name)
        test_split.delete()
        train_split.delete()
        self._autosplit()

    def get_clusters(self):
        from browse.models import DrugSplit, WsAnnotation
        test_split = DrugSplit.objects.get(ws=self.ws, name=self.test_name)
        train_split = DrugSplit.objects.get(ws=self.ws, name=self.train_name)

        all_drugs = test_split.drugs.all() | train_split.drugs.all()
        existing_split_wsa_ids = set(x.id for x in all_drugs)

        input_wsa_ids = self.ws.get_wsa_id_set(self.input_drugset_name)
        if existing_split_wsa_ids != input_wsa_ids:
            # There was a change to our input dataset since we split.
            manual_edits = test_split.manual_edits or train_split.manual_edits
            if manual_edits:
                logger.warning(f'{self.input_drugset_name} changed; manual')
                # Someone has gone through the trouble of hand-curating the
                # datasets.  Don't re-autosplit, tweak existing sets instead.
                #
                # Note that this puts all new stuff in test, rather than
                # looking at DPI clusters. This is maybe not ideal, but
                # assures that if we've used align_with_base to align
                # multiple test/train sets, they'll stay aligned after
                # modification.
                #
                # Note also that this all happens silently, so the user
                # may not know to check and possibly adjust the balance.
                # If this becomes an issue, it's probably worth developing
                # a whole flow around creating and maintaining these splits.
                new_ids = input_wsa_ids - existing_split_wsa_ids
                for split in (test_split,train_split):
                    name = split.name
                    if split is test_split and new_ids:
                        logger.info(f'adding wsa_ids {new_ids} to {name}')
                        test_split.drugs.add(*new_ids)
                    gone_ids = set(
                            x.id for x in split.drugs.all()
                            if x.id not in input_wsa_ids
                            )
                    if gone_ids:
                        logger.info(f'removing wsa_ids {gone_ids} from {name}')
                        split.drugs.remove(*gone_ids)
                    split.save()
            else:
                logger.warning(f'{self.input_drugset_name} changed; autosplit')
                # We're already using an auto-split, let's just split again.
                self.redo_autosplit()

        test_split = DrugSplit.objects.get(ws=self.ws, name=self.test_name)
        train_split = DrugSplit.objects.get(ws=self.ws, name=self.train_name)
        self.test_wsa_ids = [x.id for x in test_split.drugs.all()]
        self.train_wsa_ids = [x.id for x in train_split.drugs.all()]

    @transaction.atomic
    def _autosplit(self):
        from browse.models import DrugSplit
        clusters = KtSplit(self.ws, self.input_drugset_name, 2).clusters
        test_split = DrugSplit.objects.create(
                ws=self.ws,
                name=self.test_name,
                )
        test_split.drugs.add(*clusters[0].drugs())
        test_split.save()
        train_split = DrugSplit.objects.create(
                ws=self.ws,
                name=self.train_name,
                )
        train_split.drugs.add(*clusters[1].drugs())
        train_split.save()
        


class KtSplit(object):
    def __init__(self, ws, drugset_name, num_clusters):
        self.ws = ws
        self.ws_id = ws.id
        self.drugset_name = drugset_name
        self.num_clusters = num_clusters
        self._cluster()

    def _make_drug_dpis(self, wsa_ids):
        from browse.models import Workspace, WsAnnotation
        from dtk.prot_map import DpiMapping, AgentTargetCache
        ws = self.ws
        wsas = [WsAnnotation.objects.get(pk=x) for x in wsa_ids]
        targ_cache = AgentTargetCache(
                mapping=DpiMapping(ws.get_dpi_default()),
                agent_ids=[wsa.agent.id for wsa in wsas],
                dpi_thresh=ws.get_dpi_thresh_default(),
                )

        pairs = []
        for wsa in wsas:
            agent = wsa.agent
            for uniprot, gene, direction in targ_cache.info_for_agent(agent.id):
                pairs.append((wsa.id, uniprot))
        from dtk.data import MultiMap
        drug_dpis = MultiMap(pairs)
        return drug_dpis


    def _cluster(self):
        # Going to compute a 'natural' clustering of drugs, then try to
        # fit to the number of clusters requested.
        # A totally different, more holistic approach could be more effective.

        print("Pulling drug dpis for ", self.drugset_name)
        wsa_ids = self.ws.get_wsa_id_set(self.drugset_name)
        drug_dpis = self._make_drug_dpis(wsa_ids)
        print(f"Making maps for {len(wsa_ids)} wsas")

        drug2dpis = drug_dpis.fwd_map()
        dpi2drugs = drug_dpis.rev_map()

        for wsa_id in wsa_ids:
            if not wsa_id in drug2dpis:
                drug2dpis[wsa_id] = set()

        natural_clusters = []

        print("Pulling clusters")
        clustered_drugs = set()
        for drug, dpis in drug2dpis.items():
            if drug in clustered_drugs:
                continue
            cluster = make_cluster(drug, drug2dpis, dpi2drugs)
            clustered_drugs.update(cluster.drugs())
            natural_clusters.append(cluster)

        # If we have more than 2 clusters, let's merge down to 2.
        # If we have fewer than 2, let's just bail out for now
        if len(natural_clusters) < self.num_clusters:
            print("Couldn't general enough clusters, will have empties")
            self.clusters = natural_clusters
            for _ in range(len(natural_clusters), self.num_clusters):
                self.clusters.append(Cluster([], {}))
            return
        
        # Merge clusters by size, to try to balance groups.  Use id as
        # secondary for stability.
        # Would instead be better to merge based on a similarity metric.
        natural_clusters.sort(key=lambda x: (len(x.drugs()), x.id()))
        N = self.num_clusters
        merged_clusters = [
                Cluster.merge(natural_clusters[i::N])
                for i in range(N)
                ]
                # XXX rather than assigning clusters round-robin, it might
                # XXX be better to always assign the largest unmerged
                # XXX natural cluster to the smallest output cluster
        self.clusters = merged_clusters

        cluster_sizes = [len(x.drugs()) for x in self.clusters]
        assert len(wsa_ids) == sum(cluster_sizes), f'Weird counts: {len(wsa_ids)} vs {cluster_sizes}'

class ParsedSplitDrugsetName(object):
    def __init__(self, name):
        self.drugset_name = name
        self.is_split_drugset = name.startswith('split-')
        if self.is_split_drugset:
            parts = name.split('-')
            prefix, split_type = parts[:2]
            input_drugset_name = '-'.join(parts[2:])
            self.split_type = split_type
            self.input_drugset = input_drugset_name
            if split_type == 'test':
                compl_type = 'train'
            else:
                compl_type = 'test'
            self.complement_drugset = '-'.join([prefix, compl_type, input_drugset_name])

def parse_split_drugset_name(name):
    return ParsedSplitDrugsetName(name)

def is_split_drugset(name):
    return ParsedSplitDrugsetName(name).is_split_drugset

def get_split_drugset(name, ws):
    splitds = ParsedSplitDrugsetName(name)
    ktsplit = EditableKtSplit(ws, splitds.input_drugset)
    if splitds.split_type == 'test':
        return set(ktsplit.test_wsa_ids)
    elif splitds.split_type == 'train':
        return set(ktsplit.train_wsa_ids)
    else:
        assert False, "Unknown split type %s" % splitds.split_type


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run KT Split')
    parser.add_argument('-w', '--ws-id', type=int, required=True, help="Workspace id")
    parser.add_argument('-n', '--num-clusters', type=int, default=2, help="Number of clusters to create.")
    parser.add_argument('-d', '--drugset', default='kts', help="Drugset to split.")
    args = parser.parse_args()
    from browse.models import Workspace
    ws = Workspace.objects.get(pk=args.ws_id)
    kts = KtSplit(ws, args.drugset, args.num_clusters) 

    clusters = kts.clusters
    for cluster in clusters:
        print("-- Cluster -- ")
        print(cluster.dpi_table_str())

