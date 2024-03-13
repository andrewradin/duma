import logging
logger = logging.getLogger(__name__)
class ProtFeature:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class StructFeature:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class IndirectTargets(ProtFeature):
    description = "Common direct & disease-relevant-indirect targets between molecules"
    def generate(self, mol_to_prots):
        prots = {prot for molname, prots in mol_to_prots.items() for prot in prots}

        TOP_N = self.top_n_prots
        dis_prot_ord = sorted(self.dis_prot_sig.items(), key=lambda x: -x[1])
        top_dis_prots = {x[0] for x in dis_prot_ord[:TOP_N]}

        from browse.default_settings import PpiDataset, PpiThreshold
        ppi = PpiDataset.value(ws=self.ws)
        ppi_t = PpiThreshold.value(ws=self.ws)
        from dtk.prot_map import PpiMapping
        ppi_map = PpiMapping(ppi)

        # Look up indirect for each prot
        data = list(ppi_map.get_ppi_info_for_keys(prots, min_evid=ppi_t))
        prot2s = {x.prot2 for x in data if x.prot2 in top_dis_prots} | set(prots)
        from dtk.data import kvpairs_to_dict
        p1_p2_ev = kvpairs_to_dict([(x.prot1, (x.prot2, x.evidence)) for x in data])

        from collections import defaultdict
        prot2_cnt = defaultdict(set)

        if self.apply_dis_score:
            prot_mult = lambda x, is_direct: self.dis_prot_sig.get(x, 0)
        else:
            prot_mult = lambda x, is_direct: 1.0 if is_direct else 0.5

        import pandas as pd
        out = pd.DataFrame(0, columns=prot2s, index=mol_to_prots.keys(), dtype=float) 
        for mol, mol_prots in mol_to_prots.items():
            for mol_prot in mol_prots:
                out.at[mol, mol_prot] = prot_mult(mol_prot, True)
                for prot2, ev in p1_p2_ev.get(mol_prot, []):
                    if prot2 in prot2s:
                        out.at[mol, prot2] = max(out.at[mol, prot2], ev * prot_mult(mol_prot, False))
                        prot2_cnt[prot2].add(mol)
        
        # Filter out any prot2's that only hit 1 mol (unless they're a direct target).
        for prot2, molset in prot2_cnt.items():
            if len(molset) < 2 and prot2 not in prots:
                del out[prot2]

        return out


class Pathways(ProtFeature):
    description = "Disease-enriched pathways that are affected by multiple molecules"

    def prep_d2ps(self, moas):
        from browse.default_settings import PpiDataset
        ppi_id = PpiDataset.value(ws=None)
        from dtk.d2ps import D2ps
        d2ps = D2ps(ppi_id, self.ps_id)
        d2ps.update_for_moas(moas)
        return d2ps
    
    def get_moa_score(self, d2ps, moa):
        records = d2ps.get_moa_pathway_scores(moa)
        return {x.pathway:x.score for x in records}

    def generate(self, name_to_prots):
        from dtk.d2ps import MoA
        name_to_moa = {
            name: MoA((prot, 1.0, 0.0) for prot in prots)
            for name, prots in name_to_prots.items()
        }
        d2ps = self.prep_d2ps(name_to_moa.values())

        TOP_N = self.top_n_pws
        dis_pw_ord = sorted(self.dis_path_sig.items(), key=lambda x: -x[1])
        top_dis_pw = {x[0] for x in dis_pw_ord[:TOP_N]}

        from collections import defaultdict
        pw2mol = defaultdict(set)
        import pandas as pd
        out = pd.DataFrame(0, columns=top_dis_pw, index=name_to_prots.keys(), dtype=float) 
        for mol, moa in name_to_moa.items():
            pw2score = self.get_moa_score(d2ps, moa)
            for pw, score in pw2score.items():
                if self.apply_dis_score:
                    score *= self.dis_path_sig[pw]
                if pw in top_dis_pw and score > 1e-4:
                    out.at[mol, pw] = max(out.at[mol, pw], score)
                    pw2mol[pw].add(mol)

        for pw in out.columns:
            mols = pw2mol[pw]
            if len(mols) < 2:
                del out[pw]
        return out



class RelatedDiseases(ProtFeature):
    description = "Diseases with clinical treatments involving a target from these molecules (opentargets known_drug)"
    def generate(self, name_to_prots):
        TOP_N_PER_PROT = 10
        prots = {prot for molname, prots in name_to_prots.items() for prot in prots}

        from browse.default_settings import openTargets, efo
        from dtk.open_targets import OpenTargets
        ot = OpenTargets(openTargets.latest_version())
        scores = ot.get_prots_scores(prots, ['known_drug'])
        efo2name = ot.get_disease_key_name_map(None)
        scores_by_prot = [(x[2], x) for x in scores]
        from dtk.data import kvpairs_to_dict
        prot_to_scores = kvpairs_to_dict(scores_by_prot)

        from dtk.disease_efo import load_efo_otarg_graph
        g = load_efo_otarg_graph(efo.latest_version(), sep='_')

        # to_ignore will contain the higher level concepts that are too general to be useful,
        # and that have been replaced with more specific terms.
        to_ignore = set()
        def expand(cur_nodes, to_expand):
            if len(g[to_expand].keys()) > 0:
                to_ignore.add(to_expand)
                cur_nodes.remove(to_expand)
                cur_nodes.update(g[to_expand].keys())
        
        kDisEfo = 'EFO_0000408'
        kPhenEfo = 'EFO_0000651'
        
        cur_nodes = {kDisEfo, kPhenEfo}
        for _ in range(2):
            for x in list(cur_nodes):
                expand(cur_nodes, x)
        extra_expands = [
            'EFO_0000616', # Neoplasm
            'MONDO_0000651', # Thoracic disease -> break into lung, heart, etc.
        ]
        for x in extra_expands:
            expand(cur_nodes, x)
    
        from collections import defaultdict
        mapping = defaultdict(set)
        def assign(cur_node, assignment):
            # Apparently some versions of this data are cyclic, so put a stop to any cycles...
            if cur_node in mapping:
                return
            mapping[cur_node].add(assignment)
            for child in g[cur_node]:
                assign(child, assignment)
        for node in cur_nodes:
            assign(node, node)


        cur_dis = set()
        otargs = self.ws.get_disease_default('OpenTargets').split(',')
        otargs = [x.split('key:')[1] for x in otargs if 'key:' in x]
        for otarg in otargs:
            if otarg in mapping:
                logger.info(f"{otarg} maps to {mapping[otarg]}")
                cur_dis.update(mapping[otarg])
            else:
                logger.info(f"Couldn't find mapping for current disease key {otarg}")


        # prot_scores is a set of tuples:
        # (efo_id, efo_name, uniprot, score_type, score_val)

        kScoreIdx = -1

        missing = set()
        def remapped(efo_key):
            if efo_key not in mapping:
                missing.add(efo_key)
                #return f'({efo2name.get(efo_key, efo_key)})'
                return [('(Other)', None)]
            new_efos = mapping.get(efo_key, efo_key)
            outs = []
            for new_efo in new_efos:
                new_name = efo2name.get(new_efo, new_efo)
                if new_efo in cur_dis:
                    new_name = '*' + new_name
                outs.append((new_name, new_efo))
            return outs

        top_dis_full = set(('*' + efo2name.get(x,x), x) for x in cur_dis)
        for prot, prot_scores in prot_to_scores.items():
            top_prot_scores = sorted(prot_scores, key=lambda x: -x[kScoreIdx])[:TOP_N_PER_PROT]
            if not top_prot_scores:
                continue

            thresh = top_prot_scores[-1][kScoreIdx]
            top_prot_scores = [x for x in prot_scores if x[kScoreIdx] >= thresh]

            efo_keys = [x[0] for x in top_prot_scores]
            for efo_key in efo_keys:
                if efo_key in to_ignore:
                    continue
                top_dis_full.update(remapped(efo_key))
        
        if missing:
            pass
            # raise Exception(f"Missing {len(missing)}")


        from collections import defaultdict
        dis2mol = defaultdict(set)

        # Then strip off the efo and just keep the name.
        top_dis = [x[0] for x in top_dis_full]
        
        import pandas as pd
        out = pd.DataFrame(0, columns=top_dis, index=name_to_prots.keys(), dtype=float) 
        for mol, mol_prots in name_to_prots.items():
            for prot in mol_prots:
                for dis_key, dis_name, prot, scoretype, value in prot_to_scores.get(prot, []):
                    if dis_key in to_ignore:
                        continue
                    remapped_out = remapped(dis_key)
                    for remapped_name, _ in remapped_out:
                        if remapped_name in top_dis:
                            out.at[mol, remapped_name] = max(out.at[mol, remapped_name], value)
                            dis2mol[remapped_name].add(mol)

        # This will remove singletons... but let's keep them in this one for now.
        if False:
            for dis, mols in dis2mol.items():
                if len(mols) < 2:
                    del out[dis]
        
        return out


class StructSimilarity(StructFeature):
    description = "Structural similarity between molecules"
    def generate(self, name_to_wsaid):
        wsa_ids = name_to_wsaid.values()

        from browse.models import WsAnnotation
        wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)
        from scripts.metasim import metasim, load_wsa_smiles_pairs

        smiles = [[v,k] for k, v in load_wsa_smiles_pairs(self.ws, wsas)]

        ms = metasim(refset = wsa_ids,
                     all_drugs=wsa_ids,
                     wts = {'rdkit': 1, },
                     smiles=smiles,
                     )
        ms.setup()
        ms.run()
        ms.full_combine_scores()

        names = name_to_wsaid.keys()
        wsa2name = {v:k for k,v in name_to_wsaid.items()}
        import pandas as pd
        out = pd.DataFrame(0, columns=names, index=names, dtype=float) 
        for name, wsa_id in name_to_wsaid.items():
            for name2, wsa_id2 in name_to_wsaid.items():
                out.at[name, name2] = ms.all_scores[wsa_id][wsa_id2].get('rdkit', -1)

        return out

def cluster(ws, name_to_prots, name_to_wsas, feature_types, **passthrough_args):
    """
    The idea is that we featurize each thing and then run standard clustering algorithms on those features.
    We want to overfeaturize at first, the filter out things not in common, to try to find as many links as possible.
    """

    kwargs = dict(
        ws=ws,
        **passthrough_args,
        )
    features = [
        Type(**kwargs) for Type in feature_types
    ]

    dfs = []
    for feature in features:
        if isinstance(feature, ProtFeature):
            dfs.append(feature.generate(name_to_prots))
        elif isinstance(feature, StructFeature):
            dfs.append(feature.generate(name_to_wsas))
        else:
            raise Exception(f"Unknown feature type for {feature}")

    import pandas as pd
    df = pd.concat(dfs, axis=1)

    return df
