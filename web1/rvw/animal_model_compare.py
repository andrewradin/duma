

"""
Each feature type should return a summary, a (possibly None) details, and a (possibly None) numerics.

Summary is a single value that gets inserted into the table cell.
Details are what gets displayed on click of that cell.
    'type' can be either per_cell, or per_column
        'per_column' expects a list of dicts with the details data to display, e.g.:
            [{'gene': 'DRD2', 'value1': 1, 'value2': 2},
            {'gene': 'DRD3', 'value1': 0, 'value2': None} ]
            The same details will be displayed for any cell in the column.

        'per_cell' expects a list of data, one per row, each cell formatted the same as the per_column above.
Numerics are used for building a classifier, should be either a float or list of float per protein.

"""

class Name:
    def generate(self, wsa_prots, **kwargs):
        names = [wsa.get_name(False) for wsa, prot in wsa_prots]
        ids = [str(wsa.id) for wsa, prot in wsa_prots]
        return names, None, ids

class Gene:
    def generate(self, wsa_prots, ws, uni2gene, **kwargs):
        names = [uni2gene.get(prot, prot) for wsa, prot in wsa_prots]
        urls = [ws.reverse('protein', prot) for wsa, prot in wsa_prots]
        from dtk.html import link
        return [link(name, url) for name, url in zip(names, urls)], None, names


class Orthology:
    description = "How conserved the gene is in the model species.  See protein page for other species and more details"
    def generate(self, wsa_prots, ws, species, **kwargs):
        from dtk.orthology import get_ortho_records
        all_prots = [prot for wsa,prot in wsa_prots]

        prot_to_ortho = {}
        for entry in get_ortho_records(ws, uniprots=all_prots):
            if entry['organism'].lower() != species.lower():
                continue
            prot_to_ortho[entry['uniprot']] = entry['similarity(%)']

        out = []
        num_out = []
        for prot in all_prots:
            ortho = prot_to_ortho.get(prot, 0)
            out.append(f'{ortho:.1f}%')
            num_out.append(ortho / 100)

        return out, None, num_out

class ConservedPpi:
    description = "How conserved indirect targets of this gene are in the model species.  First number is their average homology, second is the jaccard of >0.9 PPI in human STRING vs species STRING (requires orthov3+)"
    def generate(self, wsa_prots, ws, species, ppi, uni2gene, **kwargs):
        from dtk.orthology import get_ortho_records
        from collections import defaultdict
        all_prots = [prot for wsa,prot in wsa_prots]
        all_ppi_prots = set(all_prots)
        ppis = defaultdict(set)

        from dtk.prot_map import PpiMapping
        ppi_map = PpiMapping(ppi)
        for p1, p2, ev, dr in ppi_map.get_ppi_info_for_keys(all_prots, min_evid=PpiMapping.default_evidence):
            ppis[p1].add(p2)
            all_ppi_prots.add(p2)

        prot_to_ppijacc = {}
        prot_to_sim = {}
        for entry in get_ortho_records(ws=None, uniprots=all_ppi_prots):
            if entry['organism'].lower() != species.lower():
                continue
            prot_to_sim[entry['uniprot']] = float(entry['similarity(%)'])
            if 'ppi_jacc' in entry:
                prot_to_ppijacc[entry['uniprot']] = float(entry['ppi_jacc'])

        out = []
        all_details = []
        num_out = []
        for prot in all_prots:
            sim_sum = 0
            ppijacc = prot_to_ppijacc.get(prot, None)
            if ppijacc is not None:
                ppijacc = f'{ppijacc:.3f}'
            prot_details = []
            for ind_prot in ppis[prot]:
                sim = prot_to_sim.get(ind_prot, 0)
                sim_sum += sim
                prot_details.append({
                    'Ind Gene': uni2gene.get(ind_prot, ind_prot),
                    'Ortho%': f'{sim:.1f}'
                })

            N = len(ppis[prot])
            sim_mean = sim_sum / N if N > 0 else 0
            out.append(f'{sim_mean:.1f}% | {ppijacc}')
            num_out.append([sim_mean / 100, float(ppijacc) if ppijacc else 0])
            all_details.append(prot_details)


        details = {
            'type': 'per_cell',
            'data': all_details,
        }
        return out, details, num_out


class BestRanker:
    """Ranker that returns the best rank for each item amongst a list of rankers."""
    def __init__(self, rankers):
        self.rankers = rankers
        for ranker in rankers.values():
            assert ranker.none_if_missing, "Initialize rankers with none_if_missing"

    def get(self, prot):
        out = None
        for ranker in self.rankers.values():
            rank = ranker.get(prot)
            if rank is None:
                continue
            if out is None or out > rank:
                out = rank
        if out is None:
            # Propagate NaN out of here instead of None, fewer special cases.
            out = float('nan')
        return out

    def get_all(self, prot):
        out = {}
        for name, ranker in self.rankers.items():
            out[name] = ranker.get(prot)
        return out

def _make_animal_ranker(ws, amod_ss_id):
    from browse.models import ScoreSet
    from runner.process_info import JobInfo
    from dtk.scores import Ranker
    amod_ss = ScoreSet.objects.get(pk=amod_ss_id)
    jobtype_to_jid = amod_ss.job_type_to_id_map()
    rankers = {}
    # These wouldn't actually feed into the end WZS results.
    CODE_SKIPLIST = [
        'fold',
        'tisscnt',
        'absDir',
        'codesMax', # any codesMax will be redundant with the input prot score
    ]
    # NOTE: If you ran the animal model with uniprot dpi file, you'll get uniprot scores
    # for what are normally wsa-scored CMs, so this will include things like pathsum and gpbr.
    # This does also end up double-counting e.g. gesig + gesig_codes, but should have same results.
    for jobtype, jid in jobtype_to_jid.items():
        bji = JobInfo.get_bound(ws, jid)
        score_codes = list(bji.get_data_catalog().get_codes('uniprot', 'efficacy'))
        for code in score_codes:
            if code in CODE_SKIPLIST:
                continue
            ord = bji.get_data_catalog().get_ordering(code, True)

            if '_gesig_' in jobtype:
                # gesig has negatives, we just want the rank of the abs value.
                ord = [(id, abs(ev)) for id, ev in ord]
                ord.sort(key=lambda x: -x[1])

            # Assume we're running with uniprot jobs here, have to map from wsa to prot.
            uni_wsas = [x[0] for x in ord]
            ranker = Ranker(ord, none_if_missing=True)
            rankers[f'{jobtype}_{code}'] = ranker
    return BestRanker(rankers)

class TopPpi:
    description = 'Fraction of top-scored PPIs of this gene that are still top-scored in the animal model, weighted by human model score'
    def generate(self, ppi, wsa_prots, human_glf_jid, ws, amod_ss_id, top_n_prots, uni2gene, **kwargs):
        # Get top prots.
        from runner.process_info import JobInfo
        glf_bji = JobInfo.get_bound(ws, human_glf_jid)
        input_score = glf_bji.parms['input_score']
        prot_job_id,code = input_score.split('_')
        bji = JobInfo.get_bound(ws, prot_job_id)
        ord = bji.get_data_catalog().get_ordering(code, True)
        prot2score = dict(ord)
        from dtk.scores import Ranker
        human_ranker = Ranker(ord, none_if_missing=True)
        animal_ranker = _make_animal_ranker(ws, amod_ss_id)

        from dtk.prot_map import PpiMapping
        ppi_map = PpiMapping(ppi)
        out = []
        from collections import defaultdict
        all_details = []
        num_out = []
        for _, prot in wsa_prots:
            in_human_cnt = 0
            in_both_cnt = 0
            in_human_weightsum = 0
            in_both_weightsum = 0

            prot_details = []
            for p1, p2, ev, dr in ppi_map.get_ppi_info_for_keys([prot], min_evid=PpiMapping.default_evidence):
                p2_rank = human_ranker.get(p2)
                in_human = p2_rank is not None and p2_rank < top_n_prots
                p2_animal_rank = animal_ranker.get(p2)
                in_animal = p2_animal_rank is not None and p2_animal_rank < top_n_prots
                if in_human:
                    in_human_cnt += 1
                    in_human_weightsum += prot2score[p2]
                if in_animal and in_human:
                    in_both_cnt += 1
                    in_both_weightsum += prot2score[p2]

                if in_human:
                    details = {
                        'gene': uni2gene.get(p2, p2),
                        'human_rank': p2_rank,
                        'best_anim_rank': p2_animal_rank,
                    }
                    details.update(animal_ranker.get_all(p2))
                    prot_details.append(details)

            if in_human_weightsum == 0:
                in_human_weightsum = 1
            perc = f'{in_both_weightsum*100/in_human_weightsum:.0f}%'
            out.append(f'{perc} (counts: {in_both_cnt}/{in_human_cnt})')
            all_details.append(prot_details)
            num_out.append(in_both_weightsum / in_human_weightsum)
        details = {
            'type': 'per_cell',
            'data': all_details,
        }
        return out, details, num_out


class TopRank:
    description = 'Best animal model rank across all scores for this target'
    def generate(self, wsa_prots, ws, amod_ss_id, uni2gene, **kwargs):
        animal_ranker = _make_animal_ranker(ws, amod_ss_id)

        out = []
        all_details = []
        num_out = []
        for _, prot in wsa_prots:
            rank = animal_ranker.get(prot)
            out.append(f'{rank}')
            num_out.append((1000 - rank) / 1000)
            details = {
                'gene': uni2gene.get(prot, prot),
                'best rank': rank,
                }
            for job, rank in animal_ranker.get_all(prot).items():
                details[job + ' rank'] = rank
            all_details.append(details)

        details = {
            'type': 'per_column',
            'data': all_details,
        }
        return out, details, num_out


class Pathways:
    def __init__(self):
        self.code = 'wFEBE'

    def prep_d2ps(self, moas, ppi_id, ps_id):
        from browse.default_settings import PpiDataset
        ppi_id = PpiDataset.value(ws=None)
        from dtk.d2ps import D2ps
        d2ps = D2ps(ppi_id, ps_id)
        d2ps.update_for_moas(moas)
        return d2ps

    def get_moa_score(self, d2ps, moa):
        records = d2ps.get_moa_pathway_scores(moa)
        return {x.pathway:x.score for x in records}

    def make_pathway_ranker(self, ws, jids):
        from runner.process_info import JobInfo
        from dtk.scores import Ranker
        rankers = {}
        for jid in jids:
            bji = JobInfo.get_bound(ws, jid)
            ps_id = bji.parms['std_gene_list_set']
            ord = bji.get_data_catalog().get_ordering(self.code, True)
            ranker = Ranker(ord, none_if_missing=True)
            rankers[f'{bji.job.role}_{self.code}'] = ranker
        return BestRanker(rankers), ps_id

    def get_animal_model_jids(self, amod_ss_id):
        from browse.models import ScoreSet
        amod_ss = ScoreSet.objects.get(pk=amod_ss_id)
        self.jobtype_to_jid = amod_ss.job_type_to_id_map()
        return [jid for jobtype, jid in self.jobtype_to_jid.items() if jobtype.endswith('_glf')]

    def get_corr(self):
        from dtk.data import kvpairs_to_dict
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.ws, self.human_glf_jid)
        ps_id = bji.parms['std_gene_list_set']
        human_scores = bji.get_data_catalog().get_ordering(self.code, True)

        data1 = []
        masked_data1 = []
        top_pws=[]
        for x in human_scores:
            if x[1] <= 0.:
                continue
            data1.append(x[1])
            human_rank = self.human_pathway_ranker.get(x[0])
            if human_rank is not None and human_rank < self.top_n_pathways:
                masked_data1.append(x[1])
                top_pws.append(x[0])
        corrs = []
        lookup = {v:k for k,v in self.jobtype_to_jid.items()}
        from scipy.stats import spearmanr
        for jid in self.animal_glf_jids:
            bji = JobInfo.get_bound(self.ws, jid)
            d = kvpairs_to_dict(bji.get_data_catalog().get_ordering(self.code, True))
            data2 = [d.get(x[0], 0.) for x in human_scores if x[1] > 0.]
            val,p = spearmanr(data1, data2)
            masked_data2 = [d.get(x, 0.) for x in top_pws]
            masked_val, masked_p = spearmanr(masked_data1, masked_data2)
            corrs.append([lookup[jid], round(val, 3), round(masked_val, 3)])
        to_return = sorted(corrs, key = lambda x: x[2], reverse=True)
        return to_return, [len(data1), len(masked_data1)]


class AnimalPathways(Pathways):
    description = 'Score-weighted portion of the top-scored Pathways in each animal pathway score that are hit by this molecule; averaged across pathway scores'
    def generate(self, wsa_prots, ppi, human_glf_jid, amod_ss_id, ws, top_n_pathways, **kwargs):
        from dtk.data import kvpairs_to_dict
        from runner.process_info import JobInfo

        wsa_to_prots = kvpairs_to_dict(wsa_prots)

        from dtk.d2ps import MoA
        wsa_to_moa = {
            wsa: MoA((prot, 1.0, 0.0) for prot in prots)
            for wsa, prots in wsa_to_prots.items()
        }

        out = []
        all_details = []
        num_out = []
        from dtk.url import pathway_url_factory
        pw_url_factory = pathway_url_factory()

        from dtk.gene_sets import get_pathway_id_name_map
        pw_to_name = get_pathway_id_name_map()

        animal_glf_jids = self.get_animal_model_jids(amod_ss_id)

# this order of nesting isn't the most efficient, but it should be fast enough and it's easiest
        for wsa, _ in wsa_prots:
            moa = wsa_to_moa[wsa]
            wsa_details = []
            scores = []
            pws = set()
            in_score_cnt=0
            in_score_sum=0
            for jid in  animal_glf_jids:
                animal_pathway_ranker, ps_id = self.make_pathway_ranker(ws, [jid])
                d2ps = self.prep_d2ps(wsa_to_moa.values(), ppi, ps_id)
                pw2score = self.get_moa_score(d2ps, moa)
                for pw, score in pw2score.items():
                    score_rank = animal_pathway_ranker.get(pw)
                    in_score = score_rank is not None and score_rank <= top_n_pathways
                    if not in_score:
                        continue
                    in_score_cnt += 1
                    in_score_sum += score
                    pws.add(pw)

                # now get the denominator: the sum of scores  the top_n_pathways
                bji = JobInfo.get_bound(ws, jid)
                ord = bji.get_data_catalog().get_ordering(self.code, True)[:top_n_pathways]
                denom = sum([x[1] for x in ord])

                scores.append(in_score_sum/denom)

    # add detail section
            meta_animal_pathway_ranker, _ = self.make_pathway_ranker(ws, animal_glf_jids)
            for pw in pws:
                pw_url = pw_url_factory(pw)
                pw_name = pw_to_name.get(pw)
                details = {'pw': f"<a href='{pw_url}'>{pw_name}</a>"}
                details.update(meta_animal_pathway_ranker.get_all(pw))
                wsa_details.append(details)

            avg_score = sum(scores)/len(scores)
            perc = f'{100*avg_score:.0f}%'
            all_details.append(wsa_details)
            num_out.append([avg_score,in_score_cnt/top_n_pathways*len(animal_glf_jids)])
            out.append(f'{perc} (counts: {in_score_cnt}/{top_n_pathways*len(animal_glf_jids)})')
        details = {
            'type': 'per_cell',
            'data': all_details
        }
        return out, details, num_out


class TopPathways(Pathways):
    description = 'Fraction of Pathways top-scored in the human model & hit by this molecule that are still top-scored in the animal model, weighted by human model score'
    def generate(self, wsa_prots, ppi, human_glf_jid, amod_ss_id, ws, top_n_pathways, **kwargs):
        # saving for use with correlation analysis
        self.human_glf_jid = human_glf_jid
        self.ws=ws
        self.top_n_pathways = top_n_pathways
        # Find top DEPEND pathways for each molecule in human score
        # Fraction of those that are still in the amod score at the top?
        from dtk.data import kvpairs_to_dict
        wsa_to_prots = kvpairs_to_dict(wsa_prots)

        # saving for use with correlation analysis
        self.animal_glf_jids = self.get_animal_model_jids(amod_ss_id)
        self.human_pathway_ranker, ps_id = self.make_pathway_ranker(ws, [human_glf_jid])
        animal_pathway_ranker, _ = self.make_pathway_ranker(ws, self.animal_glf_jids)


        from runner.process_info import JobInfo
        human_glf_bji = JobInfo.get_bound(ws, human_glf_jid)
        pw_to_dis_score = {id: score for id, score in human_glf_bji.get_data_catalog().get_ordering('wFEBE', True)}

        from dtk.d2ps import MoA
        wsa_to_moa = {
            wsa: MoA((prot, 1.0, 0.0) for prot in prots)
            for wsa, prots in wsa_to_prots.items()
        }
        d2ps = self.prep_d2ps(wsa_to_moa.values(), ppi, ps_id)

        out = []
        all_details = []
        num_out = []
        from dtk.url import pathway_url_factory
        pw_url_factory = pathway_url_factory()

        from dtk.gene_sets import get_pathway_id_name_map
        pw_to_name = get_pathway_id_name_map()

        for wsa, _ in wsa_prots:
            moa = wsa_to_moa[wsa]
            pw2score = self.get_moa_score(d2ps, moa)

            in_human_cnt = 0
            in_both_cnt = 0
            in_human_sum = 0
            in_both_sum = 0
            wsa_details = []
            for pw, score in pw2score.items():
                human_rank = self.human_pathway_ranker.get(pw)
                in_human = human_rank is not None and human_rank <= top_n_pathways
                animal_rank = animal_pathway_ranker.get(pw)
                in_animal = animal_rank is not None and animal_rank <= top_n_pathways

                if in_human:
                    in_human_cnt += 1
                    in_human_sum += pw_to_dis_score[pw]

                    pw_url = pw_url_factory(pw)
                    pw_name = pw_to_name.get(pw)

                    details = {
                        'pw': f"<a href='{pw_url}'>{pw_name}</a>",
                        'human_rank': human_rank,
                        'best_anim_rank': animal_rank,
                    }
                    details.update(animal_pathway_ranker.get_all(pw))
                    wsa_details.append(details)

                if in_animal and in_human:
                    in_both_cnt += 1
                    in_both_sum += pw_to_dis_score[pw]

            if in_human_sum == 0:
                in_human_sum = 1
            perc = f'{in_both_sum*100/in_human_sum:.0f}%'
            all_details.append(wsa_details)
            num_out.append(in_both_sum/in_human_sum)
            out.append(f'{perc} (counts: {in_both_cnt}/{in_human_cnt})')
        details = {
            'type': 'per_cell',
            'data': all_details
        }
        return out, details, num_out



class GETarget:
    name = 'Sig Tissues'
    description = '# of animal model GE tissues this protein was significantly enriched in'
    def generate(self, wsa_prots, amod_gesig_jobs, uni2gene, **kwargs):
        from browse.models import Tissue
        tissue_ids = []
        for bji in amod_gesig_jobs:
            tissue_ids.extend(bji.tissue_ids)

        tissues = Tissue.objects.filter(pk__in=tissue_ids)
        from collections import defaultdict
        prot_counter = defaultdict(int)
        prot_total = defaultdict(int)

        details_by_tissue = []
        for tissue in tissues:
            u2data = {x[0]:x for x in tissue.sig_results()}
            tissue_prots = set(tissue.sig_prots())
            details = {
                'tissue': tissue.name,
            }
            # Important to use set in the loop to avoid double-counting if we
            # have duplicate prots.
            for prot in set(prot for _, prot in wsa_prots):
                detail = ''
                if prot in tissue_prots:
                    prot_total[prot] += 1
                    detail = '0'
                if prot in u2data:
                    prot_counter[prot] += 1
                    detail = '1'

                details[uni2gene.get(prot, prot)] = detail
            details_by_tissue.append(details)

        out = []
        num_out = []
        for _, prot in wsa_prots:
            cnt = prot_counter[prot]
            total = prot_total[prot]
            perc = cnt/total if total > 0 else 0
            num_out.append(perc)
            out.append(f'{100.*perc:.0f}% ({cnt}/{total})')
        details = {
            'type': 'per_column',
            'data': details_by_tissue
        }
        return out, details, num_out



# TODO add details? Not sure what it would be in this case
class Selectivity:
    description = "A 0-4 heurisitic rewarding drugs that have been empirically shown to only inhibit a few targets. Currently uses only human data"
    def generate(self, wsa_prots, **kwargs):
        from moldata.models import score_selectivity
        scores = []
        for wsa, _ in wsa_prots:
            (score, txt, reasons) = score_selectivity(wsa)
# we might be able to use the reasons for the details section, but I haven't tried yet
            scores.append(score)
        return scores, None, [x/4. for x in scores]


# TODO add details? Not sure what it would be in this case
class Potency:
    description = "A 0-4 score based on thresholded pharmacology experimental results; currently uses only human data"
    def generate(self, wsa_prots, **kwargs):
        from moldata.models import score_potency
        scores = []
        for wsa, prot in wsa_prots:
            score, reasons = score_potency(wsa, [prot])
# we might be able to use the reasons for the details section, but I haven't tried yet
            scores.append(score)
        return scores, None, [x/4. for x in scores]

# TODO add class descriptions, and possibly details
# TODO is there an efficient (i.e. not loading all the scores everytime) way to get this acting more like the other classes?
class HitSel:
    def generate(self, wsa_prots, **kwargs):
        import dtk.molecule_table as MT
        from browse.models import WsAnnotation
        relevant_selection_scores = ['Pharmacokinetics', 'Dosing, RoA', 'Tolerability']
        output_scores = {x:[] for x in relevant_selection_scores}
        wsa2prots = {}
        for wsa, prot in wsa_prots:
            if wsa.id not in wsa2prots:
                wsa2prots[wsa.id]=[]
            wsa2prots[wsa.id].append(prot)
        wsa_list = WsAnnotation.objects.filter(pk__in=wsa2prots.keys())
        wsas = WsAnnotation.prefetch_agent_attributes(wsa_list)
        max_phase_data = WsAnnotation.bulk_max_phase(wsas)
        score_cols = MT.EditableScoreColumns(max_phase_data)
        score_data, overall_scores = score_cols.make_datas(wsas)
        for extractor, col in score_data:
            for wsa, _ in wsa_prots:
                if col.label not in relevant_selection_scores:
                    continue
                note, score, detail_data = extractor(wsa)
                if (score is None or score == '') and not note:
                    score = 0.
                output_scores[col.label].append(score)
        return relevant_selection_scores, output_scores


def compare(ws, ds, dpi, ppi, pathways, amod_ss_id, glf_jid, top_n_prots, top_n_pathways):
    from browse.models import WsAnnotation, Species, Tissue, ScoreSet

    wsa_ids = ws.get_wsa_id_set(ds)
    wsas = WsAnnotation.objects.filter(pk__in=wsa_ids)

    from dtk.prot_map import AgentTargetCache, DpiMapping
    from runner.process_info import JobInfo
    atc = AgentTargetCache.atc_for_wsas(wsas, ws, dpi_mapping=DpiMapping(dpi))

    amod_ss = ScoreSet.objects.get(pk=amod_ss_id)
    jobtype_to_jid = amod_ss.job_type_to_id_map()
    species = None
    amod_gesig_jobs = []
    for jobtype, jid in jobtype_to_jid.items():
        job_parts = jobtype.split('_')
        if job_parts[-1] == ('gesig'):
            bji = JobInfo.get_bound(ws, jid)
            amod_gesig_jobs.append(bji)
            # we were running into issues if a tissue had subsequently been excluded (it no longer has a species)
            # the easiest fix is to just check the next tissue
            for tid in  bji.tissue_ids:
                try:
                    species_id = Tissue.objects.get(pk=tid).tissue_set.species
                    break
                except:
                    print(f'skipping tissue ID {tid}, as it did not have a species. It was likely excluded')
                    pass
            tissue_species = Species.get('label', species_id)
            assert species is None or species == tissue_species, "Multiple species"
            species = tissue_species
        elif job_parts[0] == 'customsig' and len(job_parts) == 2:
            bji = JobInfo.get_bound(ws, jid)
            species_id = int(bji.parms['species'])
            job_species = Species.get('label', species_id)
            assert species is None or species == job_species, "Multiple species"
            species = job_species


    assert species is not None, "No species found from animalmodel workflow"

    wsa_prots = []
    for wsa in wsas:
        targets = atc.info_for_agent(wsa.agent_id)
        for uni, gene, dr in targets:
            wsa_prots.append((wsa, uni))

    from browse.models import Protein
    u2g = Protein.get_uniprot_gene_map()

    ctx = dict(
        ppi=ppi,
        ws=ws,
        wsa_prots=wsa_prots,
        atc=atc,
        species=species,
        amod_gesig_jobs=amod_gesig_jobs,
        human_glf_jid=glf_jid,
        amod_ss_id=amod_ss_id,
        top_n_prots=top_n_prots,
        top_n_pathways=top_n_pathways,
        uni2gene=u2g,
    )

    # This list is somewhat tightly coupled with the HTML page styling - take a look
    # at the template and make sure the right columns are being merged & styled.
    features = [Name, Gene, Orthology, GETarget, TopPpi, TopPathways, AnimalPathways, TopRank, ConservedPpi, Potency, Selectivity]
    from dtk.table import Table
    rows = [[] for _ in wsa_prots]
    cols = []
    all_details = []
    all_nums = []
    pathway_corrs = None
    for i, feature_cls in enumerate(features):
        feature = feature_cls()
        col, details, numeric = feature.generate(**ctx)
        all_nums.append(numeric)
        all_details.append(details)
        for cell, row in zip(col, rows):
            row.append(cell)
        if hasattr(feature_cls, 'name'):
            name = feature_cls.name
        else:
            name = feature_cls.__name__
        if hasattr(feature_cls, 'description'):
            from dtk.html import glyph_icon
            from django.utils.safestring import mark_safe
            name += glyph_icon('info-sign',hover=feature_cls.description)
            name = mark_safe(name)
        cols.append(Table.Column(name, idx=i))
# XXX tacking this on here to take advantage of the some of the data loaded for pathways
        if feature_cls == TopPathways:
            pathway_corrs,n_paths=feature.get_corr()

    assert pathway_corrs is not None, "TopPathways was expected, but not examined"

# XXX
    # tag on several columns for hit selection.
    # I wrote this separate b/c the generator actually creates multiple columns at once
    # b/c it pulls from code for the hit selection page
    # There is almost certainly a more efficient way to do this, but it's working for now
    feature_cls = HitSel
    feature = feature_cls()
    ft_names, score_dict = feature.generate(wsa_prots)
    for fn in ft_names:
        i+=1
        vals = score_dict[fn]
# all these scores were ranged from 0 to 4, but for later analysis I'd rather it was 0-1
        all_nums.append([v/4. for v in vals])
        all_details.append(None)
        for cell, row in zip(vals, rows):
            row.append(f'{cell:.2f}')
        name = fn
# TODO add a description for the various hit sel types
        cols.append(Table.Column(name, idx=i))
    return Table(rows, cols), all_details, all_nums, pathway_corrs, n_paths
