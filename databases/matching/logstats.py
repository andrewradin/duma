#!/usr/bin/env python3

from dtk.lazy_loader import LazyLoader

# 2021.01.05-15:55:53 - unmodified
# 2021.01.05-16:09:47 - preserve std_smiles
# 2021.01.05-17:31:56 - strip single clusters
# 2021.01.07-15:59:54 - name suffix exclusion
# 
# Done:
# - some unique std_smiles codes were getting stripped by trim_props, which
#   was then causing those molecules to go through orphan assignment rather
#   than getting their own subgroups; this was fixed, and seems better, but
#   some also causes some small std_smiles differences that were previously
#   ignored to now trigger separate subgroups; this better exposes std_smiles
#   issues
#   - to get more comparable stats with previous runs, we now strip
#     single-drug clusters after cluster building
# - identify biotech flag characteristics:
#   - only 133 of 7769 cluster breaks involved the biotech flag
#     - of 271 "biotech" molecules in those cluster breaks:
#       - 80% (228) were clustered via std_smiles
#         - of these, 64 were in 'mixed' subclusters (so, molecules with
#           identical std_smiles were labeled biotech in one source and
#           small molecule in another)
#       - only 43 were orphans (lacking std_smiles, and therefore potentially
#         getting assigned to a small_molecule cluster even though they were
#         biotech molecules
#       - taken together, this means at most 40% of biotech molecules were
#         labeled in a meaningful way, and at least 60% were labeled in error
#   - only 160 additional clusters which weren't already suspicious could be
#     flagged for cluster breaking based on mixed biotech flags
#     - only 22 of these didn't have a std_smiles code; the ones I spot-checked
#       were labeling inconsistencies
# - produce a list of possibly irrelevant std_smiles difference using a text
#   distance metric
# - log promiscuous std_smiles codes in match_detail.log for investigation as
#   possible SMILES standardization issues
# - remove names that end with a word indicating they're a likely class of
#   chemicals, rather than a single chemical
# To Do:
# - an alternative way to identify irrelevant std_smiles differences is to
#   examine subset prop overlap logs:
#   - inchi or inchi_key overlaps imply strong structural similarity
#   - others are probably worth reviewing as well, both to identify std_smiles
#     issues, and to identify untrustworthy linking properties
# - fix the CAS issue with pubchem (PLAT-3637)
# - sometimes drugbank data is associated with a salt rather than with the
#   main molecule record -- maybe retrieve if main molecule data is missing?
#   (e.g. DB12755)
# - there are no 'big' gains to be gotten from the biotech flag; some
#   low-hanging fruit might be:
#   - experiment with chembl and drugbank ETL to see if some other field is
#     a better proxy for flagging a molecule as biotech (or for providing an
#     alternative match check); mol_weight? mol_formula? presence of sequence
#     data (drugbank only -- as opposed to the presence of an empty sequences
#     tag, which is what we check now)? don't flag certain classes of molecule
#     (e.g. drugbank allergens seem particularly useless and troublesome)?
#   - do a final review to see if there are identifiable patterns under
#     biotech mismatches that reliably flag actual mis-clustering; these
#     might affect orphan assignment as well
# - conceptually, it's tempting to flag all mixed biotech clusters as
#   suspicious, but the current clustering algorithm won't do anything to
#   break them up (since they have only 0 or 1 potential subgroups); we
#   could look at using a graph-based bad link removal to do cluster breaking
#   both in this case and for orphans with unclear affinity

class ParsedLog:
    def find(self,pattern):
        for line in self.source:
            if line.startswith(pattern):
                return line.rstrip('\n')
        raise RuntimeError(f"'{pattern}' not found")
    def cluster_counts(self):
        line = self.find('total_clusters:')
        clusters = int(line.split()[1])
        line = self.find('   total_drugs:')
        mols = int(line.split()[1])
        return (clusters,mols)
    def __init__(self,logdir):
        import os
        self.source = open(os.path.join(logdir,'match.log'))
        line = self.find('total_keys:')
        self.mol_inputs = int(line.split()[1])
        self.find('building clusters')
        (self.pass_1_clusters,self.pass_1_mols) = self.cluster_counts()
        self.pass_1_unmatched = self.mol_inputs - self.pass_1_mols
        self.find('rebuilding clusters')
        line = self.find('')
        self.cluster_breaks = int(line.split()[0])
        (self.pass_2_clusters,self.pass_2_mols) = self.cluster_counts()
        self.pass_2_unmatched = self.mol_inputs - self.pass_2_mols

class ParsedDetail:
    class BreakAttempt:
        pass
    class Subgroup:
        def type_label(self):
            if self.bt is None:
                return 'mixed'
            if self.bt:
                return 'bio'
            return 'sm'
    def parse_key_set(self,s):
        import json
        return set(json.loads(s.replace("'",'"')))
    def __init__(self,logdir):
        import os
        from dtk.files import get_file_records
        self.break_attempts = []
        bt_decode = {
                'bt:True':True,
                'bt:False':False,
                'bt:None':None,
                }
        for rec in get_file_records(
                os.path.join(logdir,'match_detail.log'),
                keep_header=None,
                parse_type='tsv',
                ):
            if rec[0] == 'rebuild_clusters':
                if rec[1].endswith('biotech with'):
                    self.cur_ba = self.BreakAttempt()
                    self.break_attempts.append(self.cur_ba)
                    self.cur_ba.mixed = rec[1].startswith('mixed')
                    self.cur_ba.subgroups = []
                    self.cur_ba.total_smiles = int(rec[2])
                    self.cur_ba.total_drugs = int(rec[4])
                    self.cur_ba.sm_keys = self.parse_key_set(rec[6])
                    self.cur_ba.bio_keys = self.parse_key_set(rec[7])
            elif rec[0] == 'subset dump':
                sg = self.Subgroup()
                self.cur_ba.subgroups.append(sg)
                sg.smiles = rec[2]
                sg.drug_keys = self.parse_key_set(rec[5])
                sg.bt = bt_decode[rec[1]]

class LogStats(LazyLoader):
    _kwargs=['logdir']
    def _log_loader(self):
        return ParsedLog(self.logdir)
    def _detail_loader(self):
        return ParsedDetail(self.logdir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='match log analysis'
            )
    parser.add_argument('--bio-counts',action='store_true'),
    parser.add_argument('--smiles-diff',type=int),
    parser.add_argument('--dump-non-smiles',action='store_true'),
    parser.add_argument('--dump-single-smiles',action='store_true'),
    parser.add_argument('--dump-multi-smiles',action='store_true'),
    parser.add_argument('logdir'),
    args = parser.parse_args()

    ml = LogStats(logdir=args.logdir)
    print(ml.log.mol_inputs,'molecules input')
    print(ml.log.pass_1_mols,'molecules in',
            ml.log.pass_1_clusters,'pass 1 clusters;',
            ml.log.pass_1_unmatched,'mols unmatched',
            )
    print(ml.log.pass_2_mols,'molecules in',
            ml.log.pass_2_clusters,'pass 2 clusters;',
            ml.log.pass_2_unmatched,'mols unmatched',
            )
    print(ml.log.cluster_breaks,'cluster break attempts yielded',
            ml.log.pass_2_clusters-ml.log.pass_1_clusters,
            'new clusters',
            )

    
    # XXX We could also potentially look at the similarity of smiles codes
    # XXX within clusters, regardless of the biotech flag, to attempt to
    # XXX get a feeling for whether smiles standardization isn't being
    # XXX aggressive enough. The python 'textdistance' module might be
    # XXX useful here.

    # cases of interest:
    cases = {
            1:'clusters without biotech differences',
                # these are handled by the pre-biotech cluster breaking code,
                # and serve as a baseline for the relative improvement that
                # might be available by using the biotech flag
            2:'clusters with biotech differences only',
                # these don't have std_smiles differences, and so currently
                # don't get marked as suspect; we might possibly get better
                # clusters by splitting biotech from non-biotech; presumably
                # there's a single std_smiles that goes with the non-biotech
                # molecules in the cluster
            3:'clusters with biotech and smiles differences',
                # these are split by the pre-biotech code, but biotech
                # molecules might be incorrectly assigned or dropped as
                # orphans, rather than landing in a pure-biotech cluster
            }
    from collections import defaultdict,Counter
    by_case = defaultdict(list)
    for ba in ml.detail.break_attempts:
        if len(ba.subgroups) < 2:
            assert ba.mixed
            case = 2
        elif ba.mixed == True:
            case = 3
        else:
            case = 1
        by_case[case].append(ba)
    # summarize case counts (with optional histogram)
    indent = '   '
    for case in sorted(cases):
        print(cases[case]+':',len(by_case[case]))
        if False:
            # XXX This is mostly interesting when key (cluster size) is
            # XXX small (which characterizes the more prevalent small
            # XXX clusters) or large (which shows the size and distribution
            # XXX of outliers). Maybe figure out a better way to summarize.
            ctr = Counter()
            for ba in by_case[case]:
                ctr[ba.total_drugs] += 1
            print(indent,'cluster size histogram:')
            for key in sorted(ctr.keys()):
                print(indent,key,ctr[key])
    # show selected case detail
    def exemplar(s):
        priority=['DB','CHEMBL']
        def score(k):
            for i,prefix in enumerate(priority):
                if k.startswith(prefix):
                    return len(priority)-i
            return 0
        best = None
        for key in s:
            if best is None:
                best = key
            elif score(key) > score(best):
                best = key
        return best
    def show_keyset(label,s):
        if not s:
            print(label,'empty')
            return
        best = exemplar(s)
        import dtk.url
        if best.startswith('DB'):
            url = dtk.url.drugbank_drug_url(best)
        else:
            url = dtk.url.chembl_drug_url(best)
        print(label,len(s),'drug(s) including',url)
    if args.dump_non_smiles:
        # no smiles; these are ignored by pre-biotech code
        for ba in (x for x in by_case[2] if x.total_smiles == 0):
            print('mixed biotech;',ba.total_drugs,
                    'with',ba.total_smiles,'smiles')
            show_keyset('  bio:',ba.bio_keys)
            show_keyset('  sm:',ba.sm_keys)
        # XXX examples with 0 smiles:
        # XXX - DB11599 (cornstarch) and DB10913 (tapioca starch) are linked
        # XXX   by a common CAS number (9005-25-8, which seems to be just
        # XXX   starch, regardless of origin); they have different UNIIs
        # XXX - DB14838 (Telisotuzumab) is linked by multiple synonyms to
        # XXX   CHEMBL3545419, which is labeled as a small molecule. Presumably
        # XXX   they're the same, but there's not enough information about
        # XXX   either
        # XXX - DB05085 (TM30339) is linked to CHEMBL3545106 via synonyms.
        # XXX   there's not much info about either. In this case the chembl
        # XXX   molecule is classified as a protein and the drugbank molecule
        # XXX   is labeled as a small molecule, but described as a hormone
        # XXX   analogue
    if args.dump_single_smiles:
        # only one smiles; these are ignored by pre-biotech code
        for ba in (x for x in by_case[2] if x.total_smiles != 0):
            print('mixed biotech;',ba.total_drugs,
                    'with',ba.total_smiles,'smiles')
            show_keyset('  bio:',ba.bio_keys)
            show_keyset('  sm:',ba.sm_keys)
        # XXX examples with 1 smiles:
        # XXX - DB15093 (somapacitan) is linked by multiple synonyms to
        # XXX   CHEMBL3707290, which is labeled as a small molecule despite
        # XXX   a mol weight of 1308. The mol formula is also inconsistent
        # XXX   between the two records (C1038H1609N273O319S9 vs
        # XXX   C55H97N13O19S2).
        # XXX - DB04976 (M40403) is linked to CHEMBL3544979 via HY-13336
        # XXX   name m40403; these might be the same, although chembl labels
        # XXX   it a small molecule and drugbank a biotech
        # XXX - PUBCHEM444041 has the synonym 'caraway' that links
        # XXX   DB03995 (Betadex) to DB10671 (Caraway Seed). The latter
        # XXX   record is really sparse, but appears to be there for
        # XXX   allergenic testing, so the connection is probably spurious.
        # XXX   They have different UNIIs (which we don't collect).
        # XXX - PUBCHEM9001 and 9002 (Tiron and Tiron free acid) link to
        # XXX   DB13236 (Stibopen) via std_smiles, but have the synonyms
        # XXX   bax 1526 and chymopapain which link to DB06752 (Chymopapain)
        # XXX   which is clearly a biologic
        # XXX - DB03518 (mevalonic acid) links to DB15483 (Modified vaccinia
        # XXX   ankara) via the PUBCHEM449 synonym "mva"
        # XXX - DB03514 (4-Vinylguaiacol) links to DB10669 (Allspice) via
        # XXX   PUBCHEM332 synonym 'Allspice"
        # XXX   (this follows the first example where the drugbank biotech
        # XXX   is an alergen without any real information on its composition;
        # XXX   we could consider filtering these (via "biologic
        # XXX   classification")
        # XXX - DB01539 (1-Piperidinocyclohexanecarbonitrile) links to DB11330
        # XXX   (Factor IX Complex) via PUBCHEM62529 synonym 'pcc'
        # XXX - DB00113 arcitumomab is linked to DB14227 Tc-99m, which is the
        # XXX   radioisotope it is labeled with; linked via multiple variations
        # XXX   on its name associated with PUBCHEM26476
        # XXX   that seems to match what's in chembl
        # XXX - DB12755 and CHEMBL3544954 appear to be the same molecule,
        # XXX   although chembl calls it a protein and drugbank a small
        # XXX   molecule
        # XXX   NOTE that although DB12755 hasn't got much data at the top
        # XXX   level, there's a <salts> tag in the XML that has a more
        # XXX   fully-described entry for the acetate form, including smiles
    if args.dump_multi_smiles:
        # more than one smiles; these are processed by pre-biotech code,
        # and so have per-smiles subset data available; also, any mixed biotech
        # subset will be set to bt:None
        for ba in by_case[3]:
            print('mixed biotech;',ba.total_drugs,
                    'with',ba.total_smiles,'smiles')
            # accumulate all assigned keys
            assigned = set()
            for sg in ba.subgroups:
                assigned |= sg.drug_keys
            show_keyset('  orphan bio:',ba.bio_keys-assigned)
            show_keyset('  orphan sm:',ba.sm_keys-assigned)
            for sg in ba.subgroups:
                label = '  '+sg.type_label()+' assigned'
                show_keyset(f'{label} bio:',ba.bio_keys & sg.drug_keys)
                show_keyset(f'{label} sm:',ba.sm_keys & sg.drug_keys)
        # XXX examples:
        # XXX - same molecule smiles w/ inconsistent type labeling (DB12199
        # XXX   vs CHEMBL4297447)
        # XXX - it's possible that chembl vs drugbank smiles will result in
        # XXX   some minor std_smiles difference; if there's also a sm/bio
        # XXX   labeling difference, we'll get two subsets with bt:True and
        # XXX   bt:False rather than a single bt:None subset. The above
        # XXX   doesn't necessarily print anything useful in that case.
        # XXX   an example is DB01278 vs CHEMBL2103758, which is labeled as
        # XXX   a small molecule by drugbank
    if args.smiles_diff:
        pairwise_comparisons = []
        from dtk.text import diffstr
        for ba in by_case[3]:
            for i,sg in enumerate(ba.subgroups):
                for j in range(i):
                    other = ba.subgroups[j]
                    pairwise_comparisons.append((
                            diffstr(other.smiles,sg.smiles),
                            exemplar(other.drug_keys),
                            exemplar(sg.drug_keys),
                            ))
        def diff_metric(diff):
            unchanged=sum(len(x[0]) for x in diff)
            changed = sum(len(x[1])+len(x[2]) for x in diff)
            return changed/(changed+unchanged)
        pairwise_comparisons.sort(key=lambda x:diff_metric(x[0]))
        print(len(pairwise_comparisons),'total smiles comparisons')
        print('showing',args.smiles_diff,'most similar pairs:')
        for i in range(args.smiles_diff):
        #for i in range(0,len(pairwise_comparisons),100):
        #for i in range(0,100,5):
            diff,d1,d2 = pairwise_comparisons[i]
            print('  ',diff_metric(diff), d1, d2, [x[1:3] for x in diff])
    if args.bio_counts:
        total_pure = 0
        total_mixed = 0
        total_orphans = 0
        for ba in by_case[3]:
            bio_assigned_pure = 0
            bio_assigned_mixed = 0
            for sg in ba.subgroups:
                bio_assigned = len(ba.bio_keys & sg.drug_keys)
                if sg.bt is True:
                    bio_assigned_pure += bio_assigned
                else:
                    bio_assigned_mixed += bio_assigned
            bio_assigned = bio_assigned_mixed + bio_assigned_pure
            total_pure += bio_assigned_pure
            total_mixed += bio_assigned_mixed
            total_orphans += len(ba.bio_keys) - bio_assigned
        total_bio = total_pure + total_mixed + total_orphans
        print('total bio molecules in cluster breaks:',total_bio)
        print(f'  assigned pure: {total_pure}')
        print(f'  assigned mixed: {total_mixed}')
        print(f'  orphans: {total_orphans}')

