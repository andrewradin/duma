import os
import logging
from path_helper import PathHelper
logger = logging.getLogger(__name__)

# Scorebox tuples are:
# (label,score,max_score,higher,tied,novel_higher,novel_tied,bound_job_info)
# All except the first 2 can be None (or totally absent).
# see templates/moldata/_scorebox_section.html
class ScoreBox:
    def __init__(self):
        self.scores = []
        self.non_novel_keys = None
        self.marker = 0
    def set_non_novel(self,non_novel_keys):
        self.non_novel_keys = non_novel_keys
    def set_marker(self):
        self.marker = len(self.scores)
    def sort_from_marker(self):
        before = self.scores[:self.marker]
        after = self.scores[self.marker:]
        def by_rank(row):
            def get_rank(col):
                import re
                return int(re.match('.*>(.*)<.*',col).group(1))
            return get_rank(row[3])+(get_rank(row[4])/2)
        after.sort(key=by_rank)
        self.scores=before+after
    def add_value(self,label,value):
        from dtk.html import decimal_cell
        self.scores.append( (
                label,
                decimal_cell(value),
                ) )
    def add_from_source(self,src,code,key,non_novel_fmt='%d'):
        from dtk.cache import Cacher
        cacher = Cacher('browse.utils.Scorebox',log_miss=False)
        cache_key = cacher.signature((
                src.label(),
                code,
                key,
                src.bji().job.id,
                non_novel_fmt,
                ))
        row = cacher.check(
                cache_key,
                lambda: self._get_base_row(src,code,key,non_novel_fmt,cache_key)
                )
        if not row:
            return
        row += (
                src.bji(),
                )
        self.scores.append(row)
    def _get_base_row(self,src,code,key,non_novel_fmt,cache_key):
        # The following log will record the key and associated wsa_id for
        # each scoreboard cache entry. These can be post-processed by a
        # script to selectively delete cache entries based on workspace
        # and then-current wsa indication value. This is the only reason
        # cache_key is passed in here, and relies on the fact that this
        # function is called only on a cache miss.
        #
        # The creation timestamp of each file is available, so we
        # can both delete files we know are expired, and put an
        # upper limit on how far back in the log we need to search.
        #
        # This leverages the fact that scorebox cache files are kept in
        # their own directory.
        logger.info('scorebox cache wsa %d key %s',key,cache_key)
        from dtk.html import decimal_cell
        cat = src.bji().get_data_catalog()
        val,attrs = cat.get_cell(code,key)
        if val is None:
            return tuple()
        row = (
                ' '.join([src.label(),cat.get_label(code)]),
                decimal_cell(val,attrs.get('href'),attrs.get('fmt')),
                )
        l = cat.get_ordering(code,True)
        novel_higher = 0
        higher = 0
        novel_tied = 0
        tied = 0
        for item in l:
            if item[1] > val:
                higher += 1
                if self.non_novel_keys \
                            and item[0] not in self.non_novel_keys:
                    novel_higher += 1
            elif item[1] == val:
                tied += 1
                if self.non_novel_keys \
                            and item[0] not in self.non_novel_keys:
                    novel_tied += 1
            else:
                break
        # tied includes this drug, so subtract it (but if the score
        # is zero, we're not really in the list, so don't subtract
        # in that case)
        if tied:
            tied -= 1
        if novel_tied:
            novel_tied -= 1
        row += (
                decimal_cell(l[0][1]),
                decimal_cell(higher,fmt='%d'),
                decimal_cell(tied,fmt='%d'),
                )
        if self.non_novel_keys:
            row += (
                    decimal_cell(novel_higher,fmt=non_novel_fmt),
                    decimal_cell(novel_tied,fmt=non_novel_fmt),
                    )
        else:
            row += ('','')
        return row

def append_tox_scores(scorebox,drug_ws):
    # We want to display tox scores for drug_ws.  However, tox scores
    # are loaded as properties of particular drug collections, and
    # drug_ws.agent may not be in one of those collections (and
    # certainly not in all of them).  So, first, we build a list of
    # all the collections that actually have tox scores:
    from drugs.models import Prop,Collection,Drug
    tox_props = Prop.prefix_properties_list('rt') \
                + Prop.prefix_properties_list('pt')
    tox_collections = Collection.objects.filter(
                            drug__metric__prop__in=tox_props
                            ).distinct()
    # then, build a list of agents that are in those collections,
    # and either are drug_ws.agent, or are linked to drug_ws.agent
    # via an 'm_' property; if drug_ws.agent belongs on the list, we
    # put it first, so if it has a 'local' tox score, it's displayed first
    tox_agents = []
    if drug_ws.agent.collection in tox_collections:
        tox_agents.append(drug_ws.agent)
    for c in tox_collections:
        if c == drug_ws.agent.collection:
            continue
        try:
            for v in getattr(drug_ws.agent,'m_'+c.key_name+'_set'):
                tox_agents.append(
                        Drug.objects.get(
                                tag__prop__name=c.key_name,
                                tag__value=v,
                                )
                        )
        except AttributeError:
            pass
    # now, search for each tox property in all the available agents
    from dtk.html import decimal_cell
    for p in tox_props:
        for a in tox_agents:
            try:
                v = getattr(a,p.name)
                if v is not None:
                    label = p.name
                    if a != drug_ws.agent:
                        label = a.get_key()+' '+label
                    scorebox.add_value(label,v)
            except AttributeError:
                pass

def get_default_mapping(ws, dpi_opts):
    from dtk.prot_map import DpiMapping
    # return the first listed default that applies
    ordered_defaults = (ws.get_dpi_default(), )
    for opt in ordered_defaults:
        if opt in dpi_opts:
            return opt
    # if none apply, choose the first mapping available
    if dpi_opts:
        return dpi_opts[0]
    # if nothing is available, return something arbitrary
    # even though it won't match the drug
    return ordered_defaults[0]

class DrugRunInfo:
    def __init__(self,drug_ws,sl):
        self.sl = sl
        self.wsa = drug_ws
        self.tissue_namers = {}
        self.dpi_data = {}
        self.used_uniprots = set()
        self.dpi_opts = drug_ws.get_possible_dpi_opts()
    def get_tissue_namer(self,job_type):
        try:
            return self.tissue_namers[job_type]
        except KeyError:
            pass
        from browse.models import Tissue
        if job_type == 'path':
            qs = Tissue.objects.filter(ws_id=self.wsa.ws_id)
            tissue_map = dict(
                    qs.values_list('id','name')
                    )
            func = lambda x:tissue_map.get(int(x),"(deleted tissue %s)" % x)
        elif job_type == 'gpath':
            qs = self.wsa.ws.get_gwas_dataset_qs()
            tissue_map = dict(
                    qs.values_list('id','phenotype')
                    )
            func = lambda x:'%s: %s' % (
                    x,
                    tissue_map.get(int(x[4:]),x).replace('_',' '),
                    )
        elif job_type == 'capp':
            func = lambda x:x.replace('_',' ')
        else:
            # if not specifically remapped above,
            # just show what's in the detail file
            func = lambda x:x
        self.tissue_namers[job_type] = func
        return func
    def get_dpi_data(self,mapping):
        try:
            return self.dpi_data[mapping]
        except KeyError:
            pass
        from dtk.prot_map import DpiMapping
        dpi_mapping = DpiMapping(mapping)
        info = (
                dpi_mapping.get_dpi_info(self.wsa.agent),
                dpi_mapping.get_dpi_keys(self.wsa.agent),
                )
        for m in info[0]:
            self.used_uniprots.add(m.uniprot_id)
        self.dpi_data[mapping] = info
        return info
    def append_job_scores(self, scorebox):
        scorebox.set_marker()
        for src in self.sl.sources():
            bji = src.bji()
            cat = bji.get_data_catalog()
            for code in cat.get_codes('wsa','score'):
                try:
                    scorebox.add_from_source(src,code,self.wsa.id)
                except (IOError,):
                    pass
        scorebox.sort_from_marker()
    def get_drug_prot_info(self,mapping,bji):
        # Returns a tuple defining the contents of a Path Information
        # collapse section. The tuple contains:
        # - a list of dpi binding descriptions; each item in the list
        #   is itself a list (probably should be a tuple) containing:
        #   - a DpiBinding object, specifying drug, protein, evidence,
        #     and direction
        #   - a list of dictionaries, each describing a path to a tissue;
        #     the keys are protID, tissue, score, and direction. For
        #     direct paths, the protID is the same as DpiBinding
        #   - a tuple holding any extra column data from the DPI file
        # - the number of DPI bindings that matched a tissue
        # - the total number of DPI bindings
        # - the percentage of DPI bindings that matched a tissue
        # - the column headings for any extra columns in the DPI file
        # - a list of DPI keys matched by this drug
        # - a list of DPI bindings available for this drug in the workspace
        # - the selected DPI binding
        # - the text to use if a DPI binding has no dictionaries tying it
        #   to tissues
        # - bound job info
        # ---------------------------------------
        # for the selected candidate, group all the paths by the
        # protein at the tissue end: { sig_prot: [path,...] }
        # XXX Note that load_index_from_path_file expects a single
        # XXX native key for extracting the info from the path_detail
        # XXX file, but the get_dpi_data returns values for any
        # XXX key in the mapping, possibly duplicating values.
        # XXX This should probably be re-worked to either:
        # XXX - show only data for a single selected native key, or
        # XXX - show data for all distinct proteins, but label them
        # XXX   based on which native keys they're associated with
        # XXX Since proteins should overlap for different native keys
        # XXX in a cluster, maybe there should be 2 distinct tables:
        # XXX - the first lists the DPI data for each key
        # XXX - the second shows the detail network for each protein
        # XXX   mentioned in the list above (which should be the same
        # XXX   in any native key detail that protein appears in)
        index = {}
        no_match_reason = "No matching tissues"
        if bji:
            try:
                tissue_namer = self.get_tissue_namer(bji.job_type)
                load_index_from_path_file(index,self.wsa,bji,tissue_namer)
            except IOError:
                logger.info('path info not available for wsa %d, job %d',
                        self.wsa.id,
                        bji.job.id,
                        )
                no_match_reason = "No path data available"
        # generate a list of all proteins that bind to this drug
        # from the master p2d database; pair the protein with the
        # list of paths that terminate in that protein from above
        # (or an empty list if nothing was found).  This is basically
        # a left outer join from all proteins that could possibly
        # bind to the drug to those that actually did.
        dpi_info,dpi_keys = self.get_dpi_data(mapping)
        prot_list = []
        first_extra_col = 4
        for m in dpi_info:
            # each record is [DpiBinding,[Path,Path,...],[extra columns]];
            # the path list will be empty if this binding isn't significant
            rec = [m, index.setdefault(m.uniprot_id,[]), m[first_extra_col:]]
            prot_list.append(rec)
            # direct uniprots get noted by get_dpi_data; add indirect ones here
            self.used_uniprots |= set([
                    d['protID']
                    for d in rec[1]
                    ])
        prot_list.sort(key=lambda x: x[0].evidence, reverse=True)
        ###
        # now, add match counts and extra column labels
        matched = 0
        total = len(prot_list)
        for p in prot_list:
            if len(p[1]) > 0:
                matched +=1
        if total:
            row_info = (matched
                ,total
                ,100.0*matched/total
                ,prot_list[0][0]._fields[first_extra_col:] # extra column names
                )
        else:
            row_info = (0,0,0,[])
        return (prot_list,)+ row_info + (
                dpi_keys,
                self.dpi_opts,
                mapping,
                no_match_reason,
                bji,
                )
    def get_protmap_options(self):
        out = []
        for src in self.sl.sources():
            if src.bji().has_target_detail():
                out.append((src.label(), str(src.bji().job.id)))
        if not out:
            out.append(("default", "paths_0"))
        return out

    def make_path_protlist(self, job_id, mapping):
        # Add items to the meta_prot_list. Each item in the meta_prot_list is
        # a tuple defining a path information collapse section (i.e. the detail
        # of a pathsum-like job). The tuple elements are:
        # - a label for the section
        # - a unique code for the section, for labeling html elements
        # - an embedded tuple holding the information to display, as returned
        #   by get_drug_prot_info() above
        #
        # added an inner function here to convert the direction number present burried in
        # get_drug_prot_info's output to an arrow for more intuitive viewing.
        # The arrow is added as an additional object to avoid losing any data
        def direction_to_arrow(prot_list):
            from dtk.plot import dpi_arrow
            for j in range(len(prot_list[0])):
                prot_list[0][j].append(dpi_arrow(prot_list[0][j][0].direction))
                for i in range(len(prot_list[0][j][1])):
                    prot_list[0][j][1][i]['direction_arrow'] = dpi_arrow(prot_list[0][j][1][i]['direction'])
            return prot_list
        from runner.process_info import JobInfo
        bji = JobInfo.get_bound(self.wsa.ws, job_id) 
        try:
            (ts,target) = bji.get_target_detail(self.wsa)
        except IOError as ex:
            return (None,'???','???',None,None,[str(ex)])

        except JobInfo.AmbiguousKeyError as ex:
            # The following results in an empty path section,
            # with the error message displayed in place of
            # the list of matched keys. This at least allows
            # other data on the drug page to be accessed, while
            # still calling attention to the problem.
            return (None,'???','???',None,None,[str(ex)])
        if ts is None or target is None:
            return (None,'???','???',None,None,[])

        my_mapping = mapping \
                or bji.job.settings().get('p2d_file','') \
                or get_default_mapping(self.wsa.ws,self.dpi_opts)

        try:
            return direction_to_arrow(self.get_drug_prot_info(my_mapping, bji))
        except IOError as ex:
            return (None,'???','???',None,None,[str(ex)])


    def append_prot_maps(self, meta_prot_list, mapping):
        # Add items to the meta_prot_list. Each item in the meta_prot_list is
        # a tuple defining a path information collapse section (i.e. the detail
        # of a pathsum-like job). The tuple elements are:
        # - a label for the section
        # - a unique code for the section, for labeling html elements
        # - an embedded tuple holding the information to display, as returned
        #   by get_drug_prot_info() above
        #
        # added an inner function here to convert the direction number present burried in
        # get_drug_prot_info's output to an arrow for more intuitive viewing.
        # The arrow is added as an additional object to avoid losing any data
        def direction_to_arrow(prot_list):
            from dtk.plot import dpi_arrow
            for j in range(len(prot_list[0])):
                prot_list[0][j].append(dpi_arrow(prot_list[0][j][0].direction))
                for i in range(len(prot_list[0][j][1])):
                    prot_list[0][j][1][i]['direction_arrow'] = dpi_arrow(prot_list[0][j][1][i]['direction'])
            return prot_list
        from runner.process_info import JobInfo
        for src in self.sl.sources():
            # load path info corresponding to each pathsum score
            try:
                (ts,target) = src.bji().get_target_detail(self.wsa)
            except IOError:
                continue
            except JobInfo.AmbiguousKeyError as ex:
                # The following results in an empty path section,
                # with the error message displayed in place of
                # the list of matched keys. This at least allows
                # other data on the drug page to be accessed, while
                # still calling attention to the problem.
                meta_prot_list.append( (src.label()
                                , "paths_%d" % src.bji().job.id
                                , (None,'???','???',None,None,[str(ex)])
                                ) )
                continue
            if ts is None or target is None:
                continue
            my_mapping = mapping \
                    or src.bji().job.settings().get('p2d_file','') \
                    or get_default_mapping(self.wsa.ws,self.dpi_opts)
            try:
                meta_prot_list.append( (src.label()
                                , "paths_%d" % src.bji().job.id
                                , direction_to_arrow(self.get_drug_prot_info(
                                            my_mapping,
                                            src.bji(),
                                            ))
                                ) )
            except IOError:
                pass # tolerate missing (obsolete) dpi files
        if not meta_prot_list:
            my_mapping = mapping \
                    or get_default_mapping(self.wsa.ws,self.dpi_opts)
            meta_prot_list.append( ("default"
                                , "paths_0"
                                , direction_to_arrow(self.get_drug_prot_info(
                                            my_mapping,
                                            None,
                                            ))
                                    ) )

    def append_gsig_data(self, gesig_list, mapping, type):
        from dtk.scores import Ranker,SourceList
        import runner.data_catalog as dc
        from dtk.plot import dpi_arrow
        # if a mapping wasn't explicitly specified, choose one
        if not mapping:
            mapping = get_default_mapping(self.wsa.ws,self.dpi_opts)
        dpi_info,_ = self.get_dpi_data(mapping)
        prot_list = list(dpi_info)
        prot_list.sort(key=lambda x:x.evidence,reverse=True)
        extd_protlist = [[x, dpi_arrow(float(x.direction))] for x in prot_list]
        for src in self.sl.sources():
            if src.bji().job_type != type:
                continue
            cat = src.bji().get_data_catalog()
            codes = tuple(cat.get_codes('uniprot','score'))
            subtitle = src.bji().get_subtitle()
            if not subtitle:
                subtitle = src.label()

            gesig_list.append( (
                        (src.bji().job,type+'_%d'%src.bji().job.id),
                        subtitle,
                        [cat.get_label(c) + ' (Score, Higher, Tied, Lower)' for c in codes],
                        [
                            (
                                x,
                                tuple((
                                    [cat.get_cell(code,x[0].uniprot_id)] +
                                    [Ranker(cat.get_ordering(code, True)).get_details(x[0].uniprot_id)]
                                    for code in codes if cat.get_cell(code,x[0].uniprot_id)[0]
                                    ))
                            )
                            for x in extd_protlist
                        ]
                            ) )

    def append_esga_data(self, esga_list, mapping):
        from dtk.plot import dpi_arrow
        for src in self.sl.sources():
            # load info corresponding to each esga score
            if src.bji().job_type != 'esga':
                continue
            if not mapping:
                mapping = src.bji().job.settings().get('dpi_file','')
            dpi_info,_ = self.get_dpi_data(mapping)
            prot_list = list(dpi_info)
            prot_list.sort(key=lambda x:x.evidence,reverse=True)
            import pickle
            rundir = os.path.join(PathHelper.storage,
                                    str(self.wsa.ws.id),
                                    'esga',
                                    str(src.bji().job.id)
                            )
            try:
                with open(os.path.join(rundir,'output','out.pickle'), 'rb') as handle:
                    pr_d = pickle.load(handle)
            except (IOError,):
                continue # tolerate missing data
            agg_type = 'prMax'
            from scripts.connect_drugs_to_proteinSets import establish_prefixes
            _, _, prot_prefix = establish_prefixes()
            prot_list_tup = []
            for x in prot_list:
                if prot_prefix+x.uniprot_id in pr_d:
                    prot_list_tup.append((x,
                                          tuple([pr_d[prot_prefix+x.uniprot_id]]),
                                          dpi_arrow(float(x.direction))
                                        ))
            esga_list.append( (
                        (src.bji().job,'esga_%d'%src.bji().job.id),
                        src.label(),
                        [agg_type],
                        prot_list_tup
                        ))

    def get_sb_ranks(self):
        data = self.wsa.get_all_prscrn_ranks()
        from dtk.html import glyph_icon, join
        header = ['Prescreen Name', 'Prescreen ID',
                  'Primary JID', 'Eff Jid',
                  'Created on', 'Created by',
                  'Rank', 'Eff Rank']
        table_data=[]
        for row in data:
            bji_job = row[2]
            note = bji_job.get_note_text()
            job_nbr = str(bji_job.id)
            if note:
                job_nbr = join(glyph_icon('comment',hover=note),job_nbr)
            eff_bji_job = row[3]
            table_row = row[0:2]
            table_row += [job_nbr,eff_bji_job.id]
            table_row += [str(x) for x in row[4:]]
            table_data.append(table_row)
        return(header,table_data)

    def append_drug_score_data(self, drug_score_list, type):
        from dtk.files import get_file_records
        from dtk.scores import Ranker,SourceList
        import runner.data_catalog as dc
        for src in self.sl.sources():
            # load info corresponding to each esga score
            if src.bji().job_type != type:
                continue
            bji = src.bji()
            cat = bji.get_data_catalog()
            result = []
            higher = []
            tied = []
            lower = []
            labels = []

            for code in cat.get_codes('wsa', 'score'):
                score = cat.get_cell(code, self.wsa.id)[0]
                if score is None:
                    continue
                result.append(score)

                ranker = Ranker(cat.get_ordering(code, True))
                ranks = ranker.get_details(self.wsa.id)
                higher.append(ranks[0])
                tied.append(ranks[1])
                lower.append(ranks[2])
                labels.append(cat.get_label(code))

            if not result:
                continue

            drug_score_list.append( (
                        (src.bji().job,'drug_score_%d'%src.bji().job.id),
                        src.label(),
                        ['Score type', 'Score',
                        'Higher', 'Tied', 'Lower'],
                        zip(labels, result, higher, tied, lower),
                        ) )

    def append_defus_data(self, defus_list):
        from dtk.files import get_file_records
        from dtk.scores import Ranker,SourceList
        import runner.data_catalog as dc
        for src in self.sl.sources():
            # load info corresponding to each esga score
            if src.bji().job_type != 'defus':
                continue
            bji = src.bji()
            score_file = bji.outfile
            cat = bji.get_data_catalog()
            higher = []
            tied = []
            lower = []

            for code in cat.get_codes('wsa', 'score'):
                ranker = Ranker(cat.get_ordering(code, True))
                ranks = ranker.get_details(self.wsa.id)
                higher.append(ranks[0])
                tied.append(ranks[1])
                lower.append(ranks[2])

            connections = extract_defus_connections(self.wsa.id, bji)

            defus_list.append( (
                        (src.bji().job,'defus_%d'%src.bji().job.id),
                        src.label(),
                        ['Score type', 'Connecting drug', 'Score',
                        'Higher', 'Tied', 'Lower'],
                        zip(list(connections.keys()), list(connections.values()), higher, tied, lower),
                        ) )

def get_saved_drug_scores_dict(sl):
    scores={}
    for bji in sl.sources_bound_jobs():
        scores.setdefault(bji.job_type,set()).add(bji.job.id)
    return scores

def get_wzs_jid_qparm(sl, pscr, jid_only=False):
    jid = None
    if pscr:
        jid = pscr.eff_jid()

    acceptable_code = 'wzs'
    if jid is None:
        d = get_saved_drug_scores_dict(sl)
        if acceptable_code in d and len(d[acceptable_code]) == 1:
            jid = list(d[acceptable_code])[0]
    if jid:
        from dtk.duma_view import qstr
        if jid_only:
            return jid
        else:
            return qstr({acceptable_code+'_jid': jid})
    return ''

def extract_defus_connections(wsa_id, bji):
    connections = {}
    header = None
    result = []
    from dtk.files import get_file_records
    score_file = bji.outfile
    for frs in get_file_records(score_file,
                                keep_header = True,
                                select=([str(wsa_id)],0),
                                ):
        if header is None:
            header = frs
            continue
        result.append(frs)
    if not result:
        return connections
    assert len(result) == 1, "Found multiple records with same wsa"

    from browse.models import WsAnnotation
    for k,v in zip(header, result[0]):
        if k == 'wsa':
            continue
        if k.endswith('Score'):
            name = k[:-len('Score')]
            if v == '0.0':
                continue
            connections[name] = [float(v)]
        elif k.endswith('ConnectingDrug'):
            assert k == name+'ConnectingDrug'
            if name in connections:
                connections[name].append(WsAnnotation.all_objects.get(pk=v))
        else:
            print('WARNING: DEFUS expects all columns to either be a WSA, ' +
                    'or end in Score or ConnectingDrug. ' +
                    'That was not the case: ' + k
                    )
    return connections

class TrgImpParmBuilder:
    trgimp_scores = {'path':'psm_jid',
                     'codes':'cds_jid',
                     'depend':'dpd_jid',
                     'esga':'esg_jid',
                     'gpath':'gph_jid',
                     'capp':'cap_jid',
                    }
    def __init__(self):
        self._output = {} # {job_type:set(job_ids),...}
        self._prescreen_id = None
    def extract_as_qparms(self):
        from dtk.duma_view import qstr
        # prefer a shortened prescreen_id form if applicable
        if self._prescreen_id:
            return qstr(dict(prescreen_id=self._prescreen_id))
        # else build up a set of categorized job ids
        d = self._output
        qparms = {}
        for k in self.trgimp_scores:
            if k in d:
                v = d[k]
                qparms[self.trgimp_scores[k]] = ','.join([str(x) for x in v])
        return qstr(qparms)
    def extract_as_attributes(self,obj):
        d = self._output
        for k in self.trgimp_scores:
            if k in d:
                setattr(obj,self.trgimp_scores[k],list(d[k]))
    def build_from_source_list(self,sl):
        if len(sl.sources()) == 1:
            bji = sl.sources()[0].bji()
            if bji.job_type not in self.trgimp_scores:
                # there's only a single source, and it's not one that trgimp
                # can use; see if it's a prescreen score
                from browse.models import Prescreen
                prescreens= Prescreen.objects.filter(
                        primary_score__startswith='%d_'%bji.job.id,
                        ).values_list('id',flat=True)
                if prescreens:
                    # it is; use the prescreen inputs to build the trgimp parms
                    self.build_from_downstream_bji(bji)
                    # and also stash a prescreen_id for an abbreviated qparm
                    self._prescreen_id=prescreens[0]
                    return
        # build inputs directly from source list
        self._output = get_saved_drug_scores_dict(sl)
    def build_from_downstream_bji(self,bji):
        from runner.models import Process
        qs = Process.objects.filter(
                id__in=bji.get_all_input_job_ids(),
                ).values_list('id','name')
        d = {}
        for job_id,name in qs:
            job_type = name.split('_')[0]
            d.setdefault(job_type,set()).add(job_id)
        self._output = d


def find_prots(ws, drugs, mapping, dpi_t):
    # return {uniprot:[False or dpi_arrow,...],...}
    # where the nth entry in a protein's vector indicates whether the protein
    # binds with the nth drug in the input list and if so has an arrow pointing which direction.
    # if a mapping wasn't explicitly specified, choose one
    from dtk.plot import dpi_arrow
    if mapping is None:
        dpi_opts = [x for d in drugs for x in d.get_possible_dpi_opts()]
        mapping = get_default_mapping(ws, dpi_opts)
    from dtk.prot_map import DpiMapping
    dpi_mapping = DpiMapping(mapping)
    dpi_data = {d:[(m[1], m[3])
                   for m in dpi_mapping.get_dpi_info(d.agent)
                   if float(m[2]) >= dpi_t
                  ]
                 for d in drugs
                }
    protsD = {}
    for i,d in enumerate(drugs):
        for prot_id, dir in dpi_data[d]:
            prot = protsD.setdefault(prot_id,[False]*len(drugs))
            prot[i] = dpi_arrow(float(dir))
    return protsD

def load_index_from_path_file(index,drug_ws,bji,tissue_namer):
    (ts,target) = bji.get_target_detail(drug_ws)
    if ts is None or target is None:
        return
    # for each result type
    for idx in target.paths:
        # find the relevant column indicies
        colmap = {}
        for i,colname in enumerate(ts.columns[idx]):
            colmap[colname] = i
        tissue_col = colmap['t2p:tissue']
        score_col = colmap['t2p:evidence']
        dir_col = colmap['t2p:direction']
        prot_col = colmap['t2p:protein']
        key_col = prot_col
        # The labeling of the linking column changed when we started
        # supporting multiple PPI files, and extracting the column name
        # from the file header (see 96eccb5 and related). Check for
        # both names so we can handle old and new paths files.
        for indr_col in ('p2p:prot1','p2p:prot2',):
            if indr_col in colmap:
                key_col = colmap[indr_col]
                break
        # Now, for direct, key_col == prot_col.  For indirect, key_col
        # holds the drug-side protein and prot_col holds the tissue-side
        # protein.
        #
        # for each path of this type
        for vec in target.paths[idx]:
            # Save per-path info for the template, which
            # expects these four fields
            p = {
                'tissue':tissue_namer(vec[tissue_col]),
                'score':float(vec[score_col]),
                'direction':float(vec[dir_col]),
                'protID':vec[prot_col],
                }
            key = vec[key_col]
            if key_col != prot_col and key == vec[prot_col]:
                # this is a case where the p2p record binds a protein to
                # itself; don't load it, because the web page will render
                # this as a (possibly duplicate) Direct interaction
                continue
            group = index.setdefault(key,[])
            group.append(p)

