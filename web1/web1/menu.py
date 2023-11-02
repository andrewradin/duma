import logging
logger = logging.getLogger(__name__)

class MenuItem:
    @classmethod
    def join(cls,html_list):
        from django.utils.html import format_html_join
        return format_html_join('',u'{}',[(x,) for x in html_list])
    @classmethod
    def li_list(cls,mi_list,level=0,left=False):
        return cls.join([x.html(level,left) for x in mi_list])
    def __repr__(self):
        return f"<MenuItem:{self.text}>"
    def __init__(self,text,href=None,submenu=None):
        self.text=text
        self.href=href
        self.submenu=submenu
    def html(self,level,left=False):
        import dtk.html as wgt
        # each item's content is an li wrapping one of:
        # - a menu item anchor
        # - header text
        # - a toggle anchor followed by a ul wrapping the next level
        if self.href:
            return wgt.tag_wrap('li',
                    wgt.link(self.text,self.href),
                    )
        elif self.submenu:
            return self.submenu_html(level,left=left)
        else:
            return wgt.tag_wrap('li',
                    self.text,
                    attr={'class':'dropdown-header'},
                    )
    def submenu_html(self,level,left=False):
        import dtk.html as wgt
        # Submenu classes need to be configured differently from top-level
        # dropdowns; adjust that here based on level:
        # - li class is dropdown-submenu
        # - toggle class is submenuopen
        if not level:
            wrap_li_attr = {}
        elif not left:
            wrap_li_attr={'class':'dropdown-submenu'}
        else:
            wrap_li_attr={'class':'dropdown-submenu dropdown-submenu-left'}
        toggle_class='submenuopen' if level else 'dropdown-toggle'
        # A menu can't be scrollable if it contains dropdown submenus; look
        # ahead to figure that out as well
        ul_class='dropdown-menu'
        if not any(item.submenu for item in self.submenu):
            ul_class+=' scrollable-menu'
        toggle=wgt.tag_wrap('a',
                self.join([
                        self.text,
                        wgt.tag_wrap('span','',attr={'class':'caret'}),
                        ]),
                attr={
                        'class':toggle_class,
                        'data-toggle':'dropdown',
                        'role':'button',
                        'aria-expanded':'false',
                        'href':'#',
                        },
                )
        return wgt.tag_wrap('li',
                self.join([
                        toggle,
                        wgt.tag_wrap('ul',
                                self.li_list(self.submenu,level+1,left=left),
                                attr={
                                        'class':ul_class,
                                        'role':'menu',
                                        },
                                ),
                        ]),
                attr=wrap_li_attr,
                )

def build_menu(name,context):
    if name == 'ws_main':
        return build_ws_main_menu(context)
    elif name == 'other':
        return build_other_menu(context)
    elif name == 'main':
        return build_main_menu(context)
    else:
        raise NotImplementedError("unknown menu '%s'"%name)

def build_main_menu(context):
    MI=MenuItem
    from django.urls import reverse
    menu = [
        MI('Index',href=reverse('index')),
        MI('Upload',href=reverse('upload')),
        MI('Compare',href=reverse('nav_ws_cmp')),
        MI('Version Defaults',href=reverse('vdefaults')),
        MI('Drug Edit Proposals',href=reverse('drug_edit_review_view')),
        MI('My Review Notes',href=reverse('rvw:all_review_notes')),
        MI('Cross-Workspace Data', submenu=[
            MI('Molecule Hits',href=reverse('hits')),
            MI('Retrospective',href=reverse('retrospective')),
            MI('Ind. Ft. Explorer',href=reverse('selectabilityfeatureplot')),
            MI('Suitability',href=reverse('suitability')),
            MI('Pathway Network', href=reverse('pathway_network')),
            MI('Ongoing CTs',href=reverse('xws:ongoct')),
            MI('Retro CT Stats',href=reverse('xws:retroct')),
            ]),

        ]
    return MenuItem.li_list(menu)

def build_other_menu(context):
    ws=context.get('ws')
    user=context['user']
    MI=MenuItem
    from django.urls import reverse
    submenu = [
        MI('Account',submenu=[
            MI('Logout',href=reverse('mysite_logout')),
            MI('Password Change',href=reverse('pwchange')),
            MI('2FA',href=reverse('two_factor:profile')),
            ]),
        ]
    if ws:
        submenu += [
            MI('Jobs',href=ws.reverse('jobsum')),
            MI('Protein Search',href=ws.reverse('prot_search')),
            ]
    else:
        submenu += [
            MI('Jobs',href=reverse('nws_jobsum')),
            ]
    infosub = []
    if ws:
        infosub += [
            MI('Workflow',href=ws.reverse('wf_diag')),
            ]
    infosub += [
        MI('Credits',href=reverse('credits')),
        MI('Glossary',href='https://twoxar.app.box.com/file/512665177333'),
        ]
    if user.is_staff:
        infosub += [
            MI('FAQ',href="https://2xar-my.sharepoint.com/personal/carl_twoxar_com/_layouts/15/WopiFrame.aspx?sourcedoc={66078677-6D9F-4E84-96D0-ACABDFDD1C88}&file=DUMA%20FAQs.docx&action=default"),
            MI('Feedback',href="https://docs.google.com/a/eastshore.com/forms/d/1SBYXzBl1chAvwSvVf4URSYUBJT_zBeMSTCSvEPQ12JU/viewform?c=0&w=1"),
            ]
    submenu += [
        MI('Info',submenu=infosub)
        ]
    toolsub=[]
    if ws:
        toolsub += [
            MI('Text Diff',href=ws.reverse('nav_textdiff')),
            MI('Users',href=ws.reverse('users')),
            ]
    else:
        toolsub += [
            MI('Users',href=reverse('nws_users')),
            ]
    if user.is_staff:
        toolsub += [
            MI('Admin',href=reverse('admin:index')),
            MI('Dashboard',href=reverse('dashboard')),
            MI('ConsultantView',href=reverse('consultant_view')),
            ]
    submenu += [
        MI('Tools',submenu=toolsub)
        ]
    if user.is_staff:
        devsub = [
            MI('ETL Status',href=reverse('etl_status')),
            MI('S3 Cache',href=reverse('s3_cache')),
            MI('Collection Stats',href=reverse('coll_stats')),
            ]
        from path_helper import PathHelper
        if PathHelper.cfg('machine_type') == 'dev':
            devsub += [
                MI('Test Report',href='/publish/tests/tests.html'),
                MI('Coverage Report',href='/publish/tests/coverage/index.html'),
                ]
        submenu += [
            MI('Developers',submenu=devsub)
            ]
    menu = [
        MI('Other',submenu=submenu),
        ]
    return MenuItem.li_list(menu, left=True)

def build_ws_main_menu(context):
    # extract needed stuff from context
    ws=context['ws']
    function_root=context['function_root']
    from runner.process_info import JobCrossChecker,JobInfo
    try:
        jcc=context['job_cross_checker']
    except KeyError:
        jcc=JobCrossChecker()
    # some abbreviations to make definition more compact
    MI=MenuItem
    wsrev=ws.reverse
    # list of workspaces for index dropdown
    from browse.models import Workspace
    ws_list = Workspace.objects.filter(active=True).order_by('name')
    # list of jobs for run dropdown
    run_groups=[
            ('Gene Expression',['aesearch','path','gesig']),
            ('Genetics',['esga','gpath', 'gwasig', 'tcgamut']),
            ('Clinical',['faers','capp', 'defus', 'faerssig']),
            ('Phenotype',['mips', 'misig']),
            ('External',['otarg', 'dgn', 'agr', 'customsig']),
            ('Shared Tools',['gpbr','sigdif','codes','glf', 'depend']),
            ('Aggregation',['fvs','fdf','wzs','ml', 'compositesig', 'apr','ctretro']),
            ('Review',['flag', 'lbn', 'trgscrimp', 'selectability', 'selectabilitymodel', 'selectivity', 'dnprecompute']),
            ('Similarity',['struct','prsim', 'jacsim']),
            ]
    archived = ['glee', 'tsp', 'synergy', 'rankdel']
    submenus={}
    fixed_names=['Workflows','Other CMs', 'Archive']
    wfs={}
    wf_order=[]
    for subtype in jcc.ordered_ws_jobnames(ws):
        ubi = JobInfo.get_unbound(subtype)
        if not ubi.in_run_menu:
            continue
        label = ubi.source_label(subtype)
        item=MI(label,href=wsrev('nav_job_start',subtype))
        plugin=subtype.split('_')[0]
        used=False
        for group,plugins in run_groups:
            if plugin in plugins:
                submenus.setdefault(group,[]).append(item)
                used=True
        if not used:
            if plugin in archived:
                submenus.setdefault('Archive',[]).append(item)
            elif plugin == 'wf':
                # For workflows, source_label returns the bare workflow name,
                # so the statement below builds an index from a workflow name
                # to the menu item for the job start page
                wfs[label]=item
                wf_order.append(label)
            else:
                submenus.setdefault('Other CMs',[]).append(item)
    submenus['Workflows']=[]
    priority_wfs = [
            'CombinedGEEvalFlow',
            'CombinedGWASEvalFlow',
            'RefreshFlow',
            'ReviewFlow',
            'CandidateFlow',
            'AnimalModelFlow',
            ]
    for l in priority_wfs:
        if l not in wfs:
            # This shouldn't happen, we've seen examples of it on the jobsum
            # page; see https://app.slack.com/client/T0A4TK4AF/C798RS33J
            # It doesn't make sense, because the 'for subtype' loop above
            # should see every jobname in the workspace, including
            # wf_<ws_id>_<wf_name> for each workflow in Workflow.wf_list().
            # That in turn should populate that workflow into the wfs dict.
            logger.debug(f"'{l}' missing from wfs")
            logger.debug(f"request: {context['request']}")
            logger.debug(f"wfs: {wfs}")
            logger.debug(f"submenus: {submenus}")
            logger.debug(f"ordered_ws_jobnames: {jcc.ordered_ws_jobnames(ws)}")
        submenus['Workflows'].append(wfs[l])
    submenus['Archive'] += [
            wfs[l]
            for l in wf_order
            if l not in priority_wfs
            ]
    run_submenu=[
            MI(group,submenu=submenus[group])
            for group in [x[0] for x in run_groups]+fixed_names
            ]
    user=context['user']
    # pending review votes
    from browse.models import Vote
    needed_elections=Vote.needed_election_list(user)
    from django.urls import reverse
    # assemble menu
    extra_dataset_items=[]
    if user.groups.filter(name='duma_admin').exists():
        extra_dataset_items.append(
                MI('Copy From Another WS',href=wsrev('wsmgr:copy_ws')),
                )
    menu = [
        MI('Index',submenu=[
            MI(w.name,href=w.reverse(function_root))
            if w != ws else
            MI(w.name)
            for w in ws_list
            ]),
        MI('Workflow',href=wsrev('workflow')),
        MI('Datasets',submenu=[
            MI('Import Drug Collections',href=wsrev('nav_col2')),
            MI('Drug Sets',submenu=[
                MI('Manage Drug Sets',href=wsrev('nav_ds')),
                MI('Compare Drug Sets',href=wsrev('nav_ds_dpi_cmp')),
                MI('View Drug Sets',href=wsrev('drugset')),
                MI('Split Alignment',href=wsrev('nav_split_align')),
                ]),
            MI('Protein Sets',submenu=[
                MI('Manage Protein Sets',href=wsrev('nav_ps')),
                MI('Compare Protein Sets',href=wsrev('nav_ps_cmp')),
                MI('View Protein Sets',href=wsrev('protset_view')),
                ]),
            MI('Record Counts',href=wsrev('data_status')),
            MI('WS Custom DPI',href=wsrev('wsadmin:custom_dpi')),
            ]+extra_dataset_items),
        MI('Disease',submenu=[
            MI('Summary',href=wsrev('nav_disease_sum')),
            MI('Names',href=wsrev('nav_disease_names')),
            MI('Version Defaults',href=wsrev('ws_vdefaults')),
            MI('KT Search',href=wsrev('kts_search')),
            #MI('KT Clusters',href=wsrev('nav_kt_clusters')),
            MI('Search Omics',href=wsrev('ge:ae_search')),
            MI('Gene Expression',href=wsrev('ge:tissues')),
            MI('GWAS Search',href=wsrev('gwas_search')),
            MI('Competition',href=wsrev('competition')),
            ]),
        MI('Run',submenu=run_submenu),
        MI('Score',submenu=[
            MI('Scoreboard',href=wsrev('nav_scoreboard')),
            MI('Score Correlation',href=wsrev('nav_score_corr')),
            MI('Score Metrics',href=wsrev('nav_score_metrics')),
            MI('Score Sets',href=wsrev('nav_scoreset_list')),
            MI('Refresh QC',href=wsrev('nav_refresh_qc')),
            MI('Score Plot',href=wsrev('nav_scoreplot', 'wsa')),
            MI('Compare',href=wsrev('nav_score_cmp', 'wsa')),
            MI('FAERS',href=wsrev('faers_base')),
            MI('PCA',href=wsrev('pca')),
            MI('WZS Weight Compare',href=wsrev('score:weight_cmp')),
            MI('WZS MRMR Compare',href=wsrev('score:mrmr_cmp')),
            MI('Feature Pairs',href=wsrev('score:feat_pair')),
            MI('Feature Pair Heatmap',href=wsrev('score:feat_pair_heat')),
            MI('Cross-workspace Compare',href=wsrev('nav_xws_cmp')),
            MI('Pathways',href=wsrev('pathways')),
            MI('Protein Scores',href=wsrev('protein_scores')),
            MI('SelectabilityFeatures',href=reverse('selectabilityfeatureplot')),
            ]),
        MI('Review',submenu=[
            MI('Pre-screens',href=wsrev('nav_prescreen_list')),
            MI('Pre-screen Flags',href=wsrev('flagset')),
            MI('Review Clusters',href=wsrev('nav_rvw_clusters')),
            MI('Patent Search',href=wsrev('pats_search')),
            MI('Initial Predictions',href=wsrev('rvw:review')),
            MI('Hits',href=wsrev('rvw:review')+'?flavor=patent'),
            MI('Hit Clusters',href=wsrev('rvw:hitclusters')),
            MI('Hit Selection',href=wsrev('moldata:hit_selection')),
            MI('Animal Model Compare',href=wsrev('rvw:animal_model_compare')),
            MI('Selected Notes',href=wsrev('patent_notes')),
            MI('Appendix Notes',href=wsrev('patent_notes')+'?appendix=True'),
            MI('My Review Notes',href=wsrev('review_notes')),
            ]),
        ]
    if needed_elections:
        import dtk.html as wgt
        from dtk.text import fmt_time
        menu += [
            MI(wgt.tag_wrap('font','Votes',attr={'color':'red'}),submenu=[
                    MI(
                        fmt_time(e.due)+' '+e.ws.name,
                        href=e.ws.reverse('rvw:election', e.id),
                        )
                    for e in needed_elections
                    ]),
            ]
    return MenuItem.li_list(menu)

