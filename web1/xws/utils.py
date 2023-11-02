
import logging

logger = logging.getLogger(__name__)

from dtk.lazy_loader import LazyLoader

################################################################################
# ctretro job selection support
################################################################################
def prep_job_list(selected):
    '''Return (filtered,options,messages).

    Prepare a user-supplied list of job ids:
    - remove any ids that don't correspond to successful ctretro CM runs
    - remove all but the latest run from any workspace
    - order list in reverse job_id order
    Returned data:
    filtered - the input list, processed as described above
    options - all potential valid job ids
    messages - description of jobs removed for ws duplication
    '''
    # get all possible input jobs
    name_stem = 'ctretro_'
    from runner.models import Process
    job_options = list(Process.objects.filter(
            name__startswith=name_stem,
            status=Process.status_vals.SUCCEEDED,
            ).order_by('-id'))
    # annotate each job with its ws_id
    for job in job_options:
        job.ws_id = int(job.name[len(name_stem):])
    # filter job_list for duplicate ws_ids
    kept = []
    messages = []
    seen = dict()
    for job in job_options:
        if job.id not in selected:
            continue
        if job.ws_id in seen:
            messages.append(
                f'Ignoring job {job.id}'
                f'; same workspace as {seen[job.ws_id]}'
                )
        else:
            kept.append(job.id)
            seen[job.ws_id] = job.id
    return (kept,job_options,messages)

def get_job_choices(job_options):
    all_ws_ids = set(x.ws_id for x in job_options)
    from browse.models import Workspace
    ws_names = {k:v
            for k,v in Workspace.objects.filter(
                    id__in=all_ws_ids,
                    ).values_list('id','name')
            }
    from dtk.text import fmt_time
    # XXX We may add filtering of the job_choices list by date or
    # XXX length. At that point, pass the selected job list, and the
    # XXX range filtering parameters to this function, and do that
    # XXX filtering here, leaving the job_options list unfiltered.
    # XXX Also, make sure that anything already in the selected list
    # XXX makes it into job_choices, even if it's otherwise outside the
    # XXX filter range. This assures those things are de-selectable on
    # XXX the dropdown.
    return [
            (x.id,f'{ws_names[x.ws_id]}({x.ws_id}) {fmt_time(x.completed)}')
            for x in job_options
            ]

def get_molecule_lists(job_options,job_list):
    molecule_lists = {}
    from runner.process_info import JobInfo
    from dtk.ct_predictions import RetroMolData
    for job in job_options:
        if job.id not in job_list:
            continue
        bji = JobInfo.get_bound(job.ws_id,job)
        molecule_lists[job.id] = RetroMolData.from_tsv(bji.moldata_fn)
    return molecule_lists

################################################################################
# Clinical Trial data retrieval classes
################################################################################
class CTDataCache(LazyLoader):
    '''Bulk-fetches CT data on demand, for specified wsa_ids.'''
    _kwargs=['wsa_ids','phase','find_ct_evidence']
    def _wsas_loader(self):
        '''{wsa_id:wsa,...} for all valid wsa_ids.'''
        from browse.models import WsAnnotation
        return {
                x.id:x
                for x in WsAnnotation.objects.filter(pk__in=self.wsa_ids)
                }
    def _workspaces_loader(self):
        '''{ws_id:ws,...} for all workspaces holding any specified wsa_ids.'''
        ws_ids = set(x.ws_id for x in self.wsas.values())
        from browse.models import Workspace
        return {
                x.id:x
                for x in Workspace.objects.filter(pk__in=ws_ids)
                }
    def _ctas_loader(self):
        '''{wsa_id:latest_cta,...} for any specified wsa_ids with CTAs.'''
        from moldata.models import ClinicalTrialAudit
        result = {}
        for ws_id in self.workspaces:
            result.update({
                    x.wsa_id:x
                    for x in ClinicalTrialAudit.get_latest_ws_records(ws_id)
                    })
        return result
    def _nonctas_loader(self):
        '''set of wsa IDs that did not have a CTA'''
        return set(self.wsa_ids) - set([k for k in self.ctas])

    def _no_urls_loader(self):
        '''set of wsa IDs that did not get a URL via the CTAs'''
       # a little hacky, but this starts empty and is built in the CTAs loader
        return set()

    def _to_rescue_loader(self):
        '''{wsa_id:wsa, ...} any wsas to attempt to find a relevant CT for'''
        to_return={}
        possible_wsas = self.nonctas.union(self.no_urls)
        for k in possible_wsas:
            if k in self.wsas: # there are apparently some molecules with CTAs that we're no longer interested in
                to_return[k] = self.wsas[k]
        return to_return

    def _rescuable_indications_loader(self):
        '''a list of indication names that are ok to attempt to save CTs for'''
        always_list = ['Clinically used treatment', 'FDA Approved Treatment']
        if self.phase=='ph2':
            return always_list + ['Phase 3 treatment']
        return always_list

    def _cts_loader(self):
        '''{wsa_id:ct_data,...} for any specified wsa_ids with an evidence link.

        This data is derived from parsing the evidence link.
        ct_data is a namedtuple containing:
        - trial_id: clinical trial id for AACT links, None otherwise
        - url: the evidence link itself
        - label: a string for labeling the link
        '''
        from collections import namedtuple
        ReturnType = namedtuple('ReturnType','trial_id url label')
        result = {}
        for k,v in self.ctas.items():
            if self.phase == 'ph2':
                url = v.ph2_url
            elif self.phase == 'ph3':
                url = v.ph3_url
            else:
                raise ValueError('unsupported phase value')
            if not url:
# XXX an example of this is if the phase is Ph2 and there is Ph3 CTA, but no Ph2
# XXX in that case we assume the drug has already passed Ph2 (bc it has a Ph3 CTA)
# XXX and we note that drug as worth trying to 'rescue' a CT for
                if self.phase == 'ph2' and v.ph3_url:
                    self.no_urls.add(k)
                # disabling for now b/c it was getting spammy
                #logger.info(f'for WSA {k} could not find a URL')
                continue
            result[k] = self._finish_ct_parsing(ReturnType, url)

# In this optional block we attempt to assocate drugs
# that do not have CTAs with their Ph2s for this disease.
# We don't normally do this b/c it's possible to introduce false connections.
# So we're being particularly cautious here.
# Specifically, we're only doing this for drugs which are marked as further along than our selected phase
# e.g., if our phase is 3 that'd be clinically used or approved, if its Ph2 it'd also include Ph3
# From there we're taking the earliest completed CT
# at the specified phase for the specific drug using the WS-specific CT disease
        if self.find_ct_evidence:
            from browse.models import WsAnnotation
            enum=WsAnnotation.indication_vals
            # these are for bookkeeping and to debug why we're not matching
            failures={'disease_match':0, 'wrong_phase':0, 'not_completed':0, 'total':0}
            for id,wsa in self.to_rescue.items():
                # get info from auto associated CTs, only from ct.gov tho
                # skip out if this is not further along than our specified phase
                if enum.get('label',wsa.indication) not in self.rescuable_indications:
                    logger.info(f'WSA {wsa.id} in WS {ws.id} not a rescuable indication')
                    continue
                from dtk.aact import lookup_trials_by_molecule
                cts = lookup_trials_by_molecule(wsa)
                if not cts:
                    logger.info(f'Could not find any CTs for WSA {wsa.id} in WS {ws.id}')
                    continue
                # pull out only CTs relevant to this disease and phase
                from browse.models import Workspace
                from dtk.aact import phase_name_to_number
                ws = Workspace.objects.get(pk=wsa.ws_id)
                ct_names = ws.get_disease_default('ClinicalTrials')
                single_ct = None
                # these are for bookkeeping and to debug why we're not matching
                disease_match = False
                wrong_phase = False
                not_completed = False
                for x in cts:
                    for d in x.disease:
                        if d in ct_names:
                            disease_match = True
                            phnum = f'ph{phase_name_to_number(x.phase)}'
                            if self.phase != phnum:
                                wrong_phase=True
                                continue
                            if x.status=='Completed':
                              # get the earliest completion date while we're here
                                if not single_ct:
                                    if x.completion_date:
                                        single_ct=x
                                elif is_earlier(x.completion_date, single_ct.completion_date):
                                    single_ct=x
                            else:
                                # XXX These could be worth exploring in the future if we're wanting to rescue more
                                not_completed=True
                if not single_ct:
                    if not disease_match:
                        failures['disease_match']+=1
                    if wrong_phase:
                        failures['wrong_phase']+=1
                    if not_completed:
                        failures['not_completed']+=1
                    failures['total']+=1
                    continue
                from dtk.url import clinical_trials_url
                url = clinical_trials_url(single_ct.study)
                result[id] = self._finish_ct_parsing(ReturnType, url)
                logger.info(f'for WSA {wsa.id} in WS {ws.id} succesfully rescued: {single_ct.study}')
            logger.info(f'Summary stats for rescure failures: {failures}')
        return result

    def _finish_ct_parsing(self, ReturnType, url):
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Handle the case of an AACT url
        ct_id = parsed.path.split('/')[-1]
        if ct_id.startswith('NCT'):
            return ReturnType(ct_id,url,ct_id)
        # fallback; no id, use host as label
        # XXX Eventually we might support different
        # XXX ID spaces with custom sponsor and completion lookups.
        return ReturnType(None,url,parsed.netloc)

    def _sponsors_loader(self):
        '''{NCT_ID:sponsor,...} for each WSA evidence link.'''
        from dtk.aact import lookup_trial_sponsors
        return lookup_trial_sponsors([
                x.trial_id
                for x in self.cts.values()
                if x.trial_id
                ])
    def _trial_details_loader(self):
        '''{NCT_ID:trial_details,...} for each WSA evidence link.

        trial_details is the same data returned by dtk.aact.trial_details(),
        i.e. a dict with attributed drawn from the AACT 'studies' table.
        '''
        from dtk.aact import trial_details
        return trial_details([
                x.trial_id
                for x in self.cts.values()
                if x.trial_id
                ])
    def _atc_loader(self):
        '''Returns an AgentTargetCache object.

        The ATC is pre-loaded with all the agents associated with any
        of the provided wsa_ids. Targets come from the global default
        DPI file.
        '''
        from browse.default_settings import DpiDataset,DpiThreshold
        # We want a workspace-independent mapping to construct the
        # ATC. We also want a non-moa mapping, because we're using
        # non-moa molecules in the CT drugsets.
        from dtk.moa import un_moa_dpi_variant
        dpi_choice = un_moa_dpi_variant(DpiDataset.value(None))
        from dtk.prot_map import AgentTargetCache,DpiMapping
        return AgentTargetCache.atc_for_wsas(
                wsas = list(self.wsas.values()),
                dpi_mapping = DpiMapping(dpi_choice),
                dpi_thresh = DpiThreshold.value(None),
                )

def is_earlier(a_date, b_date):
    if not a_date or not b_date:
        return None
    from datetime import datetime
    d1 = datetime.strptime(a_date, "%Y-%m-%d")
    d2 = datetime.strptime(b_date, "%Y-%m-%d")
    return d1 < d2


class CTData(LazyLoader):
    '''Fetches data on demand for a single wsa_id.

    Expects a pre-loaded instance of the above bulk fetch class for efficiency.
    '''
    _kwargs=['wsa_id','cache']
    # Base data. Some of these members are named to directly match
    # column names in the OngoCTView. Others load intermediate results
    # used by either the TSV or HTML Table formatters.
    def _wsa_loader(self):
        return self.cache.wsas[self.wsa_id]
    def _ws_loader(self):
        return self.cache.workspaces[self.wsa.ws_id]
    def _workspace_loader(self):
        return f'{self.ws.name}({self.ws.id})'
    def _ct_data_loader(self):
        return self.cache.cts.get(self.wsa_id)
    def _ct_link_loader(self):
        from dtk.html import link
        if self.ct_data:
            return link(self.ct_data.label,self.ct_data.url,new_tab=True)
        return ''
    def _ct_id_loader(self):
        if self.ct_data:
            return self.ct_data.trial_id or '' # could be None
        return ''
    def _sponsor_loader(self):
        try:
            return self.cache.sponsors[self.ct_id]
        except KeyError:
            return 'Unavailable'
    def _completion_loader(self):
        try:
            return self.cache.trial_details[self.ct_id]['completion_date']
        except KeyError:
            return 'Unavailable'
    def _target_data_loader(self):
        return self.cache.atc.info_for_agent(self.wsa.agent_id)
    def _targets_loader(self):
        from dtk.prot_map import protein_link
        from dtk.html import join
        return join(*[
                protein_link(u,g,self.ws,direction=d)
                for u,g,d in self.target_data
                ])
