

from collections import namedtuple
rectuple = namedtuple('TrialRecord',
    'study intervention disease start_year phase status drugs completion_date',
    defaults=('',), # supply default for completion_date for older files
    )
from dtk.data import MultiMap


def phase_name_to_number(phase_name):
    if phase_name in ('Phase 3',):
        phase = 3
    elif phase_name in ('Phase 2','Phase 2/Phase 3'):
        phase = 2
    elif phase_name in ('Phase 1','Phase 1/Phase 2', 'Phase 4'):
        # Phase 4 is surprisingly more like phase 1, in that it is
        # very investigational.
        phase = 1
    else:
        phase = 0
    return phase

def lookup_trials_by_molecule(wsa):
    '''Return a set of rectuples matching wsa.'''
    return lookup_trials_by_molecules([wsa]).fwd_map().get(wsa.id, set())

def lookup_trials_by_molecules(wsas):
    '''Return a MultiMap from wsa_ids to rectuples.'''
    if not wsas:
        return []

    # Get WS.
    ws = wsas[0].ws
    ver = ws.get_dpi_version()

    from drugs.models import Drug
    drugs = [x.agent for x in wsas]

    from drugs.models import Collection,Prop
    # -3 to split off the _id, which is separately added by the next lookup.
    native_prop_names = set(x[:-3] for x in Collection.objects.all().values_list('key_name',flat=True))

    drug_native_keys = Drug.bulk_external_ids(native_prop_names, version=ver, drugs=drugs)

    all_native_keys = set(key for keys in drug_native_keys for key in keys)
    trial_key_mm, trial_name_mm = lookup_trials_by_native_keys(all_native_keys)
    all_trials = trial_key_mm.fwd_map().keys()
    trials_diseases_mm = lookup_trial_diseases(all_trials)

    trials2studies = trial_details(all_trials)
    
    out = []
    for wsa, native_keys in zip(wsas, drug_native_keys):
        trials = set()
        for key in native_keys:
            trials.update(trial_key_mm.rev_map().get(key, []))
        for trial in trials:
            study = trials2studies[trial]
            entry = rectuple(
                disease=tuple(trials_diseases_mm.fwd_map().get(trial, [])),
                drugs=tuple(trial_name_mm.fwd_map().get(trial, [])),
                **study
            )
            out.append((wsa.id, entry))

    return MultiMap(out)

def trial_details(trial_ids):
    from browse.default_settings import aact
    from dtk.files import get_file_records
    s3f = aact.get_s3_file(latest=True, role='studies')
    all_data = {}
    header = None
    for rec in get_file_records(
            s3f.path(),
            select=(trial_ids,0),
            keep_header=True,
            ):
        if header is None:
            header = list(rec)
            if 'completion_date' not in header:
                header.append('completion_date')
                # data compatability provided by namedtuple default
            continue
        all_data[rec[0]] = dict(zip(header, rec))
    return all_data

def lookup_trial_diseases(trial_ids):
    from browse.default_settings import aact
    from dtk.files import get_file_records
    s3f = aact.get_s3_file(latest=True, role='diseases')
    all_data = []
    for rec in get_file_records(s3f.path(), select=(trial_ids,0)):
        all_data.append(rec)
    return MultiMap(all_data)

def lookup_trial_sponsors(trial_ids):
    '''Return a dict like {trial_id:sponsor,...}.'''
    from browse.default_settings import aact
    from dtk.files import get_file_records
    s3f = aact.get_s3_file(latest=True, role='sponsors')
    return dict(get_file_records(
            s3f.path(),
            select=(trial_ids,0),
            keep_header=None,
            ))

def lookup_trials_by_native_keys(native_keys):
    from browse.default_settings import aact
    s3f = aact.get_s3_file(latest=True, role='drugs')
    from dtk.files import get_file_records
    trial_drug_data = []
    trial_name_data = []
    for rec in get_file_records(s3f.path(), select=(native_keys,1)):
        trial_drug_data.append((rec[0], rec[1]))
        trial_name_data.append((rec[0], rec[2]))
    return MultiMap(trial_drug_data), MultiMap(trial_name_data)
    

# lookup_trials_by_molnames was originally written as a service routine for
# lookup_trials_by_molecules, but was later abandoned in a refactor, and
# isn't used anywhere. I deleted it rather than try to implement backward
# compatibility for varying file formats.
