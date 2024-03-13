


def load_dmpk_assays(agent, attr, ver):
    if attr == 'adme':
        flavor = 'adme'
    else:
        flavor = 'full'
        

    chembl_ids = agent.external_ids('chembl', ver)
    if not chembl_ids:
        return [], f'No ChEMBL id found, so no {attr.upper()} version {ver} assay data displayed'

    from dtk.s3_cache import S3MiscBucket,S3File
    matches = agent.get_molecule_matches(ver)
    if ver is None:
        # This will get changed below; get_match_inputs_version doesn't like it with ver==None
        chembl_ver = None
    else:
        chembl_ver = matches.get_match_inputs_version('chembl')

    if chembl_ver is None or int(chembl_ver) < 7:
        # Force-upgrade old versions.  This avoids a lot of backwards-compatibility code
        # that isn't particularly valuable to be able to reproduce.
        # Similar to drug collection attributes, this is a case of most recent data is almost
        # always what you want.
        chembl_ver = 7
    s3_file = S3File('chembl',f'chembl.{flavor}.v{chembl_ver}.{attr}_assays.sqlsv')

    try:
        s3_file.fetch(unzip=True)
    except OSError:
        return [], f'No assays of type {attr} for version {ver}'

    from dtk.files import get_file_records
    assay_list=[]
    for rec in get_file_records(s3_file.path(),
                                select=(chembl_ids,0),
                                ):
        assay_list.append(DmpkAssay(*rec))
    return assay_list, None


def chembl_tox_type(descr):
    descr = descr.lower()
    tox_types = [
            ('liver',
                ("liver tox", "toxicity liver", "liver disease", "liver damage", "cirrhosis", "steatosis", "HepG2", "liver failure"),
                ("cytotox", "selectivity index")
                ),
            ('immune',
                ("neutrop",),
                [],
                ),
            ('cv',
                ("qt",),
                [],
                ),
            ('other',
                ("cell viability", "animal tox", "tumor, proven histopathologically",),
                [],
                ),
            ]
    for label,pos_list,neg_list in tox_types:
        if any(x in descr for x in neg_list):
            continue
        if any(x in descr for x in pos_list):
            return label
    return None

from collections import namedtuple
DmpkAssay = namedtuple('DMPK', 'chembl_id description relation value unit assay_type assay_chembl_id organism')

DmpkNorm = namedtuple('DMPKNorm', 'category std_value std_unit score_rating score')

from enum import Enum
class Organism(Enum):
    Human = "Homo sapiens"
    Rat = "Rattus norvegicus"
    Mouse = "Mus musculus"
    NA = "N/A"
    Other = "Other"

    @classmethod
    def from_str(cls, full_str):
        if full_str is None or full_str == 'None':
            return cls.NA
        for e in cls:
            if full_str == e.value:
                return e
        return cls.Other
        


def short_organism(full_organism):
    return Organism.from_str(full_organism).name

def default_scoring(value, low_value, high_value, low_score, high_score):
    def _score():
        if value is None or value == "None":
            return low_score
        if value <= low_value:
            return low_score
        if value >= high_value:
            return high_score
        lerp = (value - low_value) / (high_value - low_value)
        return low_score + (high_score - low_score) * lerp
    score = _score()
    return score, ''

def score_categorize(value, low, high):
    if value <= low:
        return 'Low'
    elif value <= high:
        return 'Med'
    else:
        return 'High'

def prot_binding(assay, full_text, desc, assay_type):
    if not assay_type in ['ppb', 'fu']:
        return False
    
    value = assay.value if assay_type == 'ppb' else (1.0 - assay.value) * 100
    score, score_category = default_scoring(value, 90, 99, 4, 1) 
    return DmpkNorm("Protein Binding", value, "%", score_category, score)

def log_d(assay, full_text, desc, assay_type):
    match = (assay_type == 'logd7.4' or 'logd7.4' in full_text) and 'delta' not in assay_type
    if not match:
        return False

    score, score_category = default_scoring(assay.value, -1, 2, 1, 4)
    return DmpkNorm('LogD7.4',assay. value, '', score_category, score)

def clearance(assay, full_text, desc, assay_type):
    org = Organism.from_str(assay.organism)
    if org not in [Organism.Human, Organism.Rat, Organism.Mouse]:
        return False
    matches = assay_type in ['cmax', 'cl/f', 'cl', 'clh'] and ('hepatocyte' in full_text or 'hepatic' in full_text)
    if not matches:
        return False
    
    hep_cell_ref = {
        Organism.Human: (3.5, 19),
        Organism.Rat: (5.1, 28),
        Organism.Mouse: (3.3, 17.8),
    }
    liver_flow_ref = {
        Organism.Human: 21,
        Organism.Rat: 55,
        Organism.Mouse: 90,
    }
    microsome_ref = {
        Organism.Human: (8.6, 47),
        Organism.Rat: (13.2, 71.9),
        Organism.Mouse: (8.8, 48.0),
    }
    


    # Assume uL/min is per mil cells, just poorly annotated.
    if assay.unit == 'uL/min' or assay.unit == 'uL.min-1.(10^6cells)-1':
        # Compare to in-vitro hepatocyte clearance values.
        low, high = hep_cell_ref.get(org, (None, None))
        score_category = score_categorize(assay.value, low, high)
        score, _ = default_scoring(assay.value, low, high, 4.0, 1.0)
        value = assay.value
        std_unit = 'uL/min/10^6cells'
    elif assay.unit == 'mL.min-1.kg-1':
        # Compare to liver blood flow in organism.
        liver_flow = liver_flow_ref.get(org, (None, None))
        extraction_ratio = assay.value / liver_flow
        low, high = 0.3, 0.7
        score_category = score_categorize(extraction_ratio, low, high)
        score, _ = default_scoring(assay.value, low, high, 4.0, 1.0)
        value = extraction_ratio * 100
        #std_unit = 'mL/min/kg'
        std_unit = '% Liver'
    else:
        return False
        
    return DmpkNorm('Hepatic Clearance', value, std_unit, score_category, score)

def microsome_stability(assay, full_text, desc, assay_type):
    match = 'microsome' in full_text and 'stability' in full_text
    if not match:
        return False

    if assay.value == 'None':
        return False
    import re
    if assay.unit == '%':
        m = re.search(r'(\d+) ?min', desc)
        if m:
            time_hrs = float(m[1]) / 60
        else:
            m = re.search(r'(\d+) ?hr', desc)
            if m:
                time_hrs = float(m[1])
            else:
                time_hrs = None
        
        if time_hrs:
            import numpy as np
            halflife_hrs = -time_hrs / np.log2(float(assay.value) / 100)
            score, score_category = default_scoring(halflife_hrs, 0.5, 1.5, 1, 4)
            return DmpkNorm('Microsome Stability', halflife_hrs, 'hr', score_category, score)

    elif assay.unit == 'hr':
        score, score_category = default_scoring(assay.value, 0.5, 1.5, 1, 4)
        return DmpkNorm('Microsome Stability', assay.value, assay.unit, score_category, score)
    
    return False


def blood_stability(assay, full_text, desc, assay_type):
    match = 'blood' in full_text and 'stability' in full_text
    if not match:
        return False

    # We don't have enough of this data to bother with normalizing.
    return DmpkNorm('Microsome Stability', assay.value, assay.unit, '', None)

def caco2papp(assay, full_text, desc, assay_type):
    match = assay_type in ['logpapp', 'logp app', 'papp', 'permeability', 'ratio'] and (
            'caco' in desc or 'mdck' in desc)
    if not match:
        return False

    if assay_type == 'ratio':
        score, score_category = default_scoring(assay.value, 0, 2.0, 3, 1)
        return DmpkNorm('Caco/MDCK Permeability', assay.value, 'Ratio', score_category, score)
    elif 'm/s' in assay.unit:
        std_unit = 'ucm/s'
        if assay.unit == 'ucm/s' or '-6 cm/s' in assay.unit:
            val = assay.value
        elif assay.unit == 'nm/s':
            val = assay.value / 1000
        elif '-7 cm/s' in assay.unit:
            val = assay.value / 10
        else:
            return False
        low, high = 2, 20
        score, _ = default_scoring(val, low, high, 1, 4)
        score_category = score_categorize(val, low, high)
        return DmpkNorm('Caco/MDCK Permeability', val, std_unit, score_category, score)
    else:
        return False

def bioavailability(assay, full_text, desc, assay_type):
    match = assay.assay_type == 'F' and assay.unit == '%'
    if not match:
        return False

    return DmpkNorm('Bioavailability', assay.value, assay.unit, '', None)


    
def interpret_dmpk_assay(assay):
    interpreters = [
        bioavailability,
        caco2papp,
        blood_stability,
        microsome_stability,
        clearance,
        log_d,
        prot_binding,
    ]

    full_text = ' '.join(str(x) for x in assay).lower()
    desc = str(assay.description).lower()
    assay_type = str(assay.assay_type).lower()

    # Using short_organism as a proxy, could be explicit if that changes.
    is_std_organism = not assay.organism or Organism.from_str(assay.organism) != Organism.Other
    
    if is_std_organism:
        for interpreter in interpreters:
            info = interpreter(assay, full_text, desc, assay_type)
            if info is not False:
                return info
    return DmpkNorm('', None, None, None, None)

def interpret_dmpk_assays(assays):
    interp = [interpret_dmpk_assay(x) for x in assays]
    from dtk.data import MultiMap
    data = [(x.category, (x, assay)) for x, assay in zip(interp, assays)]
    return MultiMap(data).fwd_map()
