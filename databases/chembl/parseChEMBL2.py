#!/usr/bin/env python3
import os, django, sys, re, argparse
from collections import defaultdict
try:
    from path_helper import PathHelper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from path_helper import PathHelper
from dtk.files import get_file_records

# created 5.Feb.2016 - Aaron C Daugherty - twoXAR

# Parse ChEMBL DB

# TO DO:
# pull out data from:
#   ACTIVITIES - Good source of Ki/IC50
#   ACTION_TYPE - distinct list of action types used in the drug_mechanism table, together with a higher-level parent action type
#   MOLECULE_SYNONYMS - synonyms for a compound (e.g., common names, trade names, research codes etc)
#   COMPOUND_RECORDS - similar to above; may also contain pubchem sid numbers
#   PRODUCT_PATENTS - Table from FDA Orange Book, showing patents associated with drug products.
#   DRUG_MECHANISM - mechanism of action information for FDA-approved drugs and WHO anti-malarials
#       MECHANISM_REFS - references for information in the drug_mechanism table
#   PRODUCTS - Table containing information about approved drug products (mainly from the FDA Orange Book), such as trade name, administration route, approval date. Ingredients in each product are linked to the molecule dictionary via the formulations table
#   DEFINED_DAILY_DOSE - WHO DDD (defined daily dose) information - might worth adding as drug information. Not vital, but a nice to have

# As we're learning the schema, you can hit a URL like
# https://www.ebi.ac.uk/chembl/api/data/molecule?chembl_id=CHEMBL1201586
# to see all a drug's data in XML format; we then just need to figure out
# where the useful bits are in the database.
#
# The full ChEMBL web API is described here: https://www.ebi.ac.uk/chembl/ws
# Although we should prefer the local database for most things, it provides
# potentially useful things like molecule images and structural searches.

# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def verboseOut(*objs):
    if verbose:
        print(*objs, file=sys.stderr)

def useage(*objs):
    print("USEAGE: ", *objs, file=sys.stderr)

def printOut(*objs):
    print(*objs, file=sys.stdout)

def strip_bad_char(string, replace_char=""):
    charsToRm = "!@#$%^&*()[]{};:,./<>?\\|-= +`"
    for c in charsToRm:
        if c in string:
            string =  string.replace(c, replace_char)
    return string

class DataLookup:
    trailing_garbage = set([x.decode() for x in (
            b'\xc3\x82',
            b'\xc2\xa0',
            )])
    def report(self,*objs):
        print(*objs)
    def __init__(self,chembl_ids):
        self.chembl2molregno=dict(
                ch.MoleculeDictionary.select(
                                ch.MoleculeDictionary.chembl_id,
                                ch.MoleculeDictionary.molregno,
                        ).where(
                                ch.MoleculeDictionary.chembl << chembl_ids
                        ).tuples()
                )
        self.report('loaded',len(self.chembl2molregno),'molregnos')
        self.structures=None
        self.props=None
        self.alerts=None
        self.names=None
    def add_smiles(self,chembl_id,out_data):
        if not self.structures:
            self.structures={
                    r[0]:r
                    for r in ch.CompoundStructures.select(
                            ch.CompoundStructures.molregno,
                            ch.CompoundStructures.canonical_smiles,
                            ch.CompoundStructures.standard_inchi,
                            ch.CompoundStructures.standard_inchi_key,
                            ).where(
                                    ch.CompoundStructures.molregno
                                    << list(self.chembl2molregno.values())
                            ).tuples()
                    }
            self.report('loaded',len(self.structures),'structures')
        molregno = self.chembl2molregno[chembl_id]
        try:
            r = self.structures[molregno]
        except KeyError:
            return
        smile = r[1]
        if smile:
            out_data['smiles_code'] = smile
        out_data['inchi'] = r[2]
        out_data['inchi_key'] = r[3]
    def add_properties(self,chembl_id,out_data):
        if not self.props:
            self.props={
                    r.molregno:r
                    for r in ch.CompoundProperties.select().where(
                                    ch.CompoundProperties.molregno
                                    << list(self.chembl2molregno.values())
                            ).namedtuples()
                    }
            self.report('loaded',len(self.props),'properties')
        molregno = self.chembl2molregno[chembl_id]
        try:
            r = self.props[molregno]
        except KeyError:
            return
        for attr,key in (
                ('full_mwt','full_mwt'),
                ('full_molformula','mol_formula'),
                ('alogp','ALogP__hydrophobicity'),
                ('psa','polar_surface_area'),
                ('num_ro5_violations','num_rule_of_5_violations'),
                ('acd_most_apka','most_acidic_pKa'),
                ('acd_most_bpka','most_basic_pKa'),
                ('qed_weighted','wtd_qed__drug_likliness'),
                ('num_lipinski_ro5_violations','num_lipinski_rule5_violations'),
                ('acd_logp','Kow__solubility'),
                ('aromatic_rings','aromatic_rings'),
                ('heavy_atoms','heavy_atoms'),
                ('hba','hydrogen_bond_acceptors'),
                ('hbd','hydrogen_bond_donors'),
                ('acd_logd','logd7_4'), # This is ChEMBL25 & earlier
                ('cx_logd','logd7_4'), # This is ChEMBL26+
                ):
            val = getattr(r, attr, None)
            if val is not None:
                out_data[key] = val
        out_data['rule_of_3_pass'] = str(r.ro3_pass == 'Y')
    def add_alerts(self,chembl_id,out_data):
        if not self.alerts:
            alert_map = {}
            for r in ch.StructuralAlerts.select():
                filtStr = strip_bad_char(str(r.alert_name), replace_char = "_")
                alert_map[r.alert_id] = "sa_" + filtStr.rstrip("_").lower()
            with open("chembl_structural_alerts_list.txt", 'w') as f:
                f.write("\n".join(alert_map.values()))
            self.alerts={}
            for r in ch.CompoundStructuralAlerts.select().where(
                    ch.CompoundStructuralAlerts.molregno
                    << list(self.chembl2molregno.values())
                    ).namedtuples():
                self.alerts.setdefault(r.molregno,set()).add(alert_map[r.alert])
            self.report('loaded',len(self.alerts),'alerts')
        molregno = self.chembl2molregno[chembl_id]
        try:
            s = self.alerts[molregno]
            out_data['num_structural_alerts'] = len(s)
        except KeyError:
            out_data['num_structural_alerts'] = 0
            return
        for a in s:
            out_data[a] = 'True'
    def add_names(self,md_rec,out_data):
        if not self.names:
            self.names={}
            for r in ch.CompoundRecords.select().where(
                    ch.CompoundRecords.molregno
                    << list(self.chembl2molregno.values())
                    ).namedtuples():
                if not r.compound_name:
                    continue
                self.names.setdefault(r.molregno,set()).add(r.compound_name)
            for r in ch.MoleculeSynonyms.select().where(
                    ch.MoleculeSynonyms.molregno
                    << list(self.chembl2molregno.values())
                    ).namedtuples():
                if not r.synonyms:
                    continue
                self.names.setdefault(r.molregno,set()).add(r.synonyms)
            self.report('loaded',len(self.names),'names')
        canonical = md_rec.pref_name or md_rec.chembl_id
        synonyms = set()
        for name in self.names.get(md_rec.molregno,[]):
            if name == canonical:
                continue
            if re.match('SID[0-9]{5,9}$',name):
                # we were originally saving this as 'pubchem_sid', but
                # chembl typically holds more than one per compound, so
                # the one selected was arbitrary; since it's not used,
                # just remove it for now
                #out_data['pubchem_sid'] = name
                continue
            if name == 'NA':
                continue
            # in at least one case (CHEMBL426381) a long chemical name
            # contained an embedded '\n'; guard against this
            name = name.replace('\n','')
            # some chembl synonyms end with garbage in the last character;
            # strip those here (often, the stripped version will duplicate
            # an existing synonym, but not always; e.g.
            # CHEMBL21333	synonym	Pranlukast hydrate
            if name[-1] in self.trailing_garbage:
                name = name[:-1]
            synonyms.add(name)
        if canonical == md_rec.chembl_id:
            # try to pick a synonym as a canonical name; skip those with
            # commas or left parens (likely long chemical names) or ones
            # which are all numeric
            possibilities = [
                    x for x in synonyms
                    if ',' not in x
                    and '(' not in x
                    and not re.match(r'^\d*$',x)
                    ]
            if len(possibilities) == 1:
                canonical = possibilities[0]
                synonyms.remove(canonical)
        out_data['canonical'] = canonical
        if synonyms:
            out_data['synonym'] = synonyms

def get_assays():
    '''Return a list of dicts, one per wanted_type.'''
    wanted_types = ['A','P','T']
    result = [{} for _ in wanted_types]
    # an assay type of 'A' indicates ADME data
    acts=ch.Activities.select(
                    ch.Activities.molregno,
                    ch.Assays.assay_type,
                    ch.Assays.description,
                    ch.Activities.standard_relation,
                    ch.Activities.standard_value,
                    ch.Activities.standard_units,
                    ch.Activities.standard_type,
                    ch.Assays.chembl,
                    ch.Assays.assay_organism,
            ).join(ch.Assays).where(
                    (ch.Assays.assay_type << wanted_types)
            ).tuples()
    for x in acts:
        rounded = f"{float(x[4]):.3f}" if x[4] is not None else 'None'
        data = (x[2],x[3],rounded, x[5], x[6], x[7], x[8])
        d = result[wanted_types.index(x[1])]
        if x[0] not in d:
            d[x[0]] = set()
        d[x[0]].add(data)
    return result

def output_assays(f,rec_id,assays):
    for a in assays:
        f.write("\t".join([rec_id]+[str(x) for x in a]) + "\n")
def output_drug(f,rec_id,out_data):
    def output_rec(attr,val):
        f.write("\t".join([rec_id, attr, str(val)]) + "\n")
    output_rec('canonical', out_data['canonical'])
    for k,v in sorted(out_data.items(),key=lambda x:x[0]):
        if k == 'canonical':
            continue
        if type(v) == set:
            for part in sorted(v):
                try:
                    output_rec(k, part)
                except UnicodeDecodeError:
                    print ('Unable to save:', rec_id, k, part)
        elif v is not None and v != 'None':
            output_rec(k, v)

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================

    arguments = argparse.ArgumentParser(description="This parses the ChEMBL DB and prints out the select files directly to  the drugsets directory")

    arguments.add_argument("id_file", help="which ChEMBL keys to extract, one per line")
    arguments.add_argument("chembl_version", help="source db version (like chembl_23)")
    arguments.add_argument("--boxedWarning", action="store_true", help="report boxed warning flag")
    arguments.add_argument("--compoundProperties", action="store_true", help="report compound properties")
    arguments.add_argument("--smiles", action="store_true", help="report SMILEs and InChI codes")
    arguments.add_argument("--structuralAlerts", action="store_true", help="report compound structural alerts")
    arguments.add_argument("--maxPhase", action="store_true",
                            help="report the max clinical trial (0 = preclinical, 4 = approved"
                           )
    arguments.add_argument("--names", action="store_true", help="report synonyms")
    arguments.add_argument("--unfiltered-tox", action="store_true", help="report all tox assays")
    arguments.add_argument("--all", action="store_true", help="report all possible attributes")
    arguments.add_argument("--requireStructure", action="store_true",
                            help="skip molecules without a CompoundStructure record (only effective if --smiles or --all is specified)"
                           )

    args = arguments.parse_args()

    import importlib
    ch = importlib.import_module(args.chembl_version+'_schema')

    if args.all:
        args.boxedWarning = True
        args.compoundProperties = True
        args.smiles = True
        args.structuralAlerts = True
        args.maxPhase = True
        args.names = True

    ##### INPUTS AND OUTPUTS AND SETTINGS #####
    outFile = "ds.full.tmp"
    admeFile = "ds.adme.tmp"
    admeAssayFile = "admeAssays.tmp"
    pcAssayFile = "pcAssays.tmp"
    toxAssayFile = "toxAssays.tmp"
    acceptable_ids = {
            frs[0]
            for frs in get_file_records(args.id_file, keep_header = None)
            }
    # decide which drugs we'll report
# XXX Commented out the modality filter once we started noting biotech or not
# XXX We probably want to delete the commented out code eventually
# restore filter to assess effect on biotech flagging errors
    md = ch.MoleculeDictionary.select().where(
            (ch.MoleculeDictionary.chembl << acceptable_ids)
            & (ch.MoleculeDictionary.molecule_type << [
                    'Antibody','Protein','Small molecule'
                    ])
            )
    # set up for bulk fetching from other tables
    dl = DataLookup(acceptable_ids)
    # get assays
    from dtk.assays import DmpkAssay,chembl_tox_type
    adme, physiochem, tox = get_assays()
    if not args.unfiltered_tox:
        # since tox descriptions are repeated a lot, first reduce to the
        # tox description for each assay_chembl_id, analyze each description
        # once, and use the matched ids to filter the main list
        key_idx = DmpkAssay._fields.index('assay_chembl_id')-1
        desc_idx = DmpkAssay._fields.index('description')-1
        tox_type_of_assay = {}
        for s in tox.values():
            for rec in s:
                key = rec[key_idx]
                if key in tox_type_of_assay:
                    continue
                tox_type_of_assay[key] = chembl_tox_type(rec[desc_idx])
        filtered_tox = {}
        for mol_key,s in tox.items():
            for rec in s:
                assay_key = rec[key_idx]
                if tox_type_of_assay[assay_key] is not None:
                    filtered_tox.setdefault(mol_key,set()).add(rec)
        tox = filtered_tox

    message('ADME filter keys', len(adme))
    # scan main table and write output records
    total_drug_cnt = 0
    with open(outFile, 'w') as f, \
            open(admeAssayFile, 'w') as aaf, \
            open(admeFile, 'w') as f_adme, \
            open(toxAssayFile, 'w') as taf, \
            open(pcAssayFile, 'w') as paf:
        # write header
        header="\t".join(['chembl_id','attribute','value'])+"\n"
        f.write(header)
        f_adme.write(header)

        aaf.write("\t".join(DmpkAssay._fields) + '\n')
        paf.write("\t".join(DmpkAssay._fields) + '\n')
        taf.write("\t".join(DmpkAssay._fields) + '\n')
        for r in md:
            out_data = {}
            rec_id = r.chembl_id
            max_phase = int(r.max_phase)
            if args.maxPhase:
                out_data['max_phase'] = str(max_phase)
            if args.smiles:
                dl.add_smiles(rec_id,out_data)
                # Even if you request structure, we want to keep all molecules with phase information
                if args.requireStructure and 'inchi' not in out_data and max_phase <= 0:
                    print(f"Skipping {rec_id} which has no structure or phase")
                    continue
            if args.compoundProperties:
                dl.add_properties(rec_id,out_data)
            if args.boxedWarning:
                out_data['blackbox_warning'] = str(
                        bool(int(r.black_box_warning))
                        )
            if args.structuralAlerts:
                dl.add_alerts(rec_id,out_data)
            dl.add_names(r,out_data)
            if not args.names:
                # we need add_names no matter what to get the canonical name,
                # but if names are not otherwise asked for, drop all synonyms
                out_data.pop('synonym',None)
            out_data['biotech'] = str(r.molecule_type != 'Small molecule')
            # XXX If we want to also extract sequence data, it's in the
            # XXX helm_notation field of the Biotherapeutics model. That
            # XXX would slide right in to the DataLookup class.
            # now write out the results
            total_drug_cnt += 1
            output_drug(f,rec_id,out_data)
            if r.molregno in adme or max_phase >= 1:
                output_drug(f_adme,rec_id,out_data)
                if r.molregno in adme:
                    output_assays(aaf,rec_id,adme[r.molregno])
            if r.molregno in physiochem:
                output_assays(paf,rec_id,physiochem[r.molregno])
            if r.molregno in tox:
                output_assays(taf,rec_id,tox[r.molregno])
    message('Total drugs', total_drug_cnt)
