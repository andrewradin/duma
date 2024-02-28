#!/usr/bin/env python3

from builtins import str
import xml.etree.ElementTree as etree
import sys
import argparse

global args

def out(l):
    sys.stdout.write("\t".join([x.strip() for x in l])+'\n')

def warn(*args):
    sys.stderr.write(" ".join([str(x) for x in args])+'\n')

ns='{http://www.drugbank.ca}'

reference_blacklist = set([
        # These papers are problematic, but they are also
        # the only source of some DPIs.  We experimented with removing them
        # but ultimately decided against keeping it.
        #('pubmed-id','11752352'),
        #('pubmed-id','10592235'),
        #('pubmed-id','17016423'),
        #('pubmed-id','17139284'),
        ])

reference_trace = None

# set up dictionaries to map text to numbers for calculations
evidence = {}
evidence['na'] = 0.0
evidence['no'] = 0.4
evidence['unknown'] = 0.5
evidence['yes'] = 0.9

direction = {}
direction['unknown'] = 0
direction['inhibitor'] = -1
direction['antagonist'] = -1
direction['agonist'] = 1
direction['potentiator'] = 1
direction['binder'] = 0
direction['cofactor'] = 1
direction['inducer'] = 1
direction['activator'] = 1

country = {}
country['United States'] = 'US'
country['Canada'] = '' #CA, but don't add to file

attr_list=["approved",
        "illicit",
        "experimental",
        "investigational",
        "neutraceutical",
        "withdrawn",
        ]

def parse_command_line():
    ap = argparse.ArgumentParser(description='extract data from drugbank xml on stdin')
    ap.add_argument("--reference-trace", help="dump raw DPI references")
    ap.add_argument("--action-dump", action='store_true', help="dump raw DPI actions")
    ap.add_argument("-i", "--input", help="Input zipfile")
    ap.add_argument("mode", choices=['drugset','dpi'], help="what to extract")
    args = ap.parse_args()
    return args

def transcribe_aliases(drugID,drug,listName,elemName):
    listNode = drug.find(ns+listName)
    for e in listNode.findall(ns+elemName):
        out([drugID,elemName,e.text])

# parse each drug
def handleDrug(drug,mode):
    drugID = "unknown"
    # look for the first drug in this record with a primary key
    for drugbankid in drug.findall(ns+'drugbank-id'):
         if drugbankid.get('primary') == 'true':
             drugID = drugbankid.text
             break

    # ignore if this is not a primary key drug
    if drugID != "unknown":
        name = drug.find(ns+'name')
        drugName = name.text

        if mode == 'drugset':
            out([drugID,'canonical',drugName])
            # We want to capture whether a drug is a biotech type, since
            # these typically don't have smiles, and might accidentally
            # be pulled into a small-molecule cluster based on ambiguous
            # synonyms. There are various ways to do this:
            # - a 'type' attribute of 'biotech' in the 'drug' tag
            # - 'biotech-category' tags in the 'biotech-categories' group
            # - 'sequence' tags in the 'sequences' group
            # For now, trigger on the biotech type, and verify that the
            # other two are present. We may eventually want to capture the
            # category or sequence for better matching and filtering.
            dtype = drug.get('type')
            out([drugID,'biotech',str(dtype == 'biotech')])
            if True:
                # warn about inconsistencies
                has_sequences = (drug.find(ns+'sequences') is not None)
                has_biocats = (drug.find(ns+'biotech-categories') is not None)
                if dtype == 'biotech':
                    if not (has_sequences and has_biocats):
                        warn(drugID,'biotech',has_sequences,has_biocats)
                elif dtype != 'small molecule' or has_sequences or has_biocats:
                    warn(drugID,dtype,has_sequences,has_biocats)
            ## # special handling for 'acetate' endings
            ## suffix = ' acetate'
            ## if ')' not in drugName and drugName.lower().endswith(suffix):
            ##     out([drugID,'synonym',drugName[:-len(suffix)]])
            attrs = set()
            groups = drug.find(ns+'groups')
            groupList = groups.findall(ns+'group')
            for g in groupList:
                attrs.add(g.text)
            for attr in attr_list:
                out([drugID,attr,str(attr in attrs)])

            # mol weight and formula may be under either calculated or
            # experimental properties. Check both and prefer experimental.
            mol_weight=None
            mol_formula=None
            props = drug.find(ns+'calculated-properties')
            if props != None:
                propList = props.findall(ns+'property')
                for p in propList:
                    k = p.find(ns+'kind')
                    v = p.find(ns+'value')
                    if k.text == 'SMILES':
                        smiles = v.text
                        if smiles:
                            out([drugID, 'smiles_code', smiles])
                    if k.text == 'InChI':
                        out([drugID, 'inchi', v.text])
                    if k.text == 'InChIKey':
                        val = v.text
                        prefix='InChIKey='
                        if val.startswith(prefix):
                            val = val[len(prefix):]
                        out([drugID, 'inchi_key', val])
                    if k.text == 'Molecular Formula':
                        mol_formula = v.text
                    if k.text == 'Molecular Weight':
                        mol_weight = v.text

            props = drug.find(ns+'experimental-properties')
            if props != None:
                propList = props.findall(ns+'property')
                for p in propList:
                    k = p.find(ns+'kind')
                    v = p.find(ns+'value')
                    if k.text == 'Molecular Formula':
                        mol_formula = v.text
                    if k.text == 'Molecular Weight':
                        mol_weight = v.text
            if mol_weight:
                out([drugID, 'full_mwt', mol_weight])
            if mol_formula:
                out([drugID, 'mol_formula', mol_formula])

            cas = drug.find(ns+'cas-number')
            if cas.text:
                out([drugID,'cas',cas.text])
            atc = drug.find(ns+'atc-codes')
            if atc is not None:
                atc_list = atc.findall(ns+'atc-code')
                for node in atc_list:
                    out([drugID,'atc',node.attrib['code']])
            # pharma code numbers should be stored as synonyms
            ex_codes = drug.find(ns+'external-codes')
            if ex_codes is not None:
                ex_code_list = ex_codes.findall(ns+'external-code')
                for node in ex_code_list:
                    out([drugID,'synonym',node.text])
            ext_ids = drug.find(ns+'external-identifiers')
            kegg=None
            pubchem=None
            chembl=set()
            bindingdb=set()
            if ext_ids != None:
                idList = ext_ids.findall(ns+'external-identifier')
                for item in idList:
                    k = item.find(ns+'resource')
                    v = item.find(ns+'identifier')
                    if k.text.startswith('KEGG '):
                        kegg=v.text
                    if k.text == 'PubChem Compound':
                        pubchem=v.text
                    if k.text == 'ChEMBL':
                        chembl.add(v.text)
                    if k.text == 'BindingDB':
                        bindingdb.add('BDBM'+v.text)
            if kegg:
                out([drugID,"kegg",kegg])
            if pubchem:
                out([drugID,"pubchem_cid",pubchem])
            if chembl:
                for v in chembl:
                    out([drugID,"linked_chembl_id",v])
            if bindingdb:
                for v in bindingdb:
                    out([drugID,"linked_bindingdb_id",v])
            # the 2015-11-17 dataset no longer has brands.  Instead,
            # it has "international-brands" and "products".  It's much
            # more cluttered than before, with entries like:
            # Kogenate
            # Kogenate FS     -(with Bio-set)
            # Kogenate Pws 500iu/vial
            # Kogenate - Pws IV 250I.U./vial
            # Kogenate FS
            # Kogenate Pws 1000iu/vial
            # Kogenate FS     -(with Vial Adapter)
            # Kogenate Pws 250iu/vial
            # Kogenate - Pws IV 1000I.U./vial
            # Kogenate - Pws IV 500I.U./vial
            # There are about 65,000 of these, with most of the same names
            # also appearing in mixtures.
            #
            # To filter this, collect all brand and mixture names in a
            # big list, and sort it.  Then remove any elements from the
            # list that begin with the preceeding element, followed by
            # a space. Also remove exact duplicates, and only include
            # mixtures that aren't also brands.
            brands_and_mixtures = []
            international_brands = drug.find(ns+'international-brands')
            if international_brands:
                for node in international_brands.findall(ns+'international-brand'):
                    n = node.find(ns+'name')
                    brands_and_mixtures.append( (n.text,'brand') )
            mixtures = drug.find(ns+'mixtures')
            if mixtures:
                for node in mixtures.findall(ns+'mixture'):
                    n = node.find(ns+'name')
                    brands_and_mixtures.append( (n.text,'mixture') )
            brands_and_mixtures.sort()
            keep = []
            for elem in brands_and_mixtures:
                if keep:
                    if elem[0] == keep[-1][0]:
                        continue
                    if elem[0].startswith(keep[-1][0]+' '):
                        continue
                if len(elem[0]) > 256:
                    warn(drugID,'dropping long name',elem[0])
                    continue
                keep.append(elem)
            for elem in keep:
                # a very small number of elements have embedded ^M characters
                # in place of spaces; fix that here
                txt = ' '.join(elem[0].split())
                out([drugID,elem[1],txt])
            transcribe_aliases(drugID,drug,'synonyms','synonym')

        elif mode == 'dpi':
            for targets in drug.findall(ns+'targets'):
                handleTargets(drugID, drugName, targets)


def extract_refs(target):
    refs = target.find(ns+'references')
    for ref_type,key in (
            ('article','pubmed-id'),
            ('textbook','isbn'),
            ('link','url'),
            ('attachment','url'),
            ):
        group = refs.find(ns+ref_type+'s')
        for ref in group.findall(ns+ref_type):
            if key:
                value = ref.find(ns+key).text
            else:
                value = None
            yield (ref_type,key,value)

def target_ok(target):
    has_blacklisted_ref = False
    for ref_type,key,value in extract_refs(target):
        if (key,value) in reference_blacklist:
            has_blacklisted_ref = True
            continue # ignore and keep looking
        return True # found a valid ref

    # If the only source is a blacklisted ref, discard this
    # target.  If it has no references at all, keep it.
    return not has_blacklisted_ref

# parse each drug target and write csv line
def handleTargets(drugID, drugName, targets):
    dropped = []
    for target in targets.findall(ns+'target'):
        org = target.find(ns+'organism')
        poly = target.findall(ns+'polypeptide')
        if org.text in ('Human','Humans') and poly:
            name = target.find(ns+'name')
            proteinName = name.text

            proteinID = poly[0].get('id')
            if reference_trace:
                for ref_type,key,value in extract_refs(target):
                    reference_trace.append(
                            (drugID,proteinID,ref_type,key,value)
                            )
            if not target_ok(target):
                dropped.append(f'"{proteinName}" ({proteinID})')
                continue


            knownaction = target.find(ns+'known-action')
            ka = knownaction.text
            if ka not in evidence:
                ka = 'na'

            actions = target.find(ns+'actions')
            actionList = actions.findall(ns+'action')
            if actionList:
                act = actionList[0].text
                if act not in direction:
                    act = 'unknown'
            else:
                act = 'unknown'

            if args.action_dump:
                out([drugID,proteinID,ka,act])
            else:
                out([drugID,proteinID, ("%f" % evidence[ka]), str(direction[act])])
    if dropped:
        # This will break things downstream, but can be useful to uncomment for digging into what was removed.
        #print(f'{len(dropped)} targets dropped on "{drugName}" [{",".join(dropped)}] https://drugbank.ca/drugs/{drugID}')
        pass

################################################################################
#
# program starts here
#

args = parse_command_line()

if args.reference_trace:
    from dtk.files import FileDestination
    reference_trace = FileDestination(args.reference_trace)

if args.mode == 'drugset':
    out('drugbank_id attribute value'.split())
elif args.mode == 'dpi':
    out('drugbank_id uniprot_id evidence direction'.split())

from tqdm import tqdm
import zipfile
with zipfile.ZipFile(args.input, 'r') as archive:
    for key in tqdm(sorted(archive.namelist())):
        with archive.open(key) as drugfile:
            for event,elem in etree.iterparse(drugfile, events=('start','end')):
                if event == 'end' and elem.tag == ns+'drug':
                    handleDrug(elem,args.mode)
                    elem.clear()

if args.reference_trace:
    reference_trace.close()

