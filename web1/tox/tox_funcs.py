#!/usr/bin/python
import sys, os, django
sys.path.insert(1,"../")
sys.path.insert(1,"../ML")
from run_eval_weka import convertTypes, addFilteredFeatures
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from collections import defaultdict


def get_pred_fts(drugAndIdType, featuresToUse, allBitNames, chemblFileName):
    chemblIds = convertToChemblIds(drugAndIdType)
    initialData = defaultdict(dict)
    # this is a hacky way to use existing code
    for id in chemblIds:
        initialData[id]['chembl_id'] = id
    pred_data = suplementWithChemblData(initialData, chemblFileName, atr_subset=featuresToUse)
    sys.stderr.write("Getting submolecule bits for predictions drugs...\n")
    # now we need to also get the submolecules/bits for the prediction drugs
    pred_drug_bits_strings = suplementWithChemblData(initialData, chemblFileName, atr_subset=['sub_smiles'])
    pred_drug_bits = make_select_bits_matrix(pred_drug_bits_strings, allBitNames)
    return addFilteredFeatures(pred_data, pred_drug_bits, allBitNames)

def make_select_bits_matrix(pred_drug_bits_strings, allBitNames):
    pred_drug_bits = defaultdict(list)
    for drug in pred_drug_bits_strings.keys():
        # not all drugs have sub-smiles (because not all drugs have smiles codes we can load)
        if 'sub_smiles' not in list(pred_drug_bits_strings[drug].keys()):
            pred_drug_bits[drug] = ['?']*len(allBitNames)
            continue
        allSmiles = {}
        for bit_count in pred_drug_bits_strings[drug]['sub_smiles'].split(","):
            l = bit_count.split("_")
            if len(l) > 1:
                allSmiles[l[0]] = int(l[1])
            else:
                allSmiles[l[0]] = 1
        bits_counts = []
        for bit in allBitNames:
            if bit in list(allSmiles.keys()):
                bits_counts.append(allSmiles[bit])
            else:
                bits_counts.append(0)
        pred_drug_bits[drug] = bits_counts
    return pred_drug_bits

# for this case, we already have chembl, IDs, but in the future we may need to convert from some other ID type
def convertToChemblIds(drugAndIdType):
    return list(drugAndIdType.keys())

def suplementWithChemblData(dd, chemblFileName, atr_subset = None):
    # pull out what chembl ids we have
    chemblIds = [dd[k]['chembl_id'] for k in dd.keys() if 'chembl_id' in list(dd[k].keys())]
    chemblAtrs = getChemblAtrs(chemblIds, chemblFileName, atr_subset = atr_subset)
    for id in dd.keys():
        if 'chembl_id' not in list(dd[id].keys()):
            continue
        # for those IDs which have a chembl ID to link to, load the chembl attributes for that id
        for atr in chemblAtrs[dd[id]['chembl_id']].keys():
            dd[id][atr] = chemblAtrs[dd[id]['chembl_id']][atr]
    return dd

def getChemblAtrs(chemblIds, chemblFileName, atr_subset = None):
    chemblFile = fetchDrugsetFile(chemblFileName)
    return readInAttrFile(chemblFile, id_subset=chemblIds, atr_subset = atr_subset)

# Read in the attribute data
def readInAttrFile(file, id_subset=None, atr_subset=None):
    from collections import defaultdict
    d = defaultdict(dict)
    with open(file, 'r') as f:
        header = f.readline()
        for line in f:
            fields = line.rstrip().split("\t")
            if id_subset and fields[0] not in id_subset:
                continue
            if atr_subset and fields[1] not in atr_subset:
                continue
            d[fields[0]][fields[1]] = convertTypes(fields[2])
    return d

def fetchDrugsetFile(fileName):
    from dtk.s3_cache import S3Bucket, S3File
    f=S3File(S3Bucket('drugsets'),fileName)
    f.fetch()
    return PathHelper.drugsets + fileName
