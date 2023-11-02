#!/usr/bin/env python

# this program takes smiles codes for any number of drugs and 
# reports the molecular bits for each drug

import sys, os, django, argparse
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from algorithms.exit_codes import ExitCoder
# Load RDKit
sys.path.append("/usr/lib/python2.7/dist-packages/")
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

if __name__=='__main__':
    #=================================================
    # Read in the arguments/define options
    #=================================================
    exitCoder = ExitCoder()

    arguments = argparse.ArgumentParser(description="Extract all molecular substructures from smiles codes in provided file")

    arguments.add_argument("-i", '--smiles_file', help="file of smiles codes")
    
    arguments.add_argument("-f", '--morgan_fingerprint_size', type=int, default=2, help='Morgans finger print radius. Default: %(default)s')
    
    arguments.add_argument("-b", '--bitsOutput', default='moleculeBits.tsv', help='output file for molecules bits. Default: %(default)s')
    
    arguments.add_argument('--smallerBits', action='store_true', help='Include molecular bits smaller than the Morgans finger print radius. Warning: this will result in a lot more bits.')
    
    arguments.add_argument("-c", '--reportCounts', action='store_true', help='The number of times the sub structure was seen should be reported')
    
    args = arguments.parse_args()

    # return usage information if no argvs given

    if not args.smiles_file:
        arguments.print_help()
        sys.exit(exitCoder.encode('usageError'))
    
    
    #========================================================================================
    # currently hardcoded parameters
    #========================================================================================
    smilesCol = 2
    idCol = 0
    attributeName = 'sub_smiles'
    # for rdkit
    morganFingerprintSize = args.morgan_fingerprint_size # 2 is equivalent to ECFP 4
    
    f = open(args.smiles_file, 'rb')
    t = [row.strip().split('\t') for row in f]
    
    # I want to start tracking the smiles we couldn't load
    # though at this point we don't do anything with them
    badSmiles = []
    outBits = []
    
    # for every drug provided in the smiles file, calculate the fingerprint using RDKit
    for row in t:
        # this is an attribute file with all sorts of things in it
        if row[1] != 'smiles_code':
            continue
        m1 = Chem.MolFromSmiles(row[smilesCol])
        if m1 is None:
            # it didn't work. What to do?
            badSmiles.append(row[idCol] + "\t" + row[smilesCol])
            pass
        
        myBitInfo={} # when the fingerprint is calculated, you can save the bitinfo here
        try:
            fp = AllChem.GetMorganFingerprint(m1, morganFingerprintSize, bitInfo = myBitInfo) # right now this is hardcoded above
        except : # probably because the smiles aren't coded just right
            badSmiles.append(row[idCol] + "\t" + row[smilesCol])
            pass
        toReport = []
        for bit in myBitInfo.keys():
            # find all of the bits of the full radius length.
            # note, this is optional, and if we wanted to include all bits of the full radius length or SHORTER, we wound't need to do this
            if args.smallerBits:
                ofLength = filter(None, [sublist if sublist[1] > 0 else None for sublist in myBitInfo[bit] ])
            else:
                ofLength = filter(None, [sublist if sublist[1] > (morganFingerprintSize - 1) else None for sublist in myBitInfo[bit] ])
            bitCount = len(ofLength)
            if bitCount > 0:
                # Now find what the smiles codes for this bit is,
                # that way we can more easily interpret what the bit is.
                # just for ease we'll use the first instance, but it wouldn't matter
                env = Chem.FindAtomEnvironmentOfRadiusN(m1, ofLength[0][1], ofLength[0][0]) # that is, the molecule, the radius, and the centered atom
                # make an atom map surrounding the atom of interest
                amap={}
                # pull out the submolecule
                submol=Chem.PathToSubmol(m1, env, atomMap=amap)
                # And then explain the bit by generating SMILES for that submolecule
                # the nice thing here is that it reports canonical smiles, which means the string will be identical for all
                bitSmiles = str(Chem.MolToSmiles(submol))
                if bitSmiles.rstrip() == "": # there is at least one drugbank drug that doesn't really have a stucture (it seems to be some ions), and that seems to result in a blank sub molecule
                    continue
                if args.reportCounts and bitCount > 1:
                    bitSmiles = bitSmiles + "_" + str(bitCount)
                toReport.append(bitSmiles)
        if len(toReport) > 0:
            outBits.append("\t".join([row[idCol], attributeName, ",".join(toReport)]))
        else:
            badSmiles.append(row[idCol] + "\t" + row[smilesCol])
    
    with open(args.bitsOutput, 'w') as f:
        f.write("\n".join(outBits) + "\n")
    if len(badSmiles) > 0:
        with open(args.smiles_file + "_unableToLoad.tsv", 'w') as f:
            f.write("\n".join(badSmiles) + "\n")
