#!/usr/bin/env python

# this program takes smiles codes for any number of drugs and
# calculates pariwise similarity scores between those drugs
# and known treatments (provided in an additional file)
# You can also choose which program you would like to use
# to calculate the similarity scores

import sys

from dtk.files import FileDestination

import dtk.rdkit_2019

import argparse
from builtins import range

class molecSim(object):
    def __init__(self, **kwargs):
        self.smiles = kwargs.get('smiles', None)
        self.rsfrs = kwargs.get('refset_frs', None)
        self.lib = kwargs.get('lib', None)
        # 2 is equivalent to ECFP 4
        self.morganFPSize = kwargs.get('morganFingerprintSize', 2)
        # setting for reporting the molecule bits,
        # these seem to be working, but were somewhat arbitrary
        self.minBitPor = kwargs.get('minBitPortion', 0.002)
        # minimum number of provided molecules in which a bit must be
        # found to be reported, regardles of portion
        # if a bit is found in 2 molecules, that is no use whatsoever
        self.minBitNum_floor = kwargs.get('minBitNum_floor', 2)
        self.cores = kwargs.get('cores', 1)
        self.map = kwargs.get('map', None)
        self.threshold = kwargs.get('threshold', None)
        self.bitsOutputCsv = kwargs.get('bitsOutputCsv', None)
        self.output = kwargs.get('output', None)
        self.lr_score_tsv = kwargs.get('lr_score_tsv', None)
        self.reportBits = bool(self.lr_score_tsv or self.bitsOutputCsv)
        self.useBitSmFP = kwargs.get('use_bit_smiles_fp', False)
        assert self.lib in ['rdkit', 'indigo']
        if self.reportBits:
            assert self.lib == 'rdkit'
    def run(self):
        if self.reportBits or self.useBitSmFP:
            self._prepForBits()
        self.fp = {}
        if self.lib == 'rdkit':
            self._run_rdkit()
        elif self.lib == 'indigo':
            self._run_indigo()
        self._prep_refset()
        if self.output:
            self.report()
        if self.reportBits:
            self._filter_bits()
            if self.bitsOutputCsv:
                self.report_bits(self.bitsOutputCsv)
            if self.lr_score_tsv:
                self.lr_score_bits()
    def _prep_refset(self):
        self.rs = []
        self.row = ['drug_id']
        for frs in self.rsfrs:
            if self.map:
                kt,title = frs
            else:
                kt = frs[0]
                title = str(frs[0])
            if kt in self.fp:
                self.rs.append(kt)
                if self.threshold:
                    self.row.append("near_"+title)
                else:
                    self.row.append("like_"+title)
    def _prepForBits(self):
        self.bitFp = {}
        self.all_bits = []
    def _run_indigo(self):
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/indigo")
        import indigo as igo
        self.indigo = igo.Indigo()
        # for every drug provided in the smiles file,
        # calculate the fingerwprint using Indigo
        for x in range(len(self.smiles)):
            try:
                m1 = self.indigo.loadMolecule(self.smiles[x][0])
                m1.aromatize()
                self.fp[self.smiles[x][1]] = m1.fingerprint("sim")
            except igo.IndigoException:
                pass
    def _run_rdkit(self):
        import time
        ts = time.time()
        # for every drug provided in the smiles file,
        # calculate the fingerprint using RDKit
        n = len(self.smiles)
        params = zip([self.smiles] * n,
                     range(n),
                     [self.morganFPSize] * n,
                     [self.reportBits] * n,
                     [self.useBitSmFP] * n,
                    )
        if self.cores > 1:
            import multiprocessing
            pool = multiprocessing.Pool(self.cores)
            all_res = pool.map(get_rdkit_fingerprint, params)
        else:
            all_res = map(get_rdkit_fingerprint, params)
        gen = (tup for tup in all_res if tup[0])
        for tup in gen:
            if tup[1]:
                self.fp[tup[0]] = tup[1]
            if tup[2]:
                self.all_bits += tup[3]
            if tup[3]:
                self.bitFp[tup[0]] = tup[3]
        console("for x in range(0,len(t)): took:",time.time() - ts)
    def report(self):
        outfile = FileDestination(self.output,delim=',',header=self.row)
        for row in self._compare():
            outfile.append(row)
    def get_results_dict(self):
        to_ret = {}
        for l in self._compare():
            to_ret[l[0]] = [float(x) for x in l[1:]]
        return to_ret
    def _compare(self, method = None):
        if method is None:
            method = self.lib
        if method == 'rdkit':
	        from rdkit import DataStructs
        # now we go through and calculate the similarity score for all drugs versus all reference set drugs
        for drug_id, fp1 in self.fp.items():
            row = [drug_id]
            # for this drug, compare it to every known treatment
            for kt in self.rs:
                # get the fingerprint of the KT
                fp2 = self.fp[kt]
                if method == "indigo":
                    sim1 = self.indigo.similarity(fp1, fp2, "tanimoto")
                elif method == 'rdkit':
                    sim1 = DataStructs.DiceSimilarity(fp1,fp2)
                # this is if we're using a binary approach where you require a minimum similarity score
                if self.threshold:
                    if sim1 > self.threshold:
                        row.append("1")
                    else:
                        row.append("0")
                else:
                    row.append(str(sim1))
            yield row
    def _filter_bits(self):
        import math
        from collections import Counter
        # find minimum number of molecules that a bit must be in
        # (FAERS LR uses the VarianceThreshold class from scikit_learn to
        # accomplish this same thing, but this code already existed; the
        # only advantage of the VarianceThreshold class is that it would
        # also remove features which were true in almost every case, but
        # that doesn't happen with molbits.)
        totalDrugs = len(self.bitFp)
        minBitNum = max(math.floor(totalDrugs * self.minBitPor),
                        self.minBitNum_floor
                       )
        # filter the bits for that minimum molecule number
        allBits = Counter(self.all_bits)
        self.bitsToReport = set(k for k, v
                                in allBits.items()
                                if (v > minBitNum)
                               )
        console('reduced %d to %d with minBitNum %d (%f %%)'%(
                len(allBits),
                len(self.bitsToReport),
                minBitNum,
                self.minBitPor*100,
                ))
    def report_bits(self, f):
        # print out results
        bitFile = FileDestination(f,delim=',')
        # first a header with smiles codes for the bits
        # unfortunately weka view cc(c)cc the same as c(c(c)c)c
        # (i.e. the () are ignored, as are capitalization.
        # I'll get around that by adding a number
        bitFile.append(["drug_id"]+[
                bitSmile + "_" + str(i)
                for i,bitSmile in enumerate(self.bitsToReport)
                ]) # get the smiles code for all bits to report
        import time
        ts = time.time()
        for drug_id in self.bitFp.keys():
            # for this drug, list all bits it possess or not
            bitFile.append([drug_id]+[
                    "True" if bitSmile in self.bitFp[drug_id].keys()
                    else "False"
                    for bitSmile in self.bitsToReport
                    ])
        console("for drug_id in bitFp.keys(): took:",time.time() - ts)
    def lr_score_bits(self):
        # Now build a sparse matrix
        from dtk.features import SparseMatrixBuilder
        smb = SparseMatrixBuilder()
        for drug_id, x in self.bitFp.items():
            gen = (cn for cn in x if cn in self.bitsToReport)
            for col_name in gen:
                smb.add(drug_id,col_name,True)
        fm = smb.get_matrix(bool)
        features = smb.col_names()
        row_keys = smb.row_keys()
        # create class labels for training
        labels = [
                key in self.rs
                for key in row_keys
                ]
        console(sum(labels),'of',len(labels),'are positive cases')
        # train LR classifier
        # XXX offer same LR options as FAERS?
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(fm,labels)
        console('accuracy',"{0:.4f}".format(lr.score(fm,labels)))
        # XXX calculate and print weighted accuracy?
        # apply model and output scores
        result = FileDestination(self.lr_score_tsv,header=['wsa','ktsimlr'])
        for key,score in zip(row_keys,[x[1] for x in lr.predict_proba(fm)]):
            result.append([key,'%.3e'%score])

def console(*args):
    sys.stderr.write(' '.join(str(x) for x in args)+'\n')

def get_rdkit_bitsmiles_fingerprint(id, bitSmFp, morganFingerprintSize):

    bitsmFps = []
    for bitsm, cnt in bitSmFp.items():
        fp = get_bit_smiles_bit_fp(bitsm, morganFingerprintSize)
        fp *= cnt
        bitsmFps.append(fp)

    if len(bitsmFps) != 0:
        import functools
        # Combined fingerprint from all the bit fingerprints
        fp = functools.reduce(lambda x, y: x + y, bitsmFps)
    else:
        from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
        # This is the default size empty vect.
        fp = UIntSparseIntVect(2**32 - 1)
    from collections import Counter
    allBits = []
    for k, v in bitSmFp.items():
        allBits.extend([k] * v)
    return id, fp, allBits, bitSmFp


def get_rdkit_fingerprint(params):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    t, x, morganFingerprintSize, reportBits, useBitSmiles = params
    bitFp={}
    allBits=[]
    smiles, id = t[x]

    if not isinstance(smiles, str):
        return get_rdkit_bitsmiles_fingerprint(id, smiles, morganFingerprintSize)

    m1 = Chem.MolFromSmiles(smiles, sanitize=False)
    if m1 is None:
        print("Failed to smiles on ", t[x])
        return None, None, None, None
        # it didn't work. What to do?

    process_unsanitized_mol(m1)

    myBitInfo={} # when the fingerprint is calculated, you can save the bitinfo here
    fp = AllChem.GetMorganFingerprint(m1,
                                      morganFingerprintSize,
                                      bitInfo = myBitInfo
                                     )
    # Do we need to turn the bits back into smiles codes (i.e. sub parts of the molecule)?
    # Or do bits mean the same thing everytime? = YES they are universal
    # Even if they do I think we might want them back to smiles code for interpretability
    # However for the time being I'm leaving them as bits for ease
    #
    # I don't know if it's a setting or something I'm missing,
    # but the bits include single molecules, or single bonds
    # (currently we're looking for 2)
    # I couldn't find a way to change that, so I'm simply filtering
    # For each bit, I check the radius, and only keep it if it is larger than 1
    # because the MorganFingerprint size above is 2,
    # that means only radius 2 will be saved
    #
    # We also want to remove those bits from the fp object, as that is what
    # we are actually using for comparsions.
    for bit, bitinfos in myBitInfo.items():
        for i, (atom_id, radius) in enumerate(bitinfos):
            if radius < morganFingerprintSize:
                fp[bit] = 0

    if reportBits or useBitSmiles:
        for bit, bitinfos in myBitInfo.items():
            for i, (atom_id, radius) in enumerate(bitinfos):
                if radius >= morganFingerprintSize:
                    # Now find what the smiles codes for this bit is,
                    # that way we can more easily interpret what the bit is
                    # make an atom map surrounding the atom of interest
                    # that is, the molecule, the radius, and the centered atom
                    env = Chem.FindAtomEnvironmentOfRadiusN(m1,
                                                            radius,
                                                            atom_id
                                                           )
                    amap={}
                    # pull out the submolecule
                    submol=Chem.PathToSubmol(m1, env, atomMap=amap)
                    # And then explain the bit by generating SMILES for that submolecule
                    # This is more useful when the SMILES is rooted at the central atom
                    bitSmiles = Chem.MolFragmentToSmiles(m1,
                                                         amap.keys(),
                                                         env,
                                                         rootedAtAtom=atom_id,
                                                       )
                    if bitSmiles not in bitFp:
                        bitFp[bitSmiles] = 0
                    bitFp[bitSmiles] += 1

                    allBits.append(bitSmiles)


        if useBitSmiles:
            return get_rdkit_bitsmiles_fingerprint(id, bitFp, morganFingerprintSize)
        return t[x][1], fp, allBits, bitFp
    else:
        return t[x][1], fp, None, None

def process_unsanitized_mol(mol):
    # We're not running sanitize because these fragments aren't real molecules.
    # However, because of that, certain things don't run that would prevent
    # us from being able to generate the fingerprint.  So we run them explicitly
    # here.
    mol.UpdatePropertyCache()
    from rdkit.Chem.rdmolops import FastFindRings
    FastFindRings(mol)

def get_bit_smiles_bit_fp(bitSmiles, morganFingerprintSize):
    """Generates a fingerprint for a bit smiles.

    A bit smiles is the SMILES code of a fragment (usually from a radius 2
    morgan fingerprint).

    The output fingerprint will have a single bit set.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    bitFp={}
    m1 = Chem.MolFromSmiles(bitSmiles, sanitize=False)
    try:
        process_unsanitized_mol(m1)
    except ValueError:
        from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
        return UIntSparseIntVect(2**32 - 1)

    # Molecule is setup, generate the fingerprint.
    myBitInfo={}
    fp = AllChem.GetMorganFingerprint(m1,
                                      morganFingerprintSize,
                                      bitInfo = myBitInfo,
                                      fromAtoms=[0],
                                     )
    for bit, bitinfos in myBitInfo.items():
        fp[bit] = 0
        for i, (atom_id, radius) in enumerate(bitinfos):
            if radius >= morganFingerprintSize:
                fp[bit] += 1

    # There should have been precisely 1 bit of radius N rooted at atom 0.
    nonzero = fp.GetNonzeroElements()
    assert sum(nonzero.values()) == 1, "Bit Smiles should turn back into a single bit: %s from %s" % (nonzero, bitSmiles)

    return fp



if __name__ == "__main__":

    parser=argparse.ArgumentParser(
            description='calculate similarities to specified drugs',
            )
    parser.add_argument('-s','--smiles_file'
            ,type=str
            ,default='smiles.tsv'
            ,help='file of smiles codes (default %(default)s)'
            )
    parser.add_argument('-l','--library_to_use'
            ,type=str
            ,default='indigo'
            ,help='which program to use to calculate molecular similarity, indigo or rdkit (default %(default)s)'
            )
    parser.add_argument('-t','--threshold'
            ,type=float
            ,help='threshold (default: use similarity metric, not boolean)'
            )
    parser.add_argument('-f','--morgan_fingerprint_size'
            ,type=int
            ,default=2
            ,help='threshold (default: use similarity metric, not boolean)'
            )
    parser.add_argument('-m','--map'
            ,action='store_true'
            ,help='drugfile contains a second column with output column id'
            )
    parser.add_argument('--minBitPortion'
            ,type=float
            ,default=0.002
            ,help='fraction of molecules bits must appear in (default: %(default)s)'
            )
    parser.add_argument('-b','--bitsOutputCsv'
            ,type=str
            ,help='output filename for molecules bits (default none)'
            )
    parser.add_argument('-o','--output'
            ,type=str
            ,help='output filename for similarity csv file (default none)'
            )
    parser.add_argument('--lr_score_tsv'
            ,type=str
            ,help='output filename for LR similarity score (default none)'
            )
    parser.add_argument('refset_file'
            ,type=str
            ,help='file of reference set drug ids, one per line'
            )
    parser.add_argument('--cores'
            ,type=int
            ,default=1
            ,help='number of cores allocated'
            )
    args = parser.parse_args()

    from dtk.files import get_file_records
    smiles = list(get_file_records(args.smiles_file,
                                   keep_header = True,
                                   parse_type = 'tsv'
                                  )
                 )
    rs_frs = [frs for frs
              in get_file_records(args.refset_file,
                                  parse_type = 'tsv'
                                 )
             ]
    ms = molecSim(
                  smiles = smiles,
                  refset_frs = rs_frs,
                  lib = args.library_to_use,
                  morganFingerprintSize = args.morgan_fingerprint_size,
                  minBitPortion = args.minBitPortion,
                  lr_score_tsv = args.lr_score_tsv,
                  cores = args.cores,
                  map = args.map,
                  threshold = args.threshold,
                  output = args.output,
                  bitsOutputCsv = args.bitsOutputCsv
                  )
    ms.run()
