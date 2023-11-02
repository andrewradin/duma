import sys
sys.path.append("indigo")
from indigo import *

indigo = Indigo()

f = open('smiles.tsv', 'rb')
t = [row.strip().split('\t') for row in f]

#print(len(t[0]))
#print(len(t[1]))
#print(len(t))

for x in xrange(0,len(t)):
    for y in xrange(x,len(t)):
        try:

#        m1 = indigo.loadMolecule("CC(C)C=CCCCCC(=O)NCc1ccc(c(c1)OC)O")
#         m2 = indigo.loadMolecule("COC1=C(C=CC(=C1)C=O)O")
            m1 = indigo.loadMolecule(t[x][0])
            m2 = indigo.loadMolecule(t[y][0])
        # Aromatize molecules because second molecule is not in aromatic form
            m1.aromatize()
            m2.aromatize()
 
        # Calculate similarity between "similarity" fingerprints
#        print("Similarity fingerprints:");
            fp1 = m1.fingerprint("sim");
            fp2 = m2.fingerprint("sim");
 
#        print("  Tanimoto: %s" % (indigo.similarity(fp1, fp2, "tanimoto")));
#        print("  Tversky: %s" % (indigo.similarity(fp1, fp2, "tversky")));
            sim1 = indigo.similarity(fp1, fp2, "tanimoto");
            sim2 = indigo.similarity(fp1, fp2, "tversky");
 
        # Calculate similarity between "substructure" fingerprints
#        print("Substructure fingerprints:");
            fp1 = m1.fingerprint("sub");
            fp2 = m2.fingerprint("sub");
         
#        print("  Tanimoto: %s" % (indigo.similarity(fp1, fp2, "tanimoto")));
#        print("  Tversky: %s" % (indigo.similarity(fp1, fp2, "tversky")));
            sub1 = indigo.similarity(fp1, fp2, "tanimoto");
            sub2 = indigo.similarity(fp1, fp2, "tversky");

            print("%s\t%s\t%s\t%s\t%s\t%s" % (t[x][1], t[y][1], sim1, sim2, sub1, sub2))
        except:
            print("%s\t%s\tNULL\tNULL\tNULL\tNULL" % (t[x][1], t[y][1]))
