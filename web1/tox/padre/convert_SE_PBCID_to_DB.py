from __future__ import print_function
sefile = 'sider_data_matrix_sideEffectFreq_placeboNormd.tsv'
convertfile= 'pubchemcid_to_drugbankid.tsv'
convert={}
with open(convertfile, 'r') as f:
    for l in f:
        fs = l.rstrip().split("\t")
        convert[fs[0]] = fs[1]


with open(sefile, 'r') as f:
    print(f.readline().rstrip())
    for l in f:
        fs = l.rstrip().split("\t")
        fs[0] = fs[0].lstrip('0')
        if fs[0] in list(convert.keys()):
            fs[0] = convert[fs[0]]
            print("\t".join(fs))
