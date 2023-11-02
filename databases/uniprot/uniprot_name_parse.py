#!/usr/bin/env python

import xml.etree.cElementTree as ET
import sys
import gzip
import json

def run():
    ns = '{http://uniprot.org/uniprot}'
    it = ET.iterparse(sys.stdin, events=('end',))
    sys.stderr.write("Parsing! (Expect roughly 170K elements)\n")
    output = []
    for action, elem in it:
        if elem.tag != ns+'entry':
            continue


        prot = elem.find(ns+'protein')
        uniprots = [x.text for x in elem.findall(ns+'accession')]

        rec_name_el = prot.find(ns+'recommendedName')
        if not rec_name_el:
            rec_name_el = prot.find(ns+'submittedName')

        if not rec_name_el:
            sys.stderr.write("Couldn't find recommended or submitted name\n")
            sys.stderr.write(ET.tostring(elem))
            sys.stderr.write("\n")
            continue

        alt_name_els = prot.findall(ns+'alternativeName')
        full_name = rec_name_el.find(ns+'fullName').text

        alt_names = []

        alt_names.extend([x.text for x in rec_name_el.findall(ns+'shortName')])
        alt_names.extend(['EC '+x.text for x in rec_name_el.findall(ns+'ecNumber')])
        for alt_name_el in alt_name_els:
            alt_names.extend([x.text for x in alt_name_el.findall(ns+'fullName')])
            alt_names.extend([x.text for x in alt_name_el.findall(ns+'shortName')])
            alt_names.extend(['EC ' + x.text for x in alt_name_el.findall(ns+'ecNumber')])

        gene_el = elem.find(ns+'gene')
        if gene_el:
            gene_names = [el.text for el in gene_el.findall(ns+'name')]
        else:
            gene_names = []

        out = {
            'uniprots': uniprots,
            'full_name': full_name,
            'alt_names': alt_names,
            'gene_names': gene_names,
            }
        output.append(out)

        elem.clear()
        if len(output) % 1000 == 0:
            sys.stderr.write("Parsed %d elems\n" % len(output))

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    run()
