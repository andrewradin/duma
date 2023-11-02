#!/usr/bin/env python
import os, sys, re
import collections
sys.path.insert(1,"../../web1")
from path_helper import PathHelper

# created 2.Jun.2017 - Aaron C Daugherty - twoXAR

# Parse Human Protein Atlas

class processed_elem:
    def __init__(self, elem):
        self.elem = elem
    def report(self,
              te_file = 'humanProteinAtlas_tisExp.tsv',
              abExp_file = 'humanProteinAtlas_AbExp.tsv',
              rnaExp_file = 'humanProteinAtlas_RNAExp.tsv'):
### For the time being we're just printing the results,
### but I think we'll want to put these into different files at some point
        for u in self.uniprots:
            print "\t".join([self.id,'Uniprot',u])
        for k in sorted(self.tis_exp):
            print "\t".join([self.id,k,self.tis_exp[k],'tisExp'])
        for k in sorted(self.rna_exp):
            print "\t".join([self.id,k,self.rna_exp[k],'rnaExp'])
        for k in sorted(self.rna_cl_exp):
            print "\t".join([self.id,k,self.rna_cl_exp[k],'rnaCl'])
        for k in sorted(self.ab_exp):
            print "\t".join([self.id,k,self.ab_exp[k],'abExp'])
    def process(self):
        self._get_uniprots()
        self._get_tis_exp()
        self._get_rnaseq_exp()
        self._get_antibody_te()
    def _get_uniprots(self):
        uniprots = set()
        e = self.elem.find('identifier')
        self.id=e.attrib['id']
        for e2 in e.findall('xref'):
            if e2.attrib['db'] == 'Uniprot/SWISSPROT':
                uniprots.add(e2.attrib['id'])
        self.uniprots = uniprots
    def _get_antibody_te(self):
        te = {}
        for e in self.elem.findall('antibody'):
            for e2 in e.findall('tissueExpression'):
                for e3 in e2.findall('data'):
                    t = e3.find('tissue').text
                    if t:
                        for e4 in e3.findall('tissueCell'):
                            ct = e4.find('cellType').text
                            v = 0
                            d = 0
                            for e5 in e4.findall('level'):
                                if e5.attrib['type']=='staining':
                                    v += int(self._get_expr_val(e5.text))
                                    d += 1
                                elif e5.attrib['type']=='intensity':
                                    v += self._process_ab_intensity(e5.text)
                                    d += 1
                            te[";".join([t,ct])] = str(float(v/d)) if d else '0'
        self.ab_exp = te
    def _get_tis_exp(self):
        te = {}
        for e in self.elem.findall('tissueExpression'):
            for e2 in e.findall('data'):
                if e2.find('level').attrib['type'] == 'expression':
                    te[e2.find('tissue').text] = self._get_expr_val(e2.find('level').text)
                for e3 in e2.findall('tissueCell'):
                    if e3.find('level').attrib['type'] == 'expression':
                        k = ";".join([e2.find('tissue').text,
                                        e3.find('cellType').text])
                        te[k] = self._get_expr_val(e3.find('level').text)
        self.tis_exp = te
    def _get_rnaseq_exp(self):
        te = {}
        cle={}
        for e in self.elem.findall('rnaExpression'):
            if e.attrib['technology'] != 'RNAseq':
                continue
            for e2 in e.findall('data'):
                if e2.find('level').attrib['type'] == 'abundance':
                    v = e2.find('level').attrib['tpm']
                else:
                    continue
                for e3 in e2.findall('cellLine'):
                    cle[e3.text] = v
                for e3 in e2.findall('tissue'):
                    te[e3.text] = v
        self.rna_exp = te
        self.rna_cl_exp = cle
    def _get_expr_val(self, s):
        try:
            test = int(float(s))
            return str(test)
        except ValueError:
            pass
        if s == 'low':
            return '1'
        elif s == 'medium':
            return '2'
        elif s == 'high':
            return '3'
        elif s == 'not detected':
            return '0'
        return None
    def _process_ab_intensity(self, s):
        try:
            test = int(float(s))
            return test
        except ValueError:
            pass
        if s == 'Weak':
            return 1
        elif s == 'Moderate':
            return 2
        elif s == 'Strong':
            return 3
        elif s == 'Negative':
            return 0
        return None

if __name__=='__main__':
    import xml.etree.cElementTree as ET
    from dtk.files import get_file_lines
    import argparse
    arguments = argparse.ArgumentParser(description="Parse Human Protein Atlas XML data")
    arguments.add_argument("xml", help="proteinatlas.xml.gz")
    args = arguments.parse_args()
    
    for _,elem in ET.iterparse(get_file_lines(args.xml)):
        if elem.tag == 'entry':
            e = processed_elem(elem)
            e.process()
            e.report()
            elem.clear()
