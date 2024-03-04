#!/usr/bin/env python
from __future__ import print_function
import os, django, sys, argparse, re
from collections import defaultdict
sys.path.insert(1,"../../web1")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
django.setup()
from path_helper import PathHelper
from algorithms.exit_codes import ExitCoder

# A program to parse TTD data to produce DPI and Attribute files

# As of March 2016, (prior to duplicate name filtering) this produced:
# - 16677 drugs in the dpi file
# - 37708 drugs in the attributes file
#   - 15559 of these had some attribute beyond name
#     - 9136 of these had dpi info
#     - 6423 drugs had attributes only
#   - 22149 of these had name only
#     - 7541 of these had dpi info
#     - 14608 have neither dpi nor attributes
# So, almost 40% of the drug entries are meaningless; I added the
# --filter option to avoid outputting these records, but it
# requires running --dpi and --attr in the same pass.  They can
# still be run separately for testing (without --filter).
#
# After this change, and with duplicate name filtering added,
# and fixing a structural extraction problem that deprived some
# drugs of attributes, the numbers were:
# - 16260 drugs in the dpi file
# - 23586 drugs in the attributes file
#   - 20640 of these had some attribute beyond name
#     - 13313 of these had dpi info
#     - 7327 drugs had attributes only
#   - 2947 of these had name only
#     - 2947 of these had dpi info
#     - 0 have neither dpi nor attributes
#
# After modifying the filter option to only output drugs with
# DPI info, whether they had attributes or not, the stats were:
# - 16260 drugs in both attributes and dpi files
# - 13313 had some attribute beyond DPI info
# - 2947 had DPI info only

# A few subroutines to print out as I want

def message(*objs):
    print(*objs, file=sys.stderr)

def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)

def error(*objs):
    print("ERROR: ", *objs, file=sys.stderr)
    sys.exit(-1)


#=================================================
# Manage TTD attribute information
#=================================================
class TtdDrugCollection:
    fieldmap = [
            ('DrugName','canonical',''),
            ('CAS Number','cas','CAS '),
            ('PubChem CID','pubchem_cid','CID '),
            ('SuperDrug ATC','atc',lambda x:[y.strip() for y in x.split(';')]),
            ]
    struct_fields = []
    def get_drug_by_id(self,ttd_id):
        if ttd_id not in self.id2drug:
            self.id2drug[ttd_id] = {'ttd_id':ttd_id}
        return self.id2drug[ttd_id]
    # no need for multival attributes yet
    def add_attr(self,ttd_id,attr,value):
        d = self.get_drug_by_id(ttd_id)
        d[attr] = value
        if attr == 'canonical':
            if value in self.name2drug:
                prev_ttd_id = self.name2drug[value]['ttd_id']
                warning(ttd_id,'name',value,'previously used by',prev_ttd_id)
                self.name_dups.add(ttd_id)
                self.name_dups.add(prev_ttd_id)
            else:
                self.name2drug[value] = d
    def __init__(self,fname):
        self.dpi_drugs = None # change to collection to enable filtering
        self.id2drug = {}
        self.name2drug = {}
        self.name_dups = set()
        lookup = { x[0]:x for x in self.fieldmap }
        infile = open(fname,"r")
        for i,line in enumerate(infile):
            fields = line.strip('\n').split('\t')
            if len(fields) != 4:
                warning(len(fields),"on line",i,"; skipped")
                continue
            if fields[2] in lookup:
                map_info = lookup[fields[2]]
                try:
                    unicode(fields[3])
                except UnicodeDecodeError:
                    # this is really only necessary for names;
                    # in TTD, things with non-ascii names are
                    # both rare (4 in the most recent import)
                    # and junk (i.e. not real drug names)
                    warning(fields[0],fields[2],
                            "failed unicode conversion; skipped"
                            )
                    continue
                if map_info[2]:
                    if isinstance(map_info[2],basestring):
                        # it's a prefix we need to strip
                        assert fields[3].startswith(map_info[2])
                        fields[3] = fields[3][len(map_info[2]):]
                    else:
                        # it's a conversion function
                        fields[3] = map_info[2](fields[3])
                if isinstance(fields[3],basestring):
                    fields[3] = fields[3].strip()
                self.add_attr(fields[0],map_info[1],fields[3])
        # now eliminate any duplicate names, to prevent linking
        # drugs and DPI incorrectly
        for ttd_id in self.name_dups:
            name = self.id2drug[ttd_id]['canonical']
            if name in self.name2drug:
                del(self.name2drug[name])
            if ttd_id in self.id2drug:
                del(self.id2drug[ttd_id])
    def add_structures(self,fname):
        self.struct_fields = 'smiles_code inchi inchi_key'.split()
        from rdkit import Chem
        # RDKit has an SDMolSupplier class that's supposed to parse
        # .sdf files, but the one from TTD has extra blank lines that
        # RDKit doesn't like.  So, take it apart by hand, and parse
        # each MOL block.
        f = open(fname)
        mol_block = []
        for i,line in enumerate(f):
            if line == '$$$$\n':
                m = Chem.MolFromMolBlock(''.join(mol_block))
                mol_block=[]
                if not m:
                    warning('failed to construct molecule; line',i)
                    continue
                ttd_id = m.GetProp('_Name')
                if ttd_id not in self.id2drug:
                    warning('no drug matching',ttd_id,'; line',i)
                    continue
                inchi = Chem.MolToInchi(m)
                if inchi:
                    self.add_attr(ttd_id,'inchi',inchi)
                    self.add_attr(ttd_id,'inchi_key'
                                    ,Chem.InchiToInchiKey(inchi)
                                    )
                smiles = Chem.MolToSmiles(m,isomericSmiles=True)
                if smiles:
                    self.add_attr(ttd_id,'smiles_code',smiles)
                continue
            if line == '\n' and len(mol_block) > 2:
                continue # skip extra non-conforming blank lines
            mol_block.append(line)
    def output(self,fname):
        f = open(fname,'w')
        def wr(fields): f.write('\t'.join(fields)+'\n')
        wr('ttd_id attribute value'.split())
        attr_order=[x[1] for x in self.fieldmap] + self.struct_fields
        for key in sorted(self.id2drug.keys()):
            d = self.id2drug[key]
            assert 'canonical' in d
            # don't output drugs w/o DPI
            if self.dpi_drugs and key not in self.dpi_drugs:
                continue
            for attr in attr_order:
                if attr in d:
                    val = d[attr]
                    if type(val) == list:
                        for item in val:
                            wr( [d['ttd_id'], attr, item] )
                    else:
                        wr( [d['ttd_id'], attr, val] )
        f.close()

#=================================================
# Manage TTD dpi information
#=================================================
class TtdDpiParser:
    
    # unfortunately some targets have a drug listed multiple times. The only examples I've seen they are listed as an unknown direction and a directional.
    # To deal with this I'll have to store all of the interaction information for a target, and then go through it 
    
    def __init__(self,attrs,outfile):
        self.outfile = open(outfile,'w')
        self.attrs = attrs
        # just for ease, I'll start this correctly, though I could start it as a null and code it properly, but I know this is the first entry and always will be, so I'll keep it easy.
        self.currentTargetID="TTDS00001"
        self.drugDirections=dict()
        self.uniprot=[]
        from collections import Counter
        self.stats = Counter()
        self.written_ids = set()
    def output(self,*objs):
        print(*objs, file=self.outfile)

    def flush_prev_target(self):
        self.stats['target'] += 1
        if self.stats['target'] == 1:
            self.output("\t".join(
                        ('ttd_id','uniprot_id','evidence','direction')
                        ))
        if not self.uniprot:
            warning(self.currentTargetID,"has no uniprot mapping; skipped")
            return
        self.stats['has_uniprot'] += 1
        for drug in self.drugDirections.keys():
            self.stats['drug'] += 1
            if drug not in self.attrs.name2drug:
                warning(drug + ' has no id in attrs file')
                self.stats['no_id'] += 1
                continue
            mapped_drug = self.attrs.name2drug[drug]['ttd_id']
            # this is the special case where there was conflicting information on the direction, so we took the conservative approach of using 0
            if(self.drugDirections[drug]=="zero"):
                self.drugDirections[drug]=0
            self.written_ids.add(mapped_drug)
            for u in self.uniprot:
                self.stats['interaction'] += 1
                self.output("\t".join((mapped_drug
                                ,u
                                ,"1"
                                ,str(self.drugDirections[drug])
                                )))
        
    def process(self,inputfile):
        # make a list of the 2nd column entries that are drug related, separated by their directions
        ups = ['Activator','Agonist','Inducer','Stimulator']
        downs = ['Antagonist','Blocker','Inhibitor','Suppressor']
        unknowns =['Adduct','Antibody','Binder','Breaker','Drug(s)','Ligand','Modulator','Multitarget','Opener','Regulator']
        with open(inputfile, 'r') as f:
            for line in f:
                # ensure we're not at the end of the file
                info=line.rstrip().split("\t")
                
                # not all targets are hit by drugs
                if(len(info) < 3):
                    continue
                
                targetID = info[0]
                
                if targetID != self.currentTargetID: # this means we've gone to the next target
                    # so print out the results for the previous target
                    self.flush_prev_target()
                    
                    # now update things
                    self.currentTargetID = targetID
                    self.drugDirections = dict()
                    self.uniprot = []
                
                if info[1] == "UniProt ID":
                    self.uniprot.append(info[2])
                    continue
                
                # skip the lines without drug information    
                if(info[1] not in ups + downs + unknowns):
                    continue
                
                # pull out the relevant bits
                interactionType = info[1]
                drugName = info[2]
                
                # now add the current information
                # add the interaction
                # first get the current direction
                if(interactionType in ups):
                    direction=1
                elif(interactionType in downs):
                    direction=-1
                elif(interactionType in unknowns):
                    direction=0
                    
                # now check and see if this drug has previously been seen
                if(drugName in self.drugDirections.keys()):
                    if(self.drugDirections[drugName]=="zero"):
                        continue # this happens only if a conflict has previously been observed (see below)
                    elif(self.drugDirections[drugName]!=direction): # they're not equal
                        if(abs(direction) > self.drugDirections[drugName]): # this would be the case only if the stored direction was 0
                            self.drugDirections[drugName]=direction # in that case I want to replace with the more informative direction
                        elif(-1*direction==self.drugDirections[drugName]): # uh oh we have conflicting reports
                            warning(drugName + " is reported to both activate and inhibit " + currentTargetID + ". Thus putting direction to 0.")
                            self.drugDirections[drugName]="zero" # by setting it to a non-numeric I can recognize that there has been a conflict and keep from ever overwriting this. The string is handled elsewhere.
                    #else:
                        # do nothing, the direction is the same
                else:
                    self.drugDirections[drugName]=direction
                
            self.flush_prev_target()
            message(self.stats)


#=================================================
# Read in the arguments/define options
#=================================================

if __name__ == '__main__':
    import argparse
    attr_out_filename='attr.out.tsv'
    dpi_out_filename='dpi.out.tsv'
    parser = argparse.ArgumentParser(description='TTD file conversion utility')
    parser.add_argument('--dpi'
            ,help='TTD target input file; create %s' % dpi_out_filename
            )
    parser.add_argument('--attr',action='store_true'
            ,help='create %s' % attr_out_filename
            )
    parser.add_argument('--filter',action='store_true'
            ,help='skip drugs with no dpi and no attributes other than name'
            )
    parser.add_argument('--struct'
            ,help='TTD sdf input file; add SMILES and InChI; implies --attr'
            )
    parser.add_argument('crossmatch_file'
            ,help='TTD crossmatch input file'
            )
    args = parser.parse_args()

    if args.filter and not args.dpi:
        error("--dpi required for --filter")

    attrs = TtdDrugCollection(args.crossmatch_file)
    if args.struct:
        attrs.add_structures(args.struct)
        args.attr = True
    if args.attr:
        attrs.output(attr_out_filename)
    if args.dpi:
        dpis = TtdDpiParser(attrs, dpi_out_filename)
        dpis.process(args.dpi)

    if args.attr:
        if args.filter:
            attrs.dpi_drugs = dpis.written_ids
        attrs.output(attr_out_filename)

