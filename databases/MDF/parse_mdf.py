#!/usr/bin/env python

from collections import Counter
from map_drugs import DrugNameMapper
import re
try:
    from dtk.files import get_file_lines, get_file_records
except ImportError:
    import sys
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from dtk.files import get_file_lines, get_file_records

verbose=False

def parse_meddra(fn):
    d={}
    for frs in get_file_records(fn, keep_header=False):
        if frs[1] != 'adr_term' and frs[1] != 'synonym':
            continue
        k=frs[2].lower()
        if k not in d:
            d[k] = set()
        d[k].add(frs[0])
    return d

class line_parser:
    def __init__(self,header_line):
        self.null='\N'
        self.header = self.split_line(header_line)
        self.c = Counter()
        self.d = DrugNameMapper()
        self._set_inds()
        self._MDF_specific_drugs()
        self._MDF_specific_diseases()
    def _set_inds(self):
# this need not be specific to medications, but in practice it seems to be, with the exception of the last one
        other_val_inds = [i for i,s in enumerate(self.header) if s == 'Other Value']
        med_inds = [i for i,s in enumerate(self.header) if s == 'Which of the following medications do you currently take?']
        disease_other_val_inds = other_val_inds.pop()
        self.inds={'id':self.header.index('Patient ID'),
                   'age':self.header.index('Age'),
                   'sex':self.header.index('Gender'),
                   'diagnosis':self.header.index('"What is your diagnosis, according to your doctor?"'),
                   'relation':self.header.index('Your Relationship to Affected Individual'),
                   'race':{'w':self.header.index('White'),
                           'b':self.header.index('Black African/African American'),
                           'a':self.header.index('Asian'),
                           'm':self.header.index('Mixed'),
                           'l':self.header.index('Hispanic or Latino')
                          },
                  'diseases':{'diabetes':self.header.index("Do you have a doctor's diagnosis of diabetes?"),
                              'addiction':self.header.index("Addiction"),
                              'anxiety disorder':self.header.index("Anxiety disorder"),
                              'conduct disorder':self.header.index("Conduct disorder"),
                              'depression':self.header.index("Depression"),
                              'panic attacks':self.header.index("Panic attacks"),
                              'schizophrenia':self.header.index("Schizophrenia"),
                              "asperger's syndrome":self.header.index("Asperger Syndrome"),
                              'autism':self.header.index("Autism"),
                              'obsessive-compulsive disorder':self.header.index("Obsessive Compulsive Disorder (OCD)"),
                              'pervasive developmental disorder':self.header.index("Pervasive Development Disorder (PDD)"),
                              'attention deficit disorder':self.header.index("Has a doctor diagnosed you with Attention Deficit Disorder (ADD) or Hyperactivity Disorder (ADHD)?"),
                              'benign tumor':self.header.index("Have you ever been diagnosed with a benign (non-cancerous) tumor?"),
                              'cancer':self.header.index("Have you ever been diagnosed with any type of cancer (malignant tumor)?"),
                             },
                  'drugs':{'Mexiletine':self.header.index('Mexiletine'),
                           'Phenytoin': self.header.index('Phenytoin'),
                           'Carbamazepine': self.header.index('Carbamazepine'),
                             },
                  'diseases_to_parse':[disease_other_val_inds],
                  'drugs_to_parse':[self.header.index('"Please list additional strategies, including any medications, you use to manage challenges with motor function."'),
                                    self.header.index('"Please list additional strategies, including any medications, you use to manage symptoms of myotonia."'),
                                    self.header.index('"Please list additional strategies, including any medications, you use to manage cardiac-related challenges."'),
                                    self.header.index('"Please list additional strategies, including any medications, you use to manage challenges with breathing."'),
                                    self.header.index('"Please list additional strategies, including any medications, you use to manage challenges with swallowing or feeding."'),
                                    self.header.index('"Please list additional strategies, including any medications, you use to manage challenges with sleep."'),
                                    self.header.index('"Please list additional strategies, including any medications, you use to manage pain."'),
                                    self.header.index('Please list additional strategies you use to manage vision challenges.'),
                                  ]+other_val_inds+med_inds
                 }
        if verbose:
            all_inds_used = []
            for v in self.inds.values():
                try:
                    float(v)
                    all_inds_used.append(v)
                except TypeError:
                    if type(v) == type([]):
                        all_inds_used += v
                    elif type(v) == type({}):
                        all_inds_used += v.values()
                    else:
                        print 'couldnt find', v
            for i, h in enumerate(self.header):
                if i not in all_inds_used:
                    print h, 'wasnt used'
    def _MDF_specific_diseases(self):
        # these are drugs that were identified in the text (free response sections),
        # by looking at individual word counts (there were few enough responses)
        # diseases that had their own columns were excluded from this list
        self.diseases={x:x for x in [
                 'dysgraphia',
                 'dsylexia',
                 'pseudobulbar affect',
                ]}
        self.diseases['mr'] = 'mitral regurgitation'
        self.diseases['ptsd'] = 'post-traumatic stress disorder'
        self.diseases['post tramatic stress disorder'] = 'post-traumatic stress disorder'
        self.diseases['bipolar'] = 'bipolar disorders'
        self.diseases['dsylexia'] = 'dyslexia'
    def _MDF_specific_drugs(self):
        # these are drugs that were identified in the text (free response sections),
        # but were not mapping well.
        # These were confirmed to map to CAS numbers before being added here
        drug_names=['trazodone',
                    'sotalol',
                    'phenytoin',
                    'metoprolol',
                    'atorvastatin',
                    'warfarin',
                    'methotrexate',
                    'carvedilol',
                    'tramadol',
                    'rabeprazole',
                    'myprodol',
                    'mexiletine',
                    'hydrocodone',
                    'motrin',
                    'baclofen',
                    'cyclobenzaprine',
                    'eptoin',
                    'carbamazepine',
                    'glucosamine',
                    'berotec',
                    'dantrolene',
                    'timolol',
                    'pseudoephedrine',
                    'morphine',
                    'selenium',
                    'propranolol',
                    'levothyroxine',
                    'naproxen',
                    'alprazolam',
                    'flexeril',
                    'albuterol',
                    'pyridostigmine',
                    'arginine',
                    'duloxetine',
                    'rifaximin',
                    'valerian',
                    'clonidine',
                    'clonazepam',
                    'lyrica',
                    'omeprazole',
                    'hyoscyamine',
                    'nurofen',
                    'domperidone',
                    'aspirin',
                    'skelaxin',
                    'zolpidem',
                    'melatonin',
                    'xanax',
                    'oxycodone',
                    'caffeine',
                    'restoril',
                    'codeine',
                    'gabapentin',
                    'neurontin',
                    'barium',
                    'modafinil',
                    'quinine',
                    'ibuprofen',
                    'verapamil',
                    'cimetidine',
                    'creatine',
                    'ibuprophen',
                    'potassium',
                    'miralax',
                    'diazepam',
                    'lidocaine',
                    'paracetamol',
                    'magnesium',
                    'fenitoina',
                    'amiodarona',
                    'venlafaxine',
                    'erythromycin',
                    'tizanidine',
                    'imipramine',
                    'daytrana',
                    'methylphenidate',
                    'lisinopril',
                    'bupropion',
                    'armodafinil',
                    'topiramate',
                    'meloxicam',
                    'acetazolamide',
                    'spironolactone',
                    'prednisolone',
                    'vivactil',
                    'oxcarbazepine',
                    'benfotiamine',
                    'digoxin',
                    'dronabinol',
                    'celebrex',
                    'doxepin',
                    'insulin',
                    'latanoprost',
                    'diclofenac',
                    'zaleplon',
                    'phentermine',
                   ]
        self.named_drugs={dn:set() for dn in drug_names}
        for dn in drug_names:
            for cas,how in self.d.translate(dn):
                assert cas is not None and how != 'miss'
                self.named_drugs[dn].add(cas)
    def split_line(self, l):
        return l.strip('\n').split('\t')
    def parse(self,line):
# set up
        fields = self.split_line(line)
        id = fields[0]
        results = {}
        results['age'] = int(float(fields[self.inds['age']]))
        results['sex'] = fields[self.inds['sex']].lower()[:1]
        results['cas'] = set()
        results['race'] = set()
        results['drugs'] = set()
        results['diseases'] = set(['myotonic dystrophy']) # by definition if they're in this dataset
# extract the easy ones
        for t in ['race', 'diseases', 'drugs']:
            for k,i in self.inds[t].iteritems():
                if fields[i] and fields[i].lower() != 'no':
                    results[t].add(k)
#        if fields[self.inds['diagnosis']]:
#            results['diseases'].add(fields[self.inds['diagnosis']])
# process race
        if 'race' in results:
            results['race'] = ''.join(sorted(list(results['race'])))
# process drugs
        for i in self.inds['drugs_to_parse']:
            if fields[i] == 'n/a':
                continue
            words = [x.strip().lower() for x in re.split("\W+",fields[i])]
            for word in words:
                if word in self.named_drugs:
                    results['cas'].update(self.named_drugs[word])
# This was used to refine the drug list above
#            for cas, how in self.d.translate(fields[i]):
#                if cas and cas not in results['cas']:
#                    print 'FOUND:', cas, how, fields[i]
        for i,x in enumerate(results.get('drugs', [])):
            for cas,how in self.d.translate(x):
                if cas:
                    results['cas'].add(cas)
                self.c.update([how])
# disease
        for i in self.inds['diseases_to_parse']:
            if fields[i]:
                s = fields[i].lower()
                for k in self.diseases:
                    if k in s:
                        results['diseases'].add(self.diseases[k])
        return id,results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    description='extract relevant data from the person data',
                    )
    parser.add_argument('infile')
    parser.add_argument('meddra')
    parser.add_argument('drugfile')
    parser.add_argument('indifile')
    parser.add_argument('demofile')
    args=parser.parse_args()
    meddra_diseases = parse_meddra(args.meddra)
    missed_diseases=Counter()
    with open(args.drugfile,"w") as d_out:
        with open(args.indifile,"w") as i_out:
            with open(args.demofile,"w") as o_out:
                lp = None
                for line in get_file_lines(args.infile):
                    if lp is None:
                        lp = line_parser(line)
                        continue
                    id,results = lp.parse(line)
                    for x in results['cas']:
                        d_out.write("\t".join([id, x]) + "\n")
                    for x in results['diseases']:
                        if x in meddra_diseases:
                            i_out.write("\t".join([id, x]) + "\n")
                        else:
                            missed_diseases.update([x])
                    out = [str(x) for x in
                           [results['age'] if results['age'] else lp.null,
                            lp.null,
                            results['sex'] if results['sex'] else lp.null,
                            results['race'] if results['race'] else lp.null,
                           ]
                          ]
                    o_out.write("\t".join([id]+out) + "\n")
    print missed_diseases
# this was all used to get the list of named drugs
#    possible_words = []
#    for k,c in lp.missed_word_c.iteritems():
#        if len(k) < 3:
#            continue
#        for cas,how in lp.d.translate(k):
#            if cas is not None and how == 'hit':
#                possible_words.append(k)
#    print possible_words
#    to_review = {w:set() for w in possible_words}
#    for id, results in temp:
#        for w in possible_words:
#            for s in results.get('drugs', []):
#                if w in s:
#                    to_review[w].add(s)
#    with open('to_review', 'w') as f:
#        for w,l in to_review.iteritems():
#            f.write('#'+w+'\n')
#            f.write('\n'.join(l) + '\n')
