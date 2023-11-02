#!/usr/bin/env python

import sys
try:
    import path_helper
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

import django
django.setup()

#-------------------------------------------------------------------------------
# Feature Matrix Constructor
#-------------------------------------------------------------------------------
class builder:
    '''Using the identified training labels for GE dataset classifiers,
       build a feature matrix.
    '''
    def __init__(self, **kwargs):
        self.labels_file=kwargs.get('ifile', None)
        self.search_type=kwargs.get('type', None)
        self.ws_to_skip=kwargs.get('ws_to_skip', [
                                          '86', # combo drugs
                                          '84', # hcc 2
                                          '85', # hcc3
                                          '39', # ER/PR BC the subset isn't showing up in AE
                                          '41', # Graft vs Host/Stem cell transplant
                                          '42', # QA
                                          '2', # HIV/AIDS
                                          '38', # Trichuriasis
                                         ]
                              )
        self.k_sep = kwargs.get('sep',"|")
        self.sleep_time = 0.5
    def extract(self):
        self._load_labels()
        self._get_search_terms()
        self._get_data()
    def _get_data(self):
        self.tr_flag = self.search_type == 'TR'
        import xml.etree.ElementTree as ET
        from time import sleep
        from scripts.ae_xml_parser import xml_parser
        self.data={}
        for ws_id, term_list in self.search_terms.iteritems():
            missed_accessions = []
            for term in term_list:
                xp = xml_parser(disease=term, tr_flag = self.tr_flag, ws=ws_id)
### In theory we could just use all of the accessions,
### but then we wouldn't catch poorly mapping terms
                xp.run()
                gen = (k for k in self.labels
                       if k.startswith(ws_id+self.k_sep)
                       and k not in self.data
                      )
                for k in gen:
                    accession = self._get_acc_from_k(k)
                    if accession not in xp.results:
                        missed_accessions.append(k)
                        continue
                    self.data[k] = xp.results[accession]
                sleep(self.sleep_time)
            # this ensures we don't double count
            missed_accessions = [x for x in missed_accessions if x not in self.data]
            if missed_accessions:
                missed_accessions = set(missed_accessions)
                missed_accessions = self.add_geo(term_list, missed_accessions, ws_id=ws_id)
                if missed_accessions:
                    # the first term should be the most relevant
                    all_missed = self._combine_accs(missed_accessions)
                    xp = xml_parser(disease=term_list[0], tr_flag = self.search_type == 'TR', ws=ws_id)
                    xp.run()
                    for k in missed_accessions:
                        accession = self._get_acc_from_k(k)
                        if accession not in xp.results:
                            print 'Unable to find ' + accession + ' using ' + all_missed + ' in ' + term
                            continue
                        self.data[k] = xp.results[accession]
                    sleep(self.sleep_time)
    def add_geo(self, term_list, missed_accs, ws_id):
        # conduct a backup search against GEO
        from time import sleep
        from dtk.entrez_utils import GeoSearch
        from scripts.ae_xml_parser import xml_parser
        for term in term_list:
            run = xml_parser(disease=term, tr_flag = self.tr_flag, ws=ws_id)
            run.disease.get_highlights()
            # I'm not sure why we're getting some GEO search issues
            # but it's not worth untangling at this point
            try:
                gs = GeoSearch(term=term)
            except AssertionError:
                continue
            geo_set = set(gs.results.keys())
            # extract all GEO accessions found on AE
            prefix = 'E-GEOD-'
            ae_set = set([
                    'GSE'+x[len(prefix):]
                    for x in missed_accs
                    if x.startswith(prefix)
                    ])
            # add any missed GEO datasets into the results list
            missed = geo_set & ae_set
            from scripts.ae_xml_parser import Experiment
            for geoID in missed:
                data = gs.results[geoID]
                e = Experiment(geoID,run.disease,self.tr_flag)
                e.experiment_type = ''
                e.orig_title = data['title']
                e.title = run._prep_text(data['title'])
                e.orig_description = data['summary']
                e.description = run._prep_text(data['summary'])
                e.table_headers = ''
                e.table_vals = ''
                e.doi = 1.0 if 'pmid' in data else 0.0
                e.sample_n =  data['sample_n']
                self.data[geoID] = e
                missed_accs.remove(geoID.replace('GSE', prefix))
            if len(missed_accs) > 0:
                sleep(self.sleep_time)
            else:
                break
        return missed_accs
    def _combine_accs(self, missed_accessions):
        return('"' +
               '" OR "'.join([self._get_acc_from_k(x)
                              for x in missed_accessions
                            ]) +
              '"')
    def _get_search_terms(self):
        self._get_default_terms()
        self._overwrite_select_terms()
    def _get_default_terms(self):
        from browse.models import Workspace
        self.search_terms = {}
        for ws_id in self.ws_oi:
            if ws_id in self.ws_to_skip:
                continue
            ws = Workspace.objects.get(pk=int(ws_id))
            term = ws.name
            if ' ' in term:
                term = '"' + ws.name + '"'
            self.search_terms[ws_id] = [term]
    def _overwrite_select_terms(self):
        self.search_terms['27'] = ['"non-small cell lung carcinoma"',
                                   '"non-small cell lung"'
                                   # added this 2nd to try to catch as many as possible with the more specific term
                                  ] # from 'Non-Small-Cell Lung Adenocarcinoma'
        self.search_terms['23'] = ['"squamous cell lung carcinoma"',
                                   '"non-small cell lung"'
                                   # added this 2nd to try to catch as many as possible with the more specific term
                                  ] # from Non-Small-Cell Lung Carcinoma - Squamous
        self.search_terms['58'] = ['"Familial dilated cardiomyopathy"',
                                   'cardiomyopathy'
                                   # added this 2nd to try to catch as many as possible with the more specific term
                                  ] # from Dilated Cardiomyopathy (Familial)
        self.search_terms['55'].append('pancreas cancer') # in addition to Pancreatic cancer
        self.search_terms['51'].append('myeloma') # in addition to multiple myeloma
        self.search_terms['50'] += ['"ulcerative colitis"', '"crohns disease"'] # in addition to inflammatory bowel disease
        self.search_terms['53'] = ['"Non-alcoholic Fatty Liver Disease"', '"Nonalcoholic steatohepatitis"'] # from Non-alcoholic Fatty Liver Disease (NAFLD)
        self.search_terms['52'].append('"psoriatic arthritis"') # in addition to Psoriatic Arthropathy
        self.search_terms['89'].append('"Diffuse Large Cell Lymphoma"') # in addition to Diffuse Large B-Cell Lymphoma
        self.search_terms['82'] = ['"Hodgkins lymphoma"'] # from Hodgkin's Lymphoma
        self.search_terms['80'].insert(0,'"sickle cell disease"') # in addition to sickle cell anemia, putting this first
        self.search_terms['37'] = ['"familial hypercholesterolemia"'] # removed (FH)
        self.search_terms['70'] = ['"chronic myelogenous leukemia"'] # from chronic myeloid leukemia
        self.search_terms['66'] = ['"chronic obstructive pulmonary disease"'] # from Non-asthma, Non-emphysema COPD
        self.search_terms['79'] = ['"Becker muscular dystrophy"'] # from Becker's dystrophy
        self.search_terms['12'] = ['"parkinsons disease"']
        self.search_terms['76'] = ['"huntington disease"']
        self.search_terms['33'] = ['"Irritable Bowel Syndrome"'] # previously had (IBS) appended
        self.search_terms['73'].append('systemic sclerosis') # in addition to scleroderma
    def _load_labels(self):
        from dtk.files import get_file_records
        self.labels={}
        self.ws_oi = set()
        for frs in get_file_records(self.labels_file, parse_type='tsv'):
            # epilepsy has an inncorect search term that results in 5 erroneous examples
            if (
                frs[2] != self.search_type or
                (frs[0] == '67'and frs[1] in ['E-GEOD-33814',
                                   'E-GEOD-54099',
                                   'E-GEOD-11954',
                                   'E-GEOD-59045',
                                   'E-GEOD-6773'
                                  ]
                ) or
                frs[1].startswith('GDS')
            ):
                continue
            self.ws_oi.add(frs[0])
            k = self._get_k(frs[0], frs[1])
            v = self._get_v(frs[3])
            self.labels[k] = v
    def _get_acc_from_k(self, ae_acc):
        return ae_acc.split(self.k_sep)[1]
    def _get_k(self, wsid, ae_acc):
        return self.k_sep.join([wsid, ae_acc])
    def _get_v(self, s):
        if s == 'reject':
            return False
        return True
    def build_mat(self):
        import nltk
        from scripts.ae_xml_parser import ae_matrix_builder
        amb = ae_matrix_builder(self.data)
        amb.build_mat()
        self.data = None # I was running into memory issues
        self.final_mat = amb.final_mat
        self.final_cols = amb.final_cols
        self.final_rows = amb.final_rows
    def write_arff(self, ofn):
        from scripts.ae_xml_parser import write_sparse_arff
        if not hasattr(self, 'labels'):
            self._load_labels()
        write_sparse_arff("AE %s dataset classifier" % (self.search_type),
                          self.final_cols,
                          self.final_mat,
                          self.final_rows,
                          self.labels,
                          ofn
                         )
    def readin(self, ipkl):
        import pickle
        with open(ipkl, 'rb') as handle:
            self.data = pickle.load(handle)
    def save_data(self, opkl):
        import pickle
        with open(opkl, 'wb') as handle:
            pickle.dump(self.data, handle)

#-------------------------------------------------------------------------------
# Driver
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                description='build AE SEARCH matrix',
                )
    parser.add_argument("i", help="Training labels")
    parser.add_argument("search_type", help="CC or TR")
    parser.add_argument("--opkl", help="pickle to save the data to")
    parser.add_argument("--ipkl", help="pickle to read the data to")
    parser.add_argument("-o", default='full_ae_matrix.arff', help="Output file name. Default: %s(default)")
    args = parser.parse_args()

    b = builder(ifile=args.i, type=args.search_type)
    if not args.ipkl:
        b.extract()
    else:
        b.readin(args.ipkl)
    if args.opkl:
        b.save_data(args.opkl)
    b.build_mat()
    b.write_arff(args.o)
