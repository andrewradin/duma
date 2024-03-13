def get_html(url):
    import requests
    return requests.get(url).content

def get_three_part_html(base_url, target_prefix, target):
    return get_html(base_url+target_prefix+target)

class UrlConfig:
    '''
    Extract params from querystring, and rebuild URL with altered params.
    '''
    def __init__(self,request,defaults={}):
        # allow 'request' param to be a path string
        try:
            self.path = request.path
            self.options = request.GET.copy()
        except AttributeError:
            self.path = request
            self.options = {}
        self.defaults = defaults
    def add_defaults(self,more_defaults):
        self.defaults.update(more_defaults)
    def as_string(self,name):
        try:
            selected = self.options[name]
        except KeyError:
            try:
                selected = self.defaults[name]
            except KeyError:
                selected = ''
        return selected
    def as_bool(self,name):
        return self.as_int(name,0,1)
    def _as_numeric(self,numtype,name,low_limit=None,high_limit=None):
        try:
            selected = numtype(self.options[name])
            if low_limit is not None and selected < low_limit:
                raise ValueError
            if high_limit is not None and selected > high_limit:
                raise ValueError
            return selected
        except (KeyError,ValueError):
            try:
                return self.defaults[name]
            except KeyError:
                return low_limit
    def as_int(self,*args,**kwargs):
        return self._as_numeric(int,*args,**kwargs)
    def as_float(self,*args,**kwargs):
        return self._as_numeric(float,*args,**kwargs)
    def as_list(self,name):
        s = self.as_string(name)
        if s:
            return s.split(',')
        return []
    # XXX extend as necessary, tracking extract_ functions in browse/utils.py
    def modify(self,d):
        from dtk.data import modify_dict
        modify_dict(self.options,d)
    def here_url(self,d):
        from dtk.duma_view import qstr
        return self.path+qstr(self.options,**d)

# In the search_url functions below, the 'terms' parameter is a list of
# search terms which are ANDed. A term can be either a string, or an
# embedded list of synonyms which are ORed within the higher level query.
# The search_url functions work out how these need to be formatted for
# each source.
def google_term_format(term, patent=False):
    if isinstance(term,str):
        return '"%s"'%term
    if patent:
        return '"%s"' % '","'.join([x for x in term])
    return '("%s")' % '" OR "'.join([x for x in term])

def google_search_url(terms):
    url = UrlConfig("https://www.google.com/search")
    return url.here_url({'q':' '.join([google_term_format(x) for x in terms])})

def ct_search_url(drug=None,disease=None):
    url = UrlConfig("https://clinicaltrials.gov/ct2/results")
    opts = {}
    if drug:
        opts['term']=drug
    if disease:
        opts['cond']= disease
    if opts:
        return url.here_url(opts)

def eudra_ct_url(code):
    base_url = 'https://www.clinicaltrialsregister.eu/ctr-search/search?query='
    return base_url+code

def google_patent_search_url(terms):
    url = "https://patents.google.com/"
    return url + '?' + '&'.join(
               ['q='+google_term_format(x, patent=True) for x in terms]
            )

def google_patent_url(pat_no):
    return "https://www.google.com/patents/"+pat_no

def pubmed_term_format(term,suffix):
    if isinstance(term,str):
        return '(%s)'%(term+suffix)
    return '((%s))' % ') OR ('.join([x+suffix for x in term])

def pubmed_search_url(terms,restrictions=['Title','Abstract']):
    suffix='/'.join(restrictions)
    if suffix:
        suffix = '['+suffix+']'
    url = UrlConfig("http://www.ncbi.nlm.nih.gov/pubmed/")
    return url.here_url({'term':'('
                +') AND ('.join([pubmed_term_format(x,suffix) for x in terms])
                +')'
                })

# this doesn't support the OR level as above, but so far it's only used
# with a single term
def pubchem_search_url(terms):
    url = UrlConfig("https://www.ncbi.nlm.nih.gov/pccompound/")
    return url.here_url({'term':'+'.join(['"%s"'%x  for x in terms])})

def dbsnp_url(dbsnp_id):
    return "https://www.ncbi.nlm.nih.gov/snp/"+str(dbsnp_id)

def otarg_genetics_url(chrm, base, ref, alt):
    return f'https://genetics.opentargets.org/variant/{chrm}_{base}_{ref}_{alt}'

def pubmed_url(pubmed_id):
    return "http://www.ncbi.nlm.nih.gov/pubmed/"+str(pubmed_id)

def clinical_trials_url(study_id):
    return "https://clinicaltrials.gov/ct2/show/"+study_id

def agr_url(doid):
    return 'https://www.alliancegenome.org/disease/'+doid

def monarch_pheno_url(mondoID):
    return monarch_url(mondoID, 'phenotype')

def monarch_disease_url(mondoID):
    return monarch_url(mondoID, 'disease')

def monarch_url(mondoID, type):
    return f'https://monarchinitiative.org/{type}/{mondoID}'

def disgenet_url(umls_cui,page):
    page_id = dict(
            gda_sum=0,
            gda_ev=1,
            dda_sum=2,
            vda_sum=4,
            vda_ev=5,
            dis_map=6,
            )[page]
    return f'https://www.disgenet.org/browser/0/1/{page_id}/{umls_cui}/'

def ob_nda_url(nda_id):
    base_url='https://www.accessdata.fda.gov/scripts/cder/ob/'
    return base_url+'results_product.cfm?Appl_Type=N&Appl_No=%06d'%int(nda_id)

def pathway_url_factory():
    from dtk.gene_sets import legacy_genesets_name_to_id
    legacy_name_to_id = legacy_genesets_name_to_id()
    def func(pathway_name, **kwargs):
        return pathway_url(pathway_name, legacy_name_to_id, **kwargs)
    return func

def pathway_url(pathway_name_or_id, legacy_name_to_id, init_wsa=None):
    # If it's an ID or unknown name, it will pass through.
    id = legacy_name_to_id.get(pathway_name_or_id, pathway_name_or_id)

    if id.startswith('REACTOME'):
        # This is failed conversion, try pointing back at BROAD
        return 'http://software.broadinstitute.org/gsea/msigdb/cards/' + id
    else:
        init_args = {'initPathway': id}
        if init_wsa:
            init_args['initWsa'] = init_wsa
        # Could link to here, but we'll do internal instead.
        # base_url = 'https://www.reactome.org/content/detail/'
        import json
        return '/pathway_network/#' + json.dumps(init_args)

def ext_pathway_url(pathway_id):
    if pathway_id.startswith('R-'):
        return f"https://reactome.org/content/detail/{pathway_id}"
    if pathway_id.startswith('GO:'):
        return f"https://www.ebi.ac.uk/QuickGO/term/{pathway_id}"
    return ''

def open_targets_disease_url(key):
    return "https://www.targetvalidation.org/disease/"+key

def efo_url(key):
    #return "http://www.ebi.ac.uk/efo/"+key
    # The above just redirects to the URL below, so return that directly.
    return "https://www.ebi.ac.uk/ols/ontologies/efo/terms?short_form="+key

def medgen_url(key):
    return 'https://www.ncbi.nlm.nih.gov/medgen/'+key

def mesh_url(key):
    return "https://meshb.nlm.nih.gov/record/ui?ui="+key

def nature_reviews_disease_search_url(term):
    url = UrlConfig("https://www.nature.com/search")
    suffix='&order=relevance&article_type=reviews'
    return url.here_url({'q':'"%s" %s'%(term,suffix)})

def wikipedia_disease_search_url(term):
    url = UrlConfig("https://en.wikipedia.org/w/index.php")
    suffix='hastemplate:"Infobox medical condition (new)"'
    return url.here_url({'search':'"%s" %s'%(term,suffix)})

def wikipedia_drug_search_url(term):
    url = UrlConfig("https://en.wikipedia.org/w/index.php")
    suffix='hastemplate:"Infobox drug"'
    return url.here_url({'search':'"%s" %s'%(term,suffix)})

def drugbank_drug_url(key):
    return "http://www.drugbank.ca/drugs/%s" % key

def chembl_drug_url(key,section=None):
    result = "https://www.ebi.ac.uk/chembl/compound/inspect/%s" % key
    if section == 'indication':
        result += '#Indication'
    return result

def chembl_assay_url(key):
    return 'https://www.ebi.ac.uk/chembl/g/#browse/activities/filter/molecule_chembl_id:("%s")' % key

def chembl_lit_url(key):
    result = "https://www.ebi.ac.uk/chembl/g/#browse/documents/filter/_metadata.related_compounds.all_chembl_ids:%s" % key
    return result

def bindingdb_drug_url(key):
    k = key[4:]
    return "http://bindingdb.org/bind/chemsearch/marvin/MolStructure.jsp?monomerid=%s" % k

def bindingdb_purchase_drug_url(key):
    k = key[4:]
    return "https://www.bindingdb.org/bind/purchasable.jsp?monomerid=%s" % k

def globaldata_drug_url(key):
    k = key[2:]
    return "https://pharma.globaldata.com/DrugsView/ProductView?ProductId=%s" % k

def globaldata_drug_search_url(text):
    return f'https://pharma.globaldata.com/Drugs/SiteSearch?SearchText="{text}"'

def ge_eval_link():
    return 'https://twoxar.box.com/s/ihipvsjgjpmjoqybpix3hykihloqjdlo'

def biostudies_studies_url(accession):
    return f'https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}'

def ext_drug_links(wsa, split=False):
    from dtk.html import link
    ver = wsa.ws.get_dpi_version()
    ext_drug_links, ext_drug_search_links = wsa.agent.ext_src_urls(ver)
    ext_drug_search_links = [
            ('Search for drug on Google',
                        google_search_url([
                                        wsa.agent.canonical,
                                        ])
                        ),
            ('Search for drug on Wikipedia',
                        wikipedia_drug_search_url(
                                        wsa.agent.canonical,
                                        )
                        ),
            ('Search for drug on pubchem',
                        pubchem_search_url([
                                        wsa.agent.canonical,
                                        ])
                        ),
            ] + ext_drug_search_links

    ext_drug_disease_links = [
            ('Search for co-occurance with disease on Google',
                        google_search_url([
                                        wsa.agent.canonical,
                                        wsa.ws.get_disease_aliases(),
                                        ])
                        ),
            ('Search for co-occurance with disease on Google patents',
                        google_patent_search_url([
                                        wsa.agent.canonical,
                                        wsa.ws.get_disease_aliases(),
                                        ])
                        ),
            ('Search for co-occurance with disease on PubMed',
                        pubmed_search_url([
                                        wsa.agent.canonical,
                                        wsa.ws.get_disease_aliases(),
                                        ])
                        ),
            ('Search for co-occurance with disease on ClinicalTrials.gov',
                        ct_search_url(
                                       wsa.agent.canonical,
                                       wsa.ws.get_disease_default('ClinicalTrials'),
                                      )
                        ),
            ]
    linkify = lambda lst: [link(x[0],x[1],new_tab=True) for x in lst if x[1]]

    if split:
        return [("Drug Database Links", linkify(ext_drug_links)),
                ("Drug Searches", linkify(ext_drug_search_links)),
                ("Drug + Disease", linkify(ext_drug_disease_links))
                ]
    else:
        return linkify(ext_drug_links +
                ext_drug_search_links +
                ext_drug_disease_links)


def multiprot_search_links(ws, prots):
    uniprots = [x.uniprot for x in prots]
    genes = [x.gene for x in prots]

    all_names_nested = [x.get_search_names() for x in prots]
    # Flatten the names list, but mingle them so that it's not all one gene then all the next.
    # Mingling is good because google search may truncate the last words and/or prioritize earlier ones.
    from itertools import zip_longest
    all_names = [y for x in zip_longest(*all_names_nested) for y in x if y is not None]
    ext_links = make_prot_search_links(
        disease_names=ws.get_disease_aliases(),
        uniprots=uniprots,
        genes=genes,
        prot_names=all_names
    )
    from dtk.html import link
    pairs = [(cat, link(label,lnk,new_tab=True)) for label, lnk, cat in ext_links if lnk]
    from dtk.data import kvpairs_to_dict
    return kvpairs_to_dict(pairs)


def make_prot_search_links(disease_names, uniprots, genes, prot_names):
    links = [('Search for protein co-occurance with disease on google',
                    google_search_url([
                                    disease_names,
                                    uniprots,
                                    ]), 'srch',
                    ),
        ('Search for protein name co-occurance with disease on google',
                    google_search_url([
                                    disease_names,
                                    prot_names,
                                    ]), 'srch',
                    ),
        ('Search for gene co-occurance with disease on google',
                    google_search_url([
                                    disease_names,
                                    genes,
                                    ]), 'srch',
                    ),
        ('Search for gene co-occurance with disease on PubMed',
                    pubmed_search_url([
                                    disease_names,
                                    genes,
                                    ]), 'srch',
                    ),
        ]
    return links

def ext_prot_links(ws,prot):
    from dtk.html import link
    keys = prot.get_prot_attrs()
    result = []
    otarg_diseases = ws.get_disease_default('OpenTargets').split(',')
    for otarg_disease_full in otarg_diseases:
        try:
            otarg_disease =  otarg_disease_full.split(':')[1]
            for k in keys.get('Ensembl',[]):
                result.append( (f'OpenTargets Disease Association ({k}-{otarg_disease})',
                    "https://www.targetvalidation.org/evidence/%s/%s" % (k, otarg_disease),
                        'assoc') )
        except:
            # Usually means no opentargets have been run.
            pass

    top_links = [
        ('Uniprot', f'https://www.uniprot.org/uniprot/{prot.uniprot}/', 'db'),
        ('PDB', f'https://www.ebi.ac.uk/pdbe/pdbe-kb/proteins/{prot.uniprot}', 'db')
    ]

    search_links = make_prot_search_links(
        disease_names=ws.get_disease_aliases(),
        uniprots=[prot.uniprot],
        genes=[prot.gene],
        prot_names=prot.get_search_names(),
    )

    ext_links = top_links + prot.ext_src_urls() + result + search_links
    pairs = [(cat, link(label,lnk,new_tab=True)) for label, lnk, cat in ext_links if lnk]
    from dtk.data import kvpairs_to_dict
    return kvpairs_to_dict(pairs)

