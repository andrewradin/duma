
class Writer:
    data_suffix='_data.tsv.gz'
    name_suffix='_names.tsv'
    def __init__(self,prefix):
        from collections import Counter
        self.text_key_pairs = Counter()
        self.all_keys = set()
        self.mapped_keys = set()
        self.scores = {}
        from dtk.files import FileDestination
        # XXX if FileDestination handled automatic gzipping, the rstrip below
        # XXX could be removed, along with the subsequent gzip in the ETL
        # XXX Makefile
        self.data_fd = FileDestination(
                prefix+self.data_suffix.rstrip('.gz'),
                header=['disease_key','target_name','score_name','score_value']
                )
        self.name_fd = FileDestination(prefix+self.name_suffix)
    def add(self,disease_name,disease_key,target_name_uniprot_set,scores):
        self.text_key_pairs[(disease_name,disease_key)] += 1
        self.all_keys.add(disease_key)
        # About 12K of 16M OT records map to duplicate uniprots. We resolve
        # this by taking the best available score, but this requires holding
        # all scores in memory. There are ways around this, but they're more
        # complex, so for now just run this on a big enough machine.
        for i in target_name_uniprot_set:
            self.mapped_keys.add(disease_key)
            score_key = (disease_key,i)
            if score_key in self.scores:
                s = self.scores[score_key]
                for k,v in scores.items():
                    if v > s.get(k, 0):
                        s[k] = v
            else:
                self.scores[score_key] = scores
    def close(self):
        for (disease_key,i),scores in self.scores.items():
            for k,v in scores.items():
                if not v:
                    continue # don't write zero scores
                self.data_fd.append([disease_key,i,k,v])
        unmapped_keys = self.all_keys-self.mapped_keys
        if unmapped_keys:
            print(len(unmapped_keys),'diseases without mapped Ensembl')
            print(unmapped_keys)
        for rec,count in self.text_key_pairs.items():
            self.name_fd.append((rec[0],rec[1],count))

def api_lookup(term):
    import requests
    from dtk.duma_view import qstr
    endpoint='https://www.ebi.ac.uk/ols/api/search'
    url=endpoint+qstr({
            'q':term.strip(),
            'ontology':'efo',
            'rows':50,
            })
    rsp=requests.get(url)
    if rsp.status_code != 200:
        print('error: status',rsp.status_code,url)
        return
    try:
        d = rsp.json()['response']['docs']
        return [
                (doc['obo_id'].replace(':','_'),doc['label'],doc['iri'])
                for doc in d
                if 'obo_id' in doc # filters rare metadata entries, like
                # the following, which showed up when searching for
                # "Non-alcoholic Fatty Liver Disease (NAFLD)":
                # {"id":"efo:","iri":"http://www.ebi.ac.uk/efo/efo.owl","short_form":"efo","label":"Experimental Factor Ontology","description":["The Experimental Factor Ontology (EFO) provides a systematic description of many experimental variables available in EBI databases, and for external projects such as the NHGRI GWAS catalogue. It combines parts of several biological ontologies, such as anatomy, disease and chemical compounds. The scope of EFO is to support the annotation, analysis and visualization of data handled by many groups at the EBI and as the core ontology for OpenTargets.org"],"ontology_name":"efo","ontology_prefix":"EFO","type":"ontology","is_defining_ontology":false}
                ]
    except KeyError:
        print(rsp.content)
        raise


class OpenTargets:
    def __init__(self,choice):
        from dtk.s3_cache import S3File
        file_class='openTargets'
        self._f_data=S3File.get_versioned(file_class,choice,role='data')
        self._f_data.fetch()
        self._f_names=S3File.get_versioned(file_class,choice,role='names')
        self._f_names.fetch()
        self._known_keys = None

        from dtk.files import NoMatchesError
        try:
            self._f_target_safety=S3File.get_versioned(file_class,choice,role='target_safety')
            self._f_target_safety.fetch()
        except NoMatchesError:
            # Doesn't exist for older versions.
            self._f_target_safety = None

        try:
            self._f_target_tract=S3File.get_versioned(file_class,choice,role='tractability')
            self._f_target_tract.fetch()
        except NoMatchesError:
            # Doesn't exist for older versions.
            self._f_target_tract = None


    def get_disease_key(self,term):
        from dtk.files import get_file_records
        for rec in get_file_records(self._f_names.path()):
            if rec[0].lower() == term.lower().strip():
                return rec[1]
        return None
    def check_key(self,key):
        from dtk.files import get_file_records
        for rec in get_file_records(self._f_names.path()):
            if rec[1] == key:
                return True
        return False
    def get_disease_scores(self,key):
        from dtk.cache import Cacher
        cache = Cacher('open_targets')
        cache_key = f'{self._f_data.path()}.{key}'

        def compute_func():
            return self.get_multiple_diseases_scores([key])[key]

        return cache.check(cache_key, compute_func)
    
    def get_multiple_diseases_scores(self, keys):
        from dtk.files import get_file_records
        src = get_file_records(self._f_data.path(),
                select=(keys,0),
                )
        header = next(src)
        from collections import defaultdict
        d=defaultdict(lambda: defaultdict(dict))
        for rec in src:
            d[rec[0]][rec[1]][rec[2]]=float(rec[3])
        return d

    def search_disease(self,term):
        # find all matches from lookup service
        result = api_lookup(term)
        if not result:
            raise RuntimeError('OLS search failed')
        # get all names used in OpenTargets
        from dtk.files import get_file_records
        ot_data = {
                rec[1]:rec
                for rec in get_file_records(self._f_names.path())
                }
        # filter results to only those used in OpenTargets, and
        # add OpenTargets protein count
        alternatives = [['Disease Name','OT proteins','OBO Key','URL']]
        for obo_id,label,iri in result:
            if obo_id not in ot_data:
                continue
            ot_rec = ot_data[obo_id]
            alternatives.append([
                    label,
                    ot_rec[2],
                    obo_id,
                    iri,
                    ])
        return alternatives
    
    def get_prots_scores(self, prots, types):
        from dtk.files import get_file_records
        src = get_file_records(self._f_data.path(),
                select=(prots,1),
                )
        out = []
        dis_keys = set()
        for dis, prot, scoretype, value in src:
            if types and scoretype not in types:
                continue
            dis_keys.add(dis)
            out.append((dis, prot, scoretype, float(value)))
        
        dis_name_map = self.get_disease_key_name_map(dis_keys)
        out = [(x[0], dis_name_map.get(x[0], x[0]), *x[1:]) for x in out]

        return out

    def get_disease_key_name_map(self, dis_keys):
        from dtk.files import get_file_records
        out = {}
        select = (dis_keys, 1) if dis_keys is not None else None
        for rec in get_file_records(self._f_names.path(), select=select):
            out[rec[1]] = rec[0]
        return out
    
    def keys_from_mondo_ids(self, mondo, mondo_ids):
        if self._known_keys is None:
            from dtk.files import get_file_records
            self._known_keys = {rec[1] for rec in get_file_records(self._f_names.path())}

        out = []
        for mondo_id in mondo_ids:
            related_ids = mondo.mondo2other.fwd_map().get(mondo_id, [])
            key_overlap = self._known_keys & set(related_ids)
            out.append(key_overlap)

        return out

    def get_safety_data(self, uniprot):
        if not self._f_target_safety:
            return

        from django.utils.html import urlize
        from dtk.files import get_file_records
        from dtk.url import pubmed_url
        def make_pm_link(x):
            try:
                pmid = int(x)
                out = pubmed_url(pmid)
            except ValueError:
                out = x
            return urlize(out, trim_url_limit=20, autoescape=True)

        for rec in get_file_records(self._f_target_safety.path(), keep_header=False, select=([uniprot], 0)):
            rec[1] = [make_pm_link(x) for x in rec[1].split('|')]
            yield [rec[1], *[x.split('|') for x in rec[2:]]]

    def get_small_mol_druggable_prots(self):
        return self._general_sm_drugability()
    def is_small_mol_druggable(self,uniprot):
        p = self._general_sm_drugability(uniprot)
        return len(p) > 0
    def _general_sm_drugability(self,uniprot=None):
        if not self._f_target_tract:
            return None
        prots=set()
        from dtk.files import get_file_records
        header = None
        if uniprot:
            gen = get_file_records(self._f_target_tract.path(), keep_header=True, select=([uniprot], 'uniprot'))
        else:
            gen = get_file_records(self._f_target_tract.path(), keep_header=True)
        for frs in gen:
            if header is None:
                header = frs
                continue
            b,p=self._process_tract_record(header, frs)
# By changing the number 8 you can control how much data is required to consider a protein druggable
# 8 just means that there is any evidence (currently that means it's theoreticall druggable)
            if b['sm'] and min(b['sm']) <= 8:
                prots.add(p['uniprot'])
        return prots
    def get_tractability_data(self, uniprot):
        if not self._f_target_tract:
            return None

        from dtk.files import get_file_records
        recs = list(get_file_records(self._f_target_tract.path(), keep_header=True, select=([uniprot], 'uniprot')))
        if len(recs) == 1:
            # Just the header, no data
            return None

        # Temporarily disabled
        # assert len(recs) == 2, f"Found multiple records for {uniprot}"
        header, data = recs[:2]
        return self._process_tract_record(header,data)
    def _process_tract_record(self,header,data):
        bucket_regex = r'Bucket_(\d+)_(\w+)'
        props = dict(zip(header, data))

        from collections import defaultdict
        buckets = defaultdict(list)

        # list'ify so we can pop while iterating.
        for key, value in list(props.items()):
            import re
            m = re.match(bucket_regex, key)
            if m:
                props.pop(key)
                if int(value) != 0:
                    num, moltype = m.groups()
                    buckets[moltype].append(int(num))

        return buckets, props
