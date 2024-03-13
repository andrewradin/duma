
class IndiMapper:
    '''Map disease names to the meddra standard.'''
    def __init__(self,meddra_version):
        import collections
        self.missed_names = collections.Counter()
        self.by_code = {}
        self.by_name = {}
        self.code2code = {}
        lookup_attrs = set([
                'adr_term',
                'synonym',
                ])
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned('meddra',meddra_version)
        s3f.fetch()
        from dtk.files import get_file_records
        for code,attr,val in get_file_records(s3f.path()):
            c = self.by_code.setdefault(code,{})
            c.setdefault(attr,set()).add(val)
            if attr in lookup_attrs:
                self.by_name.setdefault(val.lower(),set()).add(code)
            if attr == 'pt_llts':
                assert val not in self.code2code
                self.code2code[val] = code
        empties = [k for k,v in self.by_name.items() if not v]
    def translate(self,name):
        try:
            codeset = self.by_name[name]
        except KeyError:
            self.missed_names.update([name])
            return [(None,'miss')]
        if len(codeset) == 1:
            return [(list(codeset)[0],'exact_match')]
        return [
                (code,'multi_match')
                for code in codeset
                ]
    def code2name(self,code):
        if code in self.code2code:
            code = self.code2code[code]
        s = self.by_code[code]['adr_term']
        assert len(s) == 1
        return list(s)[0]


    def make_meddra_umls_converter(self, ws):
        from browse.default_settings import umls
        s3f = umls.get_s3_file(ws=ws, role='to_meddra')
        s3f.fetch()
        from dtk.files import get_file_records
        from dtk.data import MultiMap
        m2u = []
        for rec in get_file_records(s3f.path()):
            m2u.append((rec[1], rec[0]))
        m2uf = MultiMap(m2u).fwd_map()
        return m2uf

