# Class for accessing FDA Orange Book data

class OrangeBook:
    def __init__(self,version):
        self._use_codes=None
        self._products=None
        self._patents=None
        self._version=version
    def open_file(self,name):
        from dtk.s3_cache import S3File
        s3f = S3File.get_versioned(
                'orange_book',
                self._version,
                role=name.split('.')[0],
                )
        s3f.fetch()
        return open(s3f.path())
    def get_use_codes(self):
        if self._use_codes is None:
            f=self.open_file('use_codes.csv')
            # skip BOM, if present
            if f.read(1) != '\ufeff':
                f.seek(0)
            import csv
            src = csv.reader(f)
            from dtk.readtext import convert_records_using_colmap
            self._use_codes = list(convert_records_using_colmap(src,[
                        ('code','Code',lambda x:int(x[2:])),
                        ('desc','Definition',lambda x:x.lower()),
                        ]))
            f.close()
            self._use_lookup = {
                        uc.code:uc
                        for uc in self._use_codes
                        }
        return self._use_codes
    def get_products(self):
        if self._products is None:
            f=self.open_file('products.txt')
            from dtk.readtext import parse_delim
            src=parse_delim(f,delim='~')
            from dtk.readtext import convert_records_using_colmap
            def parse_name(raw):
                names = raw.lower().split('; ')
                result = set(names)
                for name in names:
                    if ' ' in name:
                        result.add(name.split(' ')[0])
                return result
            self._products = list(convert_records_using_colmap(src,[
                    ('nda','Appl_No',int),
                    ('name','Ingredient',lambda x:x.lower()),
                    ('parsed_name','Ingredient',parse_name),
                    # the name field is actually a list of (active?)
                    # ingredients joined by '; '
                    #
                    # the parsed_name field is a set of names obtained by
                    # parsing the name field; in addition to splitting on
                    # '; ', if any of the resulting parts contain a space,
                    # the first word is added as a separate name; this
                    # may be overly aggressive, but handles both the
                    # case where the second word is a salt form, or where
                    # the string of words provides detail or derivation
                    # of a substance described in the first word
                    #
                    # this approach improves both patent flagging and
                    # KT searching
                    ]))
            f.close()
        return self._products
    def get_patents(self):
        if self._patents is None:
            f=self.open_file('patent.txt')
            from dtk.readtext import parse_delim
            src=parse_delim(f,delim='~')
            def prep_pat_no(x):
                suffix='*PED'
                if x.endswith(suffix):
                    return x[:-len(suffix)]
                return x
            from dtk.readtext import convert_records_using_colmap
            self._patents = list(convert_records_using_colmap(src,[
                    ('nda','Appl_No',int),
                    ('patent','Patent_No',prep_pat_no),
                    ('s_flag','Drug_Substance_Flag',lambda x:x=='Y'),
                    ('p_flag','Drug_Product_Flag',lambda x:x=='Y'),
                    ('use_code','Patent_Use_Code',
                            lambda x:None if x == '' else int(x[2:])
                            ),
                    ]))
            f.close()
        return self._patents
    def get_use_codes_for_pattern(self,pattern):
        return set([
                uc.code
                for uc in self.get_use_codes()
                if pattern in uc.desc
                ])
    def get_ndas_for_uses(self,uc_set):
        return set([
                pat.nda
                for pat in self.get_patents()
                if pat.use_code in uc_set
                ])
    def get_names_for_ndas(self,nda_set):
        prs = [
                pr
                for pr in self.get_products()
                if pr.nda in nda_set
                ]
        names = set()
        for pr in prs:
            names |= pr.parsed_name
        return names
    def get_ndas_for_names(self,name_set):
        name_set = set([x.lower() for x in name_set])
        return set([
                pr.nda
                for pr in self.get_products()
                #if pr.name in name_set
                if pr.parsed_name & name_set
                ])
    def get_patents_for_ndas(self,nda_set):
        self.get_use_codes() # assure lookup is loaded
        result = {}
        for pat in self.get_patents():
            if pat.nda not in nda_set:
                continue
            if pat.use_code:
                text = self._use_lookup[pat.use_code].desc
            else:
                text = '(unspecified)'
            result.setdefault(text,set()).add(pat.patent)
        from collections import namedtuple
        Type=namedtuple('PatentInfo',['text','pat_list'])
        return [
                Type(text,pat_list)
                for text,pat_list in result.items()
                ]

