from django import forms

from dtk.subclass_registry import SubclassRegistry

class KtSourceType(SubclassRegistry):
    usable_href = True
    @classmethod
    def extract_config(cls,**kwargs):
        # the user data collection forms for each KtSource add a per-class
        # prefix so that data for all sources can be collected in a single
        # form; strip that prefix here and return the form-specific field data
        prefix = cls._code_prefix()
        config = {}
        for k,v in kwargs.items():
            if k.startswith(prefix):
                base_k = k[len(prefix):]
                config[base_k] = v
        return config
    # The use of make_form_wrapper() allows make_form() to be installed as a
    # make_XXX_form method in a view class, while binding the ws variable
    # so the code below can get at it without itself being aware of the view.
    @classmethod
    def make_form_wrapper(cls,ws):
        def make_form(data):
            from dtk.dynaform import FormFactory
            ff=FormFactory()
            cls.add_form_fields(ff,ws)
            FormClass = ff.get_form_class()
            return FormClass(data)
        return make_form
    @classmethod
    def add_form_fields(cls,ff,ws):
        ff.add_field(
                cls._full_code('search_term'),
                forms.CharField(
                        initial=ws.get_disease_default(cls.vocab_name),
                        label=cls.src_name+' Search Term',
                        required=False,
                        ),
                )
    @classmethod
    def append_parsed_config(cls,supplied_sources,name,config):
        if config.get('search_term'):
            sts = config['search_term'].split("|")
            for st in sts:
                single_conf = config.copy()
                single_conf['search_term'] = st
                supplied_sources.append((name,cls,single_conf,None))
    @classmethod
    def form_name(cls):
        return cls.__name__+'_form'
    @classmethod
    def _code_prefix(cls):
        from dtk.table import codify
        return codify(cls.src_name+' ')
    @classmethod
    def _full_code(cls,stem):
        from dtk.table import codify
        return cls._code_prefix()+codify(stem)

class GlobalDataSource(KtSourceType):
    src_name='Global Data'
    cvt_list = lambda x:[y.strip() for y in x.split(';')] if x else []
    colmap=[
            ('title','Trial Title',None),
            ('identifiers','Trial Identifier',cvt_list),
            ('drug_name','Drug Name',None),
            ('trial_phase','Trial Phase',None),
            ('trial_status','Trial Status',None),
            ]
    @classmethod
    def add_form_fields(cls,ff,ws):
        ff.add_field(
                cls._full_code('tsv_data'),
                forms.CharField(
                        label=cls.src_name+' TSV Data',
                        required=False,
                        widget=forms.Textarea(attrs={'rows':'6','cols':'120'}),
                        strip=False,
                        help_text='<br>Must contain columns for '+', '.join([
                                name
                                for _,name,_ in cls.colmap
                                ]),
                        ),
                )
    @classmethod
    def append_parsed_config(cls,supplied_sources,name,config):
        txt = config.get('tsv_data')
        if txt:
            lines = [x.rstrip('\r') for x in txt.split('\n')]
            # strip blank lines from end
            while lines[-1] == '':
                lines.pop(-1)
            # remove spurious LibreOffice insertion on multi-line fields
            def cleaned(line):
                suffix = '_x000d_'
                if line.endswith(suffix):
                    line = line[:-len(suffix)]
                return line
            lines = [cleaned(x) for x in lines]
            # parse into fields, handling multi-line field case
            from dtk.parse_aact import yield_assembled_recs
            text_records = list(
                    yield_assembled_recs(iter(lines),delim='\t')
                    )
            # convert from field arrays to tuples
            from dtk.readtext import convert_records_using_colmap
            pasted_data = list(convert_records_using_colmap(
                    iter(text_records),
                    cls.colmap,
                    ))
            if pasted_data:
                desc = f'{len(pasted_data)} pasted records'
                # filter blank drugnames & combos
                skipped = 0
                combo_skipped = 0
                keep=[]
                for x in pasted_data:
                    if x.drug_name:
                        limit = 250
                        if len(x.drug_name) > limit:
                            raise ValueError(
                                    f'Drug name exceeds {limit} characters: '
                                    +x.drug_name
                                    )
                        if x.drug_name.startswith('(') and x.drug_name.endswith(')') and ' + ' in x.drug_name:
                            combo_skipped += 1
                        else:
                            keep.append(x)
                    else:
                        skipped += 1
                if skipped or combo_skipped:
                    if skipped:
                        desc += f'; {skipped} skipped b/c of no drug name'
                    if combo_skipped:
                        desc += f'; {combo_skipped} skipped b/c they were combos'
                    pasted_data = keep
                supplied_sources.append((name,cls,desc,pasted_data))
    @classmethod
    def load_results(cls,query,transient,version_defaults):
        from .models import IndicationMapper
        im = IndicationMapper()
        import re
        from dtk.url import clinical_trials_url, google_search_url, eudra_ct_url
        def select_id_from_prefix(row,prefix):
            ct_ids = [x for x in row.identifiers if x.startswith(prefix)]
            if ct_ids:
                # in cases of multiple NCT ids, it seems like clinicaltrials.gov
                # displays the same info for all ids, but just to produce
                # consistent output, we'll use the highest available number
                return sorted(ct_ids)[-1]
            return None
        def get_trial_url(row):
            ct_id = select_id_from_prefix(row,'NCT')
            if ct_id:
                return clinical_trials_url(ct_id)
            prefix = 'EudraCT'
            ct_id = select_id_from_prefix(row,'EudraCT')
            if ct_id:
                ct_id = ct_id[len(prefix):]
                ct_id = ct_id.lstrip(' -')
                return eudra_ct_url(ct_id)
            return None
        for row in transient:
            extra = [f'{row.trial_phase} {row.trial_status}']
            url = get_trial_url(row)
            if not url:
                # fall back to google search; put ids in extra
                url = google_search_url([row.title])
                extra.append(','.join(row.identifiers))
            m = re.match(r'Phase ([IV]+)',row.trial_phase)
            # treat Phase A/B like Phase A
            phase_map={
                    'I':1,
                    'II':2,
                    'III':3,
                    'IV':1,
                    }
            if m:
                phase = phase_map[m.group(1)]
                ind_val = im.indication_of_phase(phase)
            else:
                continue
            query.search.add_item(
                    query,
                    row.drug_name,
                    url,
                    ind_val,
                    extra = '; '.join(extra),
                    )

class OBKtSource(KtSourceType):
    src_name='Orange Book'
    vocab_name='OrangeBook'
    @classmethod
    def load_results(cls,query,transient,version_defaults):
        from dtk.orange_book import OrangeBook
        ob=OrangeBook(version_defaults['orange_book'])
        uc_set = set()
        prefix = 'U-'
        for term in query.search_term().split('|'):
            if term.startswith(prefix):
                uc_set.add(int(term[len(prefix):]))
            else:
                uc_set |= ob.get_use_codes_for_pattern(term)
        nda_set=ob.get_ndas_for_uses(uc_set)
        from dtk.url import ob_nda_url
        from browse.models import WsAnnotation
        for pr in ob.get_products():
            if pr.nda not in nda_set:
                continue
            for name in pr.parsed_name:
                query.search.add_item(
                        query,
                        name,
                        ob_nda_url(pr.nda),
                        WsAnnotation.indication_vals.FDA_TREATMENT,
                        )

class ChemblKtSource(KtSourceType):
    src_name='ChEMBL'
    vocab_name='Chembl'
    usable_href = False
    @classmethod
    def load_results(cls,query,transient,version_defaults):
        from dtk.indications import ChemblIndications
        ci = ChemblIndications()
        from .models import IndicationMapper
        im = IndicationMapper()
        from dtk.url import chembl_drug_url
        names = set()
        codes = set()
        import re
        for part in query.search_term().split('|'):
            # separate MESH codes from disease names
            if re.match(r'^D\d{6}$',part):
                codes.add(part)
            else:
                names.add(part.lower())
        for name,chembl_id,max_phase in ci.search(
                names=names,
                codes=codes,
                include_phase=True,
                ):
            query.search.add_item(
                        query,
                        name,
                        chembl_drug_url(chembl_id,section='indication'),
                        ind_val = im.indication_of_phase(max_phase),
                        extra = f'Max phase {max_phase}',
                        )

class AACTKtSource(KtSourceType):
    src_name='Clinical Trials'
    vocab_name='ClinicalTrials'
    @classmethod
    def load_results(cls,query,transient,version_defaults):
        from ctsearch.utils import ClinicalTrialsSearch
        cts = ClinicalTrialsSearch(
                disease=query.search_term(),
                phases=ClinicalTrialsSearch.phases,
                after=None,
                completed=True,
                drug=None,
                version_defaults=version_defaults,
                return_all=True,
                )
        from .models import IndicationMapper
        im = IndicationMapper()
        from dtk.url import clinical_trials_url
        for study in cts.study_list:
            href = clinical_trials_url(study.study)
            from dtk.aact import phase_name_to_number
            phase = phase_name_to_number(study.phase)
            ind_val = im.indication_of_phase(phase)
            extra = '%s (%s %s)' % (
                    ', '.join(sorted([
                            x.lower()
                            for x in study.interventions
                            ])),
                    study.start,
                    study.phase,
                    )
            for drug in study.drugs:
                query.search.add_item(
                        query,
                        drug,
                        href,
                        ind_val,
                        extra=extra,
                        )
