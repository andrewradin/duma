from django import forms

class WsTissueForm(forms.Form):
    from browse.models import Tissue
    source = forms.ChoiceField(choices=Tissue.method_choices())
    geo_id = forms.CharField(max_length=256)
    tissue = forms.CharField(
        widget=forms.Textarea(attrs={'rows':'1', 'cols':'30'}),
        max_length=256,
        )
    tissue_set = forms.ChoiceField(
            choices=(('','None'),),
            )
    tissue_note = forms.CharField(
        widget=forms.Textarea(attrs={'rows':'2', 'cols':'30'}),
        required=False,
        )
    def __init__(self, ws, *args, **kwargs):
        super(WsTissueForm,self).__init__(*args, **kwargs)
        choices = []
        choices.append((0,'Excluded'))
        choices += ws.get_tissue_set_choices()
        self.fields['tissue_set'].choices = choices
        self.initial['tissue_set'] = choices[1][0]

class TissueFileForm(forms.Form):
    tsv = forms.ChoiceField(label='Import Significant Proteins:'
            ,choices=(('','None'),)
            ,required=False
            )
    tissue_set = forms.ChoiceField(
            choices=(('','None'),),
            )
    def __init__(self, ws, *args, **kwargs):
        super(TissueFileForm,self).__init__(*args, **kwargs)
        # reload choices on each form load
        from dtk.s3_cache import S3Bucket
        sigprot = S3Bucket('sigprot')
        choices = []
        choices.append(('','None'))
        suffix = '.tsv'
        for key in sigprot.list(cache_ok=True):
            if key.endswith(suffix):
                choices.append( (key,key[:-len(suffix)]) )
        self.fields['tsv'].choices = choices
        choices = []
        choices.append((0,'Excluded'))
        choices += ws.get_tissue_set_choices()
        self.fields['tissue_set'].choices = choices
        self.initial['tissue_set'] = choices[1][0]

class TissueEditForm(forms.ModelForm):
    class Meta:
        from browse.models import Tissue
        model = Tissue
        fields = ['name','ignore_missing']
    note = forms.CharField(
            widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
            required=False,
            )
    tissue_set = forms.ChoiceField(
            choices=(('','None'),),
            required=False,
            )
    def __init__(self, *args, **kwargs):
        skip_source = kwargs.pop('skip_source',False)
        super(TissueEditForm,self).__init__(*args, **kwargs)
        # reload choices on each form load
        choices = []
        choices.append((0,'Excluded'))
        tissue = kwargs['instance']
        choices += tissue.ws.get_tissue_set_choices()
        self.fields['tissue_set'].choices = choices
        self.initial['tissue_set'] = tissue.tissue_set_id
        self.initial['note'] = tissue.get_note_text()
        self.fields['name'].widget = forms.Textarea(attrs={'rows':'1','cols':'30'})
        if skip_source:
            return
        try:
            self.fields['source'] = forms.ChoiceField(
                    choices=tissue.method_choices(include_fallback=True),
                    initial = tissue.get_method_idx(),
                    )
            self.fields['fallback_reason'] = forms.CharField(
                    max_length=1024,
                    required=False,
                    initial = tissue.fallback_reason,
                    )
        except ValueError:
            pass

class form_defaults(object):
    def __init__(self, **kwargs):
        self.label = kwargs.get('label', None)
        self.initial = kwargs.get('initial', None)
        self.required = kwargs.get('required', True)
class choices_obj(form_defaults):
    def __init__(self, choices, init_ind = 0, **kwargs):
        super(choices_obj,self).__init__(**kwargs)
        self.choices = choices
        self.initial = choices[init_ind][0]
class sigGEO_settings(object):
    def __init__(self,ws):
# I dont' think we actually need to keep the species numbers in now that
# homologene is versioned, but leaving for now
        self.species = choices_obj((
                                    ('HUMAN_9606','Human'),
                                    ('MOUSE_10090','Mouse'),
                                    ('RAT_10116','Rat'),
                                    ('DOG_9615','Dog'),
                                    ('ZEBRAFISH_7955','Zebrafish'),
                                   ),
                                   label = 'Species'
                                  )
        self.algo = choices_obj((
                                 ('Limma','Limma'),
                                 ('all','Consensus'),
                                 ('SAM','SAM'),
                                 ('GeoDE','GeoDE'),
                                 ('RP','RP'),
                                ),
                                label='Algorithm'
                               )
        self.scRNAseq = form_defaults(label = 'Single-cell RNA-seq',
                                    initial = False,
                                    required = False
                                   )
        self.runSVA = form_defaults(label = 'Run SVA',
                                    initial = False,
                                    required = False
                                   )
        self.debug = form_defaults(label = 'Include debug output',
                                   initial = False,
                                   required = False
                                   )
        self.ignoreMissing = form_defaults(label = 'Proceed even if there is missing data',
                                   initial = False,
                                   required = False
                                  )
        self.top1thresh = form_defaults(label = 'Top 1% signal threshold',
                                       initial = 50
                                       )
        self.minReadPor = form_defaults(label = 'RNAseq min portion above read cutoff',
                                        initial = 0.5
                                       )
        self.minCPM = form_defaults(label = 'RNAseq min reads per million mapped',
                                    initial = 1
                                   )
        self.minDirPor = form_defaults(label = 'Min portion of probes agreeing on direction',
                                       initial = 0.66
                                      )
        self.minUniPor = form_defaults(label = 'Min portion of probes mapping to protein',
                                       initial = 0.35
                                      )
        self.permut = form_defaults(label = 'Permutations',
                                    initial = 100
                                   )
        from browse.default_settings import targetscan
        from dtk.s3_cache import S3File
        ws_v = targetscan.value(ws)
        s3f = S3File.get_versioned(
                'targetscan',
                ws_v,
                role='context_scores_human9606_ensembl',
                )
        s3f.fetch()
        self.mirMappingFile = choices_obj((
                    (s3f.path(),ws_v),
                ),
                label = 'miR - gene mapping file',
                )
    def as_dict(self):
        ks = [attr for attr in dir(self)
               if not callable(getattr(self, attr))
               and not attr.startswith("__")
              ]
        return {
                k:getattr(self, k).initial
                for k in ks
                }

class TissueSetForm(forms.ModelForm):
    class Meta:
        from browse.models import TissueSet
        model = TissueSet
        exclude = ['ws']

