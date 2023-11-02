from dtk.subclass_registry import SubclassRegistry
from browse.models import VersionDefault, Workspace

import logging
logger = logging.getLogger(__name__)

class Defaultable(SubclassRegistry):
    """
    Settings with a default value globally and/or per workspace.
    Changes to these settings are tracked in the audit log.

    In addition to methods here, subclasses can override:
        choices: Return list of select choices, str class, or float class.
        workspace_default: If it shouldn't pull worskpace default from global.
    """
    form_type = 'choice'
    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    def value(cls, ws):
        if isinstance(ws, (str,int)):
            ws = Workspace.objects.get(pk=ws)
        ws_id = ws.id if ws is not None else None
        try:
            out = VersionDefault.objects.get(ws_id=ws_id, file_class=cls.name()).choice
        except VersionDefault.DoesNotExist:
            # This triggers the default creation logic.
            VersionDefault.get_defaults(ws_id)
            out = VersionDefault.objects.get(ws_id=ws_id, file_class=cls.name()).choice
        if cls.form_type == float:
            return float(out)
        elif cls.form_type == bool:
            return out == 'True'
        elif cls.form_type == 'choice' or cls.form_type == str:
            return out
        else:
            raise Exception(f"Unexpected form type {cls.form_type}")

    @classmethod
    def display_value(cls, ws):
        value = cls.value(ws)
        if cls.form_type == 'choice':
            choices = cls.choices(ws)
            for choice in choices:
                if choice[0] == value:
                    return choice[1]

        return value

    @classmethod
    def set(cls, ws, value, username):
        """Sets the value directly.

        This is mostly used for tests, you don't often programatically set these.
        """
        ws_id = None if ws is None else ws.id
        VersionDefault.set_defaults(ws_id, [(cls.name(), value)], username)

    @classmethod
    def has_global_default(cls):
        return True

    @classmethod
    def default_global_default(cls):
        # This is used quite rarely.
        # Right when you've first created a new default type, it'll be used.
        # Otherwise, the only place it gets used is in tests.
        #
        #
        # For a choice field, we'll default to the last entry.
        # Otherwise, you should override this.
        try:
            return [x for x in cls.choices(ws=None)][-1][0]
        except IndexError:
            if cls.name() == 'test':
                # This only really happens in tests, the test fileClass is used
                # for experiments and isn't usable, just ignore it.
                return ''
            logger.error("No choices for %s", cls.name())
            raise

class VDefaultable(Defaultable):
    """Versioned ETL files are all the same, just subclass this."""
    @classmethod
    def choices(cls, ws):
        return Workspace.get_versioned_file_choices(cls.name())

    @classmethod
    def latest_version(cls):
        # Latest appears first in the list.
        return cls.choices(None)[0][0]


    @classmethod
    def get_s3_file(cls, ws=None, role=None, fetch=True, latest=False):
        if latest:
            chosen = cls.latest_version()
        else:
            chosen = cls.value(ws=ws)
        from dtk.s3_cache import S3File
        s3_file = S3File.get_versioned(cls.name(), chosen, role=role)
        if fetch:
            s3_file.fetch()
        return s3_file


class DiseaseShortName(Defaultable):
    help_text='Disease abbreviation, used e.g. in plots.'
    form_type=str
    @classmethod
    def workspace_default(cls, ws):
        return ws.make_short_name()
    @classmethod
    def default_global_default(cls):
        return ''
    @classmethod
    def has_global_default(cls):
        return False

class DpiDataset(Defaultable):
    help_text='Drug/Molecule-protein interaction data. \
            Molecule clustering/matching version is also set to this.'
    @classmethod
    def choices(cls, ws):
        from dtk.prot_map import DpiMapping
        return DpiMapping.choices(ws)
    @classmethod
    def default_global_default(cls):
        # Note that the bottom choice is typicaly going to be uniprot.default,
        # which is probably not what we want to default to.
        from dtk.prot_map import DpiMapping
        return DpiMapping.preferred
    @classmethod
    def latest_version(cls):
        # Latest appears first in the list.  This determines global default page latest.
        # This is a bit confusing because the choices are grouped unlike most other selectors.
        return cls.choices(None)[0][1][0][0]

class DpiThreshold(Defaultable):
    help_text='Algorithms will use binding data with evidence >= this value.'
    form_type=float
    @classmethod
    def default_global_default(cls):
        from dtk.prot_map import DpiMapping
        return DpiMapping.default_evidence

class PpiDataset(Defaultable):
    @classmethod
    def choices(cls, ws):
        from dtk.prot_map import PpiMapping
        return PpiMapping.choices()

    @classmethod
    def default_global_default(cls):
        from dtk.prot_map import PpiMapping
        return PpiMapping.preferred

    @classmethod
    def latest_version(cls):
        # Latest appears first in the list.  This determines global default page latest.
        return cls.choices(None)[0][0]

class PpiThreshold(Defaultable):
    form_type=float
    @classmethod
    def default_global_default(cls):
        from dtk.prot_map import PpiMapping
        return PpiMapping.default_evidence

class EvalDrugset(Defaultable):
    help_text = 'Reference treatments to compare against for evaluation and tuning'
    @classmethod
    def choices(cls, ws):
        if ws:
            return ws.get_wsa_id_set_choices(train_split=False,test_split=False)
        else:
            return Workspace.get_fixed_wsa_id_set_choices()

    @classmethod
    def workspace_default(cls, ws):
        # Grab the global default; we just override this so that
        # "select global defaults" doesn't modify this one.
        return cls.value(ws=None)


class IntolerableUniprotsSet(Defaultable):
    help_text = 'Targets too toxic to consider for this disease'
    @classmethod
    def choices(cls, ws):
        if ws:
            out = ws.get_uniprot_set_choices(auto_dpi_ps=False)
            # wsunwanted includes this set, so would create a loop.
            return [x for x in out if x[0] != 'autops_wsunwanted']
        else:
            return Workspace.get_global_uniprot_set_choices()

    @classmethod
    def workspace_default(cls, ws):
        # Grab the global default; we just override this so that
        # "select global defaults" doesn't modify this one.
        return cls.value(ws=None)

class DiseaseNonNovelUniprotsSet(Defaultable):
    help_text = 'Targets already investigated in this disease'
    @classmethod
    def choices(cls, ws):
        if ws:
            out = ws.get_uniprot_set_choices(auto_dpi_ps=True)
            # wsunwanted includes this set, so would create a loop.
            return [x for x in out if x[0] != 'autops_wsunwanted']
        else:
            return []
    @classmethod
    def workspace_default(cls, ws):
        return cls.choices(ws)[0][0]
    @classmethod
    def default_global_default(cls):
        return ''


class GeneSets(Defaultable):
    help_text = 'Which genesets to use for pathway scores'
    @classmethod
    def choices(cls, ws):
        from dtk.gene_sets import gene_set_choices
        return gene_set_choices()
    @classmethod
    def latest_version(cls):
        # Latest appears first in the list.  This determines global default page latest.
        return cls.choices(None)[0][0]

class GESigBaseComboSig(Defaultable):
    help_text = 'T/R GESig JobID for a base combo drug to subtract from other gesig results (or blank)'
    required = False
    form_type=str

    @classmethod
    def workspace_default(cls, ws):
        return ''
    @classmethod
    def default_global_default(cls):
        return ''
    @classmethod
    def has_global_default(cls):
        return False

class OmicsSearchModel(Defaultable):
    help_text = 'Model to use for scoring omics search results'
    required = False

    @classmethod
    def choices(cls, ws):
        from runner.models import Process
        from dtk.text import fmt_time
        jobs = Process.objects.filter(
            name='gesearchmodel',
            status=Process.status_vals.SUCCEEDED,
        ).order_by('-id')

        return [(p.id, f'{p.name} {p.id} {fmt_time(p.completed)}') for p in jobs]

    @classmethod
    def default_global_default(cls):
        return ''



class FaersIncludeSex(Defaultable):
    required=False
    form_type=bool
    @classmethod
    def help_text(cls, ws):
        if ws:
            demos_url = ws.reverse('faers_demo')
            demos_link = f'See <a href="{demos_url}">Demographics</a> to pick good defaults for this workspace.'
        else:
            demos_link = ''
        return f'Which demographics to require for FAERS.{demos_link}'
    @classmethod
    def default_global_default(cls):
        return True

class FaersIncludeDate(Defaultable):
    required=False
    form_type=bool
    @classmethod
    def default_global_default(cls):
        return True

class FaersIncludeAge(Defaultable):
    required=False
    form_type=bool
    @classmethod
    def default_global_default(cls):
        return True

class FaersIncludeWeight(Defaultable):
    required=False
    form_type=bool
    @classmethod
    def default_global_default(cls):
        return True

class aact(VDefaultable): pass
class agr(VDefaultable): pass
class monarch(VDefaultable): pass
class disgenet(VDefaultable): pass
class duma_gwas(VDefaultable): pass
class duma_gwas_v2d(VDefaultable): pass
class duma_gwas_v2g(VDefaultable): pass
class faers(VDefaultable): pass
class lincs(VDefaultable): pass
class mesh(VDefaultable): pass
class openTargets(VDefaultable): pass
class targetscan(VDefaultable): pass
class salmon(VDefaultable):
    @classmethod
    def choices(cls, ws):
        initial = Workspace.get_versioned_file_choices(cls.name())
        unflavored = [(x[0].split('.')[-1],x[1].split('.')[-1]) for x in initial]
        # v1 no longer works with our install of Salmon
        # it was generated with v 0.8.2 of Salmon (we're on 1.1 or higher)
        unflavored = [x for x in unflavored if x[0] != 'v1']
        return list(dict.fromkeys(unflavored))
class efo(VDefaultable): pass
class test(VDefaultable):
    visible = False
class ucsc_hg(VDefaultable):
    visible = False
    def value(ws):
        from dtk.etl import get_etl_dependencies
        v = duma_gwas.value(ws)
        dw_deps = get_etl_dependencies('duma_gwas', int(v.lstrip('v')))
        temp = [x[1] for x in dw_deps
               if x[0]=='ucsc_hg'
            ]
        assert len(temp) ==1, 'incorrect number of UCSC_HG dependencies in GWAS'
        return temp[0]


class uniprot(VDefaultable): pass
class homologene(VDefaultable):
    @classmethod
    def choices(cls, ws):
        initial = Workspace.get_versioned_file_choices(cls.name())
        unflavored = [(x[0].split('.')[-1],x[1].split('.')[-1]) for x in initial]
        return list(dict.fromkeys(unflavored))

class string(VDefaultable):
    """Gets configured separately via PPI selector."""
    visible = False
class umls(VDefaultable): pass
class meddra(VDefaultable): pass
class mondo(VDefaultable): pass
class unichem(VDefaultable): pass
class zinc(VDefaultable): pass
class orthologs(VDefaultable): pass
class similarity(VDefaultable): pass
class orange_book(VDefaultable): pass
