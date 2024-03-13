# Aggregation methods typically have at least one setting for each input
# score. A challenge is naming these settings so they can be meaningfully
# compared across runs (using the source job id as part of the setting name,
# for example, isn't helpful). A class called the JobPrefixManager
# attempted to solve this problem, but it was complex and brittle, and
# didn't work very well.
#
# The introduction of job role codes provides a tool to more easily solve
# this problem. The SourceRoleMapper is a replacement for the JPM that
# uses job roles, and attempts to solve several other JPM issues:
# - JPM imposed its own structure on all the parameters it managed. SRM
#   doesn't do that -- it just provides a role-specific string that the plugin
#   can embed into settings names.
# - JPM would fall back to job ids if the jobnames of the inputs were not
#   unique. SRM falls back to custom user labels, and has a protocol for
#   requiring the user to manually correct any ambiguities.
# - JPM relied on job names, which aren't portable across workspaces. SRM
#   uses roles, which simplifies comparing settings across workspaces.
# - JPM defined new per-source objects; SRM uses the same objects as the
#   SourceList class
class SourceRoleMapper:
    @staticmethod
    def safe_name(s):
        import string
        output=''
        for ch in s:
            if ch in string.ascii_letters:
                output+=ch
            elif output and ch in string.digits:
                output+=ch
        return output
    ##########
    # role-based codes per source
    ##########
    def __init__(self,sources):
        # by default, use job roles as the source codes
        self._mappings = [
                [src.bji().job.role, src]
                for src in sources.sources()
                ]
        # in case of conflict, try to generate source codes from
        # custom user labels
        for group in self._non_unique():
            for item in group:
                if item[1].label() != item[1].default_label():
                    item[0] = self.safe_name(item[1].label())
    def sources(self):
        '''Return list of (role,ScoreSource) pairs.'''
        return self._mappings
    def source(self,code):
        for item in self._mappings:
            if item[0] == code:
                return item[1]
        # Usually, if source codes were generated from labels rather
        # than roles, the _mappings file will be built to reflect that
        # due to the uniqueness check in the ctor. But since WZS only
        # stores inputs with non-zero weights, it's possible only one
        # of the label-derived sources will be stored, and its role will
        # appear unique.
        #
        # To work around this, if the initial search above fails, try
        # again matching against the label-derived codes.
        for item in self._mappings:
            if self.safe_name(item[1].label()) == code:
                return item[1]
    ##########
    # metadata storage and retrieval
    ##########
    meta_prefix='srm_'
    def metadata(self,code):
        for item in self._mappings:
            if item[0] == code:
                stem=self.meta_prefix+code+'_'
                return {
                        stem+'label':item[1].label(),
                        stem+'srcjob':item[1].bji().job.id,
                        }
        return {}
    @classmethod
    def get_source_job_ids_from_settings(cls,p):
        job_ids = cls._get_source_job_ids_from_settings(p,cls.meta_prefix)
        if not job_ids:
            # try recovering legacy JPM sources
            job_ids = cls._get_source_job_ids_from_settings(p,'jpm_')
        return job_ids
    @classmethod
    def _get_source_job_ids_from_settings(cls,p,prefix):
        job_ids = set()
        for k,v in p.items():
            if not k.startswith(prefix):
                continue
            if not k.endswith('_srcjob'):
                continue
            job_ids.add(int(v))
        return job_ids
    @classmethod
    def get_source_list_from_settings(cls,ws,p):
        result = cls._get_source_list_from_settings(ws,p,cls.meta_prefix)
        if result.sources():
            return result
        # try recovering legacy JPM sources
        # XXX we currently don't support recovering JPM settings, just sources
        return cls._get_source_list_from_settings(ws,p,'jpm_')
    @classmethod
    def _get_source_list_from_settings(cls,ws,p,prefix):
        by_source = {}
        for k,v in p.items():
            if not k.startswith(prefix):
                continue
            parts = k.split('_')
            src_code = '_'.join(parts[1:-1])
            d = by_source.setdefault(src_code,{})
            d[parts[-1]] = v
        from dtk.scores import SourceList
        result = SourceList(ws)
        config_string = result.rec_sep.join([
                str(v['srcjob'])+result.field_sep+v['label']
                for v in by_source.values()
                # tolerate labels without srcjobs
                if 'srcjob' in v
                ])
        if config_string:
            result.load_from_string(config_string)
        return result
    @classmethod
    def build_from_settings(cls,ws,p):
        result = cls(cls.get_source_list_from_settings(ws,p))
        # There's one special case not handled by the above.
        # If a source list had multiple sources with the same role,
        # it triggers non-unique handling in the ctor that assigns
        # aliases. But if only one of those non-unique sources gets
        # saved in the settings, it now appears unique, and so uses
        # the standard role-based labels. So, make one more pass to
        # make sure the labels from the settings always get used,
        # unique or not.
        job2label = {}
        for k,v in p.items():
            if not k.startswith(cls.meta_prefix):
                continue
            parts=k.split('_')
            if parts[-1] != 'srcjob':
                continue
            job2label[int(v)] = '_'.join(parts[1:-1])
        for item in result._mappings:
            item[0] = job2label[item[1].bji().job.id]
        return result
    ##########
    # src_code uniqueness testing
    ##########
    def _non_unique(self):
        # check for non-unique role
        from collections import Counter
        ctr=Counter([x[0] for x in self._mappings])
        return [
                [x for x in self._mappings if x[0] == key]
                for key,count in ctr.items()
                if count > 1
                ]
    def non_unique_warning(self):
        detail = [
                '%s: (jobs %s)'% (
                        group[0][1].label(),
                        ', '.join([str(x[1].bji().job.id) for x in group]),
                        )
                for group in self._non_unique()
                ]
        if detail:
            from dtk.html import join,tag_wrap,alert
            from django.utils.safestring import mark_safe
            detail += [
                    "",
                    "(Note that spaces, punctuation and leading digits"
                    " in labels don't count for disambiguation.)",
                    ]
            return join(
                tag_wrap('h3',alert('WARNING: Ambiguous source jobs')),
                tag_wrap('b','please re-label before running:'),
                *detail,
                sep=mark_safe('<br>')
                )


def jr_interpreter(jr, include_jr=True):
    s = jr.lower()
    parts = s.split('_')
    to_return = []
    # input data
    if 'otarg' in parts:
        to_return.append('OT')
        if s.startswith('animalmodel'):
            to_return.append('animal model')
        elif s.startswith('geneticassociation'):
            to_return.append('genetics')
        elif s.startswith('literature'):
            to_return.append('literature')
        elif s.startswith('rnaexpression'):
            to_return.append('RNA expr')
        elif s.startswith('knowndrug'):
            to_return.append('known drug')
        elif s.startswith('somaticmutation'):
            to_return.append('somatic mut')
    elif s.startswith('faers_faers'):
        to_return.append('FAERS')
        if 'capp' in parts:
            to_return.append('+co-morb genetics')
    elif s.startswith('tcgamut'):
        to_return.append('TCGA tumor mut')
    elif s.startswith('cc'):
        to_return.append('Gene expr')
    elif s.startswith('mirna'):
        to_return.append('miRNA expr')
    elif s.startswith('esga'):
        to_return += ['GWAS', 'PPI', 'network scored']
    elif (s.startswith('gwasig') or s.startswith('gpath')
          ):
        to_return.append('GWAS')
    elif s.startswith('dgns_dgn'):
        to_return.append('DisGeNet')

    if ('sigdif' in parts or
        s.endswith('_capis') or
        s.endswith('_indirect') or
        s.endswith('gis') or
        s.endswith('indirectbgnormed')
        ):
        to_return.append('+PPI')

    # scoring
    if s.endswith('depend_psscoremax'):
        to_return.append('| pathway scored')
    elif s.endswith('codes_codesmax'):
        to_return.append('| signature scored')
    elif ('capp' in parts or 'path' in parts or 'gpath' in parts):
        to_return.append('| pathsum scored')
        if 'gpbr' in parts:
            to_return.append('background removed')
    elif 'defus' in parts:
        to_return.append('| molecule sim scored by')
        if s.endswith('indigoscore') or s.endswith('rdkitscore'):
            to_return.append('molecule structure')
        elif s.endswith('indjacscore') or s.endswith('prmaxscore'):
            to_return.append('target overlap')

    if include_jr:
        to_return.append(f'({jr})')

    return " ".join(to_return)
