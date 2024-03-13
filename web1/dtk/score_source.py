import logging
logger = logging.getLogger(__name__)

class ScoreSource(object):
    def __init__(self, rolename, cds_choices):
        self.name = rolename
        self.cds_choices = cds_choices

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, o):
        return self.name == o.name

    def __repr__(self):
        return self.name

    def is_pathway(self):
        return 'psscoremax' in self.parts or 'depend' in self.parts

    @property
    def parts(self):
        try:
            return self._parts
        except AttributeError:
            self._parts = self.name.split('_')
            return self._parts

    def source_type(self):
        k = self.name
        k_parts = self.parts
        if k.startswith('cc'):
            return ['cc', 'C/C Gene Expression']
        elif k.startswith('mirna'):
            return ['mirna', 'miRNA']
        elif k.startswith('gpath'):
            return ['gpath', 'GWAS']
        elif k.startswith('gwasig'):
            return ['gwasig', 'GWAS']
        elif k.startswith('esga'):
            return ['esga', 'GWAS']
        elif k.startswith('dgn'):
            return ['dgn', 'External']
        elif k.startswith('agr'):
            return ['agr', 'External']
        elif k.startswith('misig'):
            return ['misig', 'Phenotype']
        elif k.startswith('mips'):
            return ['mips', 'Phenotype']
        elif k.startswith('tcgamut'):
            return ['tcgamut', 'TCGA mutations']
        elif k.startswith('customsig'):
            return ['other', 'CustomSig']
### This is clearly not right
### This boils down the the DEFUS scores having non-consistent names.
### Most of that is b/c we now condense DEFUS scores by similarity types.
### Specifically that is Target- or structure-based similarity.
### That condensing is done w/WZS, which means that k is user defined.
### That's fine if it's handled by the refresh flow, as it names
### the WZS the same: drtarget or drstruct
### However if the user runs DEFUS or the condensing then it might start
### with DEFUS.
        elif (any([x in k_parts
                   for x in [
                    n[0].lower() for n in self.cds_choices
                   ]
                 ])
              or k.startswith('drtarget')
              or k.startswith('drstruct')
              or k.lower().startswith('defus')
              or 'defus' in k.lower()
              or 'faers' in k.lower()
            ):
            return ['faers', 'Clinical']
        elif self.is_otarg():
            return ['otarg', 'OpenTargets']
        else:
            logger.warn('WARNING: Score source unrecognized job: %s', k)
            return ['other', k]

    # OT scores are e.g. knowndrug_otarg_sigdiff_...
    def is_otarg(self):
        return self.parts[1] == 'otarg'

    def otarg_source(self):
        assert self.is_otarg()
        return self.parts[0]
