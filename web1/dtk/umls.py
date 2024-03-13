from __future__ import print_function
import six
################################################################################
# utilities for accessing UMLS
################################################################################

def umls_path(fn):
    from path_helper import PathHelper
    import os
    return os.path.join(PathHelper.cfg('UMLS_directory'),fn)

class MRFILES:
    '''A column name lookup facility for UMLS tables.

    The 'get' class method takes a table name and returns an object
    with an attribute for each column name, returning the zero-based
    index of that column in the table.
    '''
    def __init__(self):
        class Dummy: pass
        # bootstrap with partial column index for MRFILES
        ci = Dummy()
        ci.FIL = 0
        ci.FMT = 2
        # scan MRFILES building column index objects for each file described
        d = dict()
        from dtk.files import get_file_records
        for fields in get_file_records(
                umls_path('MRFILES.RRF'),
                parse_type='psv',
                ):
            c = Dummy()
            for i,name in enumerate(fields[ci.FMT].split(',')):
                setattr(c,name,i)
            d[fields[ci.FIL]] = c
        # save objects
        self.file_cols = d
    _singleton=None
    @classmethod
    def get(cls,table):
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton.file_cols[table]

def sty_filter(typeset):
    ci = MRFILES.get('MRSTY.RRF')
    result = set()
    from dtk.files import get_file_records
    for fields in get_file_records(
            umls_path('MRSTY.RRF'),
            parse_type='psv',
            ):
        if fields[ci.STY] in typeset:
            result.add(fields[ci.CUI])
    return result

def get_meddra_cui2name():
    '''Return a dict mapping each MedDRA CUI to its preferred name.
    '''
    ci = MRFILES.get('MRCONSO.RRF')
    from dtk.files import get_file_records
    from dtk.data import MultiMap
    # extract all MedDRA preferred terms; save the term status w/ each name
    mm = MultiMap(
        (fields[ci.CUI],(fields[ci.TS],fields[ci.STR].lower()))
        for fields in get_file_records(
                umls_path('MRCONSO.RRF'),
                parse_type='psv',
                grep=['MDR.PT'],
                )
        if fields[ci.TTY]=='PT' and fields[ci.SAB]=='MDR'
        )
    # get a unique name for each CUI
    result = dict()
    for cui,s in six.iteritems(mm.fwd_map()):
        # first check for unique name
        if len(s) == 1:
            result[cui] = list(s)[0][1]
            continue
        # next, check for only one labeled 'Preferred'
        s2=set([x for x in s if x[0] == 'P'])
        if len(s2) == 1:
            result[cui] = list(s2)[0][1]
            continue
        # choose one at random 
        # XXX This is a relatively small set of about 100 names. We
        # XXX could instead provide a preferred mapping, or some more
        # XXX reproducible way of selecting these. Some of these are
        # XXX used heavily in Nash:
        # XXX 10855   C0037274    ['skin disorder','dermatosis']
        # XXX 9446    C0035309    ['retinopathy','retinal disorder']
        # XXX 5531    C0022658    ['nephropathy','renal disorder']
        result[cui] = list(s)[0][1]
    return result

class Name2CUI:
    def __init__(self,sab=None):
        ci = MRFILES.get('MRCONSO.RRF')
        from dtk.files import get_file_records
        from dtk.data import MultiMap
        mm = MultiMap(
                (fields[ci.STR].lower(),fields[ci.CUI])
                for fields in get_file_records(
                        umls_path('MRCONSO.RRF'),
                        parse_type='psv',
                        )
                if sab is None or fields[ci.SAB] == sab
                )
        self.fwd = mm.fwd_map()
        self.rev = mm.rev_map()
        print(len(self.fwd),'distinct names loaded')
        print(len(self.rev),'CUIs loaded')

class CUIREL:
    def __init__(self,rel,rela=None):
        ci = MRFILES.get('MRREL.RRF')
        from dtk.files import get_file_records
        from dtk.data import MultiMap
        if rela is None:
            ok = lambda x: x[ci.REL] == rel
        else:
            ok = lambda x: x[ci.REL] == rel and x[ci.RELA] == rela
        mm = MultiMap(
                (fields[ci.CUI1],fields[ci.CUI2])
                for fields in get_file_records(
                        umls_path('MRREL.RRF'),
                        parse_type='psv',
                        )
                if ok(fields) and fields[ci.CUI1] != fields[ci.CUI2]
                )
        self.fwd = mm.fwd_map()
        self.rev = mm.rev_map()
        print(len(self.fwd),'CUI1s loaded')
        print(len(self.rev),'CUI2s loaded')

from dtk.lazy_loader import LazyLoader
class UMLSIndiMapper(LazyLoader):
    # All the parts could be constructed in an __init__ method, instead
    # of using LazyLoader, but the latter allows partial replacement
    # data to be plugged dynamically for testing.
    def translate(self,name):
        '''Given a string, return CUI and mapping method.
        '''
        # check for an exact match
        try:
            cui = self.targ2cui[name]
            return (cui,'EXACT')
        except KeyError:
            pass
        # get all CUIs that match the name, and see if any are MedDRA
        try:
            cui_set = self.n2c.fwd[name]
        except KeyError:
            return (None,'NO_CUI')
        overlap = cui_set & self.targ_cui_set
        if overlap:
            return self._prep_return(overlap,'ALIAS')
        # Search up the hierarchy one level at a time, to see if
        # any of the CUIs have a MedDRA parent.
        # - the big challenge here is we typically get multiple parents back
        #   at every node, so we need a way to prioritize them
        # - to support the evolution of this priority scheme, whenever we
        #   return a CHILD how-code, we also set a last_child_history data
        #   member that the client code can use to log mapping detail;
        #   currently this holds a tuple like:
        #   (original string, (intermediate CUIs,...), final CUI, final string)
        # - the initial priority scheme is to sort the parent CUIs by the
        #   number of children; this assumes that a link directly to a very
        #   broad concept will have lots of children, and therefore is less
        #   good than a narrower concept with fewer children. As we move
        #   multiple steps up the hierarchy, we multiply these child counts
        #   together (assuming they're rough approximations of the overall
        #   branching factor at each level).
        # - to avoid lots of depth-first vs. breadth-first bookkeeping, we
        #   just check the best single candidate in the list, returning
        #   the target mapping if it's in the target set. If not, we replace
        #   it with all its parents, suitably scored, and re-sort the list.
        # XXX Although the above produces generally reasonable results, it
        # XXX sometimes gets fooled. "age-related cataract" (C0036646) seems
        # XXX to prefer "lens disorder" (C0023308) to "cataract" (C0086543).
        # XXX The trace_down() code in umls/utility.py seems to be pretty
        # XXX good at building MedDRA hierarchies. Maybe there's some way
        # XXX to pre-build these, and assign CUIs a level in advance, and
        # XXX use that rather than child count to determine which mapping
        # XXX to use?
        # XXX
        # XXX There are the same number of PAR and CHD records, which implies
        # XXX they're symmetrical, so that's not the difference between this
        # XXX code and trace_down(). About 2/3 of the PAR/CHD records have
        # XXX an 'isa' label, and the others are blank; maybe we want to ignore
        # XXX one or the other. (But just restricting to isa records without
        # XXX other changes causes this to deteriorate, maybe because it breaks
        # XXX the assumption that having more children means you're a broader
        # XXX concept.)
        # XXX
        # XXX The final solution may be to start over:
        # XXX - restrict PAR records to the ones with 'inverse_isa' labels
        # XXX - use the now-expanded list of MedDRA targets
        # XXX - try a preliminary run checking just immediate parents to
        # XXX   to determine if this handles most of the cases
        # XXX - build out conflict resolution and multi-level ancestry in
        # XXX   a restricted way based on what exceptions arise
        seen = set()
        cui_queue = [(1,x,[]) for x in cui_set]
        while cui_queue:
            cui_queue.sort()
            priority,cui,history = cui_queue.pop(0)
            if cui in seen:
                continue
            if cui in self.targ_cui_set:
                self.last_child_history = history
                return (cui,'CHILD')
            seen.add(cui)
            cui_queue += [
                    (
                            priority*len(self.parents.rev[pcui]),
                            pcui,
                            history+[cui],
                            )
                    for pcui in self.parents.fwd.get(cui,[])
                    ]
        return (None,'NO_MEDDRA')
    def _prep_return(self,s,reason):
        # XXX we assume all CUIs in a set are equivalent, and return one
        # XXX at random; somehow prioritize instead?
        return (list(s)[0],reason)
    def _targ_mm_loader(self):
        # Extract a MultiMap from cui to name for all target CUIs.
        # There may be multiple names per cui, but only one CUI per name.
        ci = MRFILES.get('MRCONSO.RRF')
        # XXX The targets loaded here are the major control over the
        # XXX behavior of the mapper. The selection of term types passed
        # XXX to the sty_filter gives a reasonable starting point for
        # XXX trying to get disease names rather than symptom names.
        # XXX This may need to be adjusted; some things that are
        # XXX "Signs and Symptoms" may be preferable to a very
        # XXX high-level disease term (e.g. diarrhea may be preferred
        # XXX to gastrointestinal disorder, chronic pain to nervous
        # XXX system disorder).
        # XXX
        # XXX Also, restricting to MedDRA preferred terms may not always
        # XXX be ideal. Some findings map over at a higher level (seizure
        # XXX maps to C0852425, an HT).
        # XXX
        # XXX Finally, we may want to drill into LLTs to get more resolution
        # XXX on a particular disease for specific workspaces, for example
        # XXX keeping all the subtypes for Macular Degeneration. This might
        # XXX imply restructuring this slightly so that it's easier to build
        # XXX and pass in a custom set of CUI targets.
        if True:
            # disease-only version
            diseases = sty_filter(set([
                    'Pathologic Function',
                    'Experimental Model of Disease',
                    'Cell or Molecular Dysfunction',
                    'Disease or Syndrome',
                    'Mental or Behavioral Dysfunction',
                    'Neoplastic Process',
                    'Anatomical Abnormality',
                    'Acquired Abnormality',
                    'Congenital Abnormality',
                    ]))
            def is_ok(rec):
                return all([
                        rec[ci.TTY]=='PT',
                        rec[ci.SAB]=='MDR',
                        rec[ci.CUI] in diseases,
                        ])
        else:
            # all MedDRA preferred terms
            def is_ok(rec):
                return all([
                        rec[ci.TTY]=='PT',
                        rec[ci.SAB]=='MDR',
                        ])
        from dtk.files import get_file_records
        from dtk.data import MultiMap
        return MultiMap(
            (fields[ci.CUI],fields[ci.STR].lower())
            for fields in get_file_records(
                    umls_path('MRCONSO.RRF'),
                    parse_type='psv',
                    )
            if is_ok(fields)
            )
    def _targ2cui_loader(self):
        def mapping_error(s):
            raise RuntimeError('ambiguous mapping: '+repr(s))
        from dtk.data import MultiMap
        return dict(MultiMap.flatten(
                self.targ_mm.rev_map(),
                selector=mapping_error,
                ))
    def _targ_cui_set_loader(self):
        return set(self.targ_mm.fwd_map().keys())
    def _n2c_loader(self):
        return Name2CUI()
    def _parents_loader(self):
        return CUIREL('PAR')

