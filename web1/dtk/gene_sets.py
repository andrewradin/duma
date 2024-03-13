
import logging
logger = logging.getLogger(__name__)


LEGACY_FN = 'annotated.pathways_reactome.reactome.uniprot.gmt'
ROOT_ID = '[root]'

def gene_set_choices():
    from dtk.files import VersionedFileName
    from dtk.s3_cache import S3Bucket
    s3b = S3Bucket('pathways')
    choices = VersionedFileName.get_choices(file_class='pathways',
            paths=s3b.list(cache_ok=True))

    choices.append([LEGACY_FN, 'annotated.pathways_reactome (legacy)'])
    return choices


def legacy_genesets_file(fn):
    from dtk.s3_cache import S3Bucket, S3File
    b = S3Bucket('glee')
    s3f=S3File(b,fn)
    return s3f

def legacy_genesets_converter_file():
    from dtk.s3_cache import S3Bucket, S3File
    b = S3Bucket('glee')
    s3f=S3File(b, 'reactome_names_to_id.tsv')
    s3f.fetch()
    return s3f

def legacy_genesets_name_to_id():
    s3f = legacy_genesets_converter_file()
    with open(s3f.path()) as f:
        name2id = {}
        for line in f:
            name, id = line.strip().split('\t')
            name2id[name] = id
        return name2id

def get_gene_set_file(choice):
    if choice == LEGACY_FN:
        s3f = legacy_genesets_file(LEGACY_FN)
    else:
        from dtk.files import VersionedFileName
        from dtk.s3_cache import S3Bucket, S3File
        s3b = S3Bucket('pathways')
        selected = VersionedFileName.get_matching_path(
                file_class='pathways',
                paths=s3b.list(cache_ok=True),
                choice=choice,
                role='genesets',
                )
        s3f = S3File(s3b, selected)
    s3f.fetch()
    return s3f


def get_pathway_prot_mm(gene_set_file):
    pairs = []
    from dtk.files import get_file_records
    for c in get_file_records(gene_set_file, parse_type='tsv', keep_header=None):
        if c[1]:
            name = c[0]
            prots = c[1].split(",")
            pairs.extend((name, prot) for prot in prots)

    from dtk.data import MultiMap
    return MultiMap(pairs)


def get_pathways_for_prots(gene_set_choice, prots):
    pairs = []
    from dtk.files import get_file_records
    fn = get_pathway_file(gene_set_choice, 'gene_to_pathway')
    for c in get_file_records(fn, keep_header=False, select=[prots, 'prot']):
        pairs.append(c)
    
    from dtk.data import MultiMap
    return MultiMap(pairs)

def get_prots_for_pathways(gene_set_choice, pathways):
    pairs = []
    from dtk.files import get_file_records
    fn = get_pathway_file(gene_set_choice, 'gene_to_pathway')
    for c in get_file_records(fn, keep_header=False, select=[pathways, 'pw']):
        pairs.append((c[1], c[0]))
    
    from dtk.data import MultiMap
    return MultiMap(pairs)


def get_pathway_file(choice, role):
    if choice is None:
        from browse.default_settings import GeneSets 
        choice = GeneSets.latest_version()
    if choice == LEGACY_FN:
        if role == 'genesets':
            s3f = legacy_genesets_file(LEGACY_FN)
        else:
            from dtk.files import NoMatchesError
            raise NoMatchesError(f"No hierarchy file for legacy {choice}")
    else:
        from dtk.files import VersionedFileName
        from dtk.s3_cache import S3Bucket, S3File
        s3b = S3Bucket('pathways')
        selected = VersionedFileName.get_matching_path(
                file_class='pathways',
                paths=s3b.list(cache_ok=True),
                choice=choice,
                role=role,
                )
        s3f = S3File(s3b, selected)
    s3f.fetch()
    return s3f.path()

def get_pathway_id_name_map(choice=None):
    from dtk.files import get_file_records
    hier_file = get_pathway_file(choice, 'hierarchy')
    out = {}
    for rec in get_file_records(hier_file, keep_header=False):
        id, name = rec[:2]
        out[id] = name
    return out

def get_pathway_data(choice=None):
    '''Return a triple (protsets, pathways_data, hier).

    where:
    protsets = {set_id:{uniprot,...},...}
    pathways_data = {set_id:desc,...}
    hier = {set_id or ROOT_ID:[set_id,...],...}
    and
    desc = {'id':set_id,'name':set_name,'type':set_type,'hasDiagram':bool}
    set_type is one of:
        pathway
        biological_process
        cellular_component
        molecular_function
        event
    '''
    from dtk.files import get_file_records
    from dtk.data import MultiMap
    gs_file = get_pathway_file(choice, 'genesets')
    hier_file = get_pathway_file(choice, 'hierarchy')

    children = []
    protsets = {k:list(v) for k, v in get_pathway_prot_mm(gs_file).fwd_map().items()}


    pathways_data = {}

    is_child = set()

    # Expect header ['id', 'name', 'type', 'description', 'children', 'has_diagram']
    header = None
    for rec in get_file_records(hier_file, keep_header=True):
        if header is None:
            header = rec
            continue
        entry = dict(zip(header, rec))
        # Rename and cast to bool
        entry['hasDiagram'] = entry.pop('has_diagram') == '1'
        # Parse out children.
        entry_children = entry.pop('children')
        if entry_children:
            for child in entry_children.split('|'):
                # Filter out empty children from the hierarchy.
                if child in protsets:
                    children.append((entry['id'], child))
                    is_child.add(child)
        
        # Filter out empty protsets from the hierarchy.
        if entry['id'] not in protsets:
            continue
        pathways_data[entry['id']] = entry

    roots = list(pathways_data.keys() - is_child)
    toplevel_pairs = [(ROOT_ID, p) for p in roots]
    children.extend(toplevel_pairs)
    hier = {k:list(v) for k, v in MultiMap(children).fwd_map().items()}

    return protsets, pathways_data, hier


def get_pathway_sets(set_ids, pathways_choice):
    if not set_ids:
        return []
    from dtk.files import NoMatchesError
    try:
        protsets, pathways_data, hier = get_pathway_data(pathways_choice)
    except NoMatchesError:
        logger.warning(f"Couldn't find data for {pathways_choice}, probably old format")
        return [set() for _ in set_ids]

    def get_pathway_set(set_id):
        key, value = set_id.split('=')
        if key == 'type':
            return [x['id'] for x in pathways_data.values() if x['type'] == value]
        elif key == 'ancestor':
            out = set()
            def visit(node, isDesc):
                if node == value:
                    isDesc = True
                if isDesc:
                    out.add(node)
                for child in hier.get(node, []):
                    visit(child, isDesc)
            
            visit(ROOT_ID, False)
            return list(out)
        elif key == 'custom':
            if value == 'single_prot':
                return [k for k, v in protsets.items() if len(v) <= 1]
        raise Exception(f"Unhandled pathway set type {set_id}")

    return [get_pathway_set(set_id) for set_id in set_ids]


def pathway_exclude_choices():
    return (
        # Better to do CC as type rather than ancestor, there are some
        # non-cellular component things that trace into the CC hierarchy via
        # 'occurs_in' relations (e.g. https://www.ebi.ac.uk/QuickGO/term/GO:0046907
        ('type=cellular_component', 'Cellular Component'),
        ('ancestor=R-HSA-162582', 'Signal Transduction'),
        ('custom=single_prot', 'Single Prot Pathways'),
        ('type=event', 'Reactome Reactions'),
    )


def make_dedupe_map(protsets, hierarchy, siblings_only, id=ROOT_ID, out_map=None, protset_map=None):
    """Returns a 'dedupe' map, which for each pwID indicates the canonical pwID for that set of proteins.

    The canonical one is assigned arbitrarily to the first one that is encountered.
    """
    out_map = out_map if out_map is not None else {}
    protset_map = protset_map if protset_map is not None else {}

    prots = frozenset(protsets.get(id, []))

    if prots not in protset_map:
        protset_map[prots] = id
    out_map[id] = protset_map[prots]

    if id in hierarchy:
        child_protset_map = {} if siblings_only else protset_map
        for child_id in hierarchy[id]:
            make_dedupe_map(
                id=child_id,
                protsets=protsets,
                hierarchy=hierarchy,
                out_map=out_map,
                protset_map=child_protset_map,
                siblings_only=siblings_only,
                )
    return out_map
