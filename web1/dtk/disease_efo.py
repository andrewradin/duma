# split this out so that the Disease object could be loaded elsewhere.
# Pickle bakes in some attributes when the class and the pickling are in the same file
# see https://stackoverflow.com/questions/40287657/load-pickled-object-in-different-file-attribute-error/40288996
class Disease:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.id = kwargs.get('id', None)
        self.synonyms = kwargs.get('synonyms', [])
        self.parents = kwargs.get('parents', [])
        self.children = []


def load_efo_otarg_graph(efo_choice, sep=':'):
    """
    There are inconsistencies in whether EFO keys are _ or : separated.
    The datafile uses ':', but other things like our opentargets datafiles use _.
    Use the 'sep' argument to convert.
    """
    from dtk.s3_cache import S3File
    from dtk.files import get_file_records
    s3f = S3File.get_versioned('efo', efo_choice, role='otarg_hier')
    s3f.fetch()

    def fix_id(x):
        if sep != ':':
            x = x.replace(':', sep)
        return x

    import networkx as nx
    g = nx.DiGraph()
    for parent, child in get_file_records(s3f.path(), keep_header=False):
        g.add_edge(fix_id(parent), fix_id(child))
    
    return g