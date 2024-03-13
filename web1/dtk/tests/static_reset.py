



def static_reset():
    """
    Unit tests all run in the same python instance, which can cause them to
    affect each other via class or module-level static state.
    This function resets that state wherever possible, which helps ensure that
    our tests are independent.
    """

    from dtk.lts import LtsRepo
    LtsRepo._cache = {}

    from dtk.prot_map import DpiMapping, PpiMapping
    DpiMapping._bucket = None
    PpiMapping._bucket = None

    from drugs.models import Prop
    Prop.reset()

    from dtk.entrez_utils import EClientWrapper
    EClientWrapper.classinit()
