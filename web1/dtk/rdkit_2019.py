"""
This is a shim for making it easier to import the newer rdkit version even
if you are executing a script with the pythton2 web1 env activated.

We need an old rdkit in our pythonpath for the python2 web1 env.  But if we
want to run a script from the new env without fully switching over, it will
make us import the wrong rdkit version.

To fix that, we find the relevant path entries and remove them.

Once we're switching over fully, we can remove this and all usages of it.
"""

import six

if six.PY3:
    import sys
    sys.path = [x for x in sys.path if not '2xar/opt/rdkit' in x]
    import rdkit

    import os
    if 'RDBASE' in os.environ:
        del os.environ['RDBASE']
