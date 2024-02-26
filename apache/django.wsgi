import os
import sys

os.environ['DJANGO_SETTINGS_MODULE'] = 'web1.settings'

import logging
logging.info("Starting up WSGI")
logging.info("sys.path=%s", sys.path)
import rdkit
logging.info("Rdkit %s", rdkit.__version__)
logging.info("LD_LIBRARY_PATH %s\n", os.environ.get('LD_LIBRARY_PATH', 'None'))

# rdkit has fairly intricate dependencies, if we can use it then we are probably
# setup properly in our virtual environment.
from rdkit import Chem
logging.info("RDKit Check: %s\n", Chem.MolToInchi(Chem.MolFromSmiles("COO")))


from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

