"""
WSGI config for web1 project.

This module contains the WSGI application used by Django's development server
and any production WSGI deployments. It should expose a module-level variable
named ``application``. Django's ``runserver`` and ``runfcgi`` commands discover
this application via the ``WSGI_APPLICATION`` setting.

Usually you will have the standard Django WSGI application here, but it also
might make sense to replace the whole Django WSGI application with a custom one
that later delegates to the Django one. For example, you could introduce WSGI
middleware here, or combine a Django application with an application of another
framework.

"""
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")

# Environment setup (to keep /etc/apache2/envvars unmodified).
# These avoid some R errors due to partial/incorrect setups leaking
# through to background jobs somehow.
# Previous setup adding /home/ubuntu/2xar/opt/conda/envs/py3web1/lib to
# LD_LIBRARY_PATH appears to be unneccesary -- lsof on the apache process
# confirms that it finds .so files from the conda environment.
oldpath=os.environ['PATH']
os.environ['PATH'] = ':'.join([
        '/home/ubuntu/2xar/opt/conda/envs/py3web1/bin',
        '/home/ubuntu/2xar/opt/conda/envs/r-env/bin',
        oldpath,
        ])
os.environ['LANG'] = 'C.UTF-8'
# HOME isn't set by default, but this prevents git from finding a needed
# override in /var/www/.gitconfig that allows www-data to access the
# git repo (which it doesn't own) in order to retrieve the release tag.
# The original version of this fix did this unconditionally, but this
# file also gets executed via runserver in other environments.
if 'HOME' not in os.environ:
    os.environ['HOME'] = '/var/www'

# These show up in /var/log/apache2/errors.log
import sys
def eprint(*args):
    print(*args,file=sys.stderr)
eprint('In wsgi.py')
eprint('UID',os.getuid())
eprint('EUID',os.geteuid())
for envvar in (
        'LANG',
        'LD_LIBRARY_PATH',
        'PATH',
        'HOME',
        ):
    eprint(envvar,'=',os.environ.get(envvar))
eprint('sys.path =',sys.path)
# rdkit has fairly intricate dependencies, if we can use it then we are probably
# setup properly in our virtual environment.
import dtk.rdkit_2019
from rdkit import Chem
eprint("RDKit Check:", Chem.MolToInchi(Chem.MolFromSmiles("COO")))
from dtk.standardize_mol import standardize_mol
eprint("RDKit Check 2:", Chem.MolToInchi(standardize_mol(Chem.MolFromSmiles("COO"))))


# This application object is used by any WSGI server configured to use this
# file. This includes Django's development server, if the WSGI_APPLICATION
# setting points here.
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Apply WSGI middleware here.
# from helloworld.wsgi import HelloWorldApplication
# application = HelloWorldApplication(application)
