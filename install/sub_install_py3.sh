#!/bin/bash

set -e # Halt on error

# This file can be executed in two different modes.
# By default, it will install the exact checked in versions specified by the freeze files.
# If run with an argument of "refresh", it will update all packages to the latest versions
# (as allowed by their specs) and regenerate the freeze files.

SCRIPTDIR="$(dirname $0)"

# We're using conda to setup our py3 environment.
# conda makes it easier to install packages with large non-python components
#  (specifically rdkit).
CONDADIR="$HOME/2xar/opt/conda"
if [ ! -d $CONDADIR ]; then
    URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    wget $URL -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $CONDADIR
    rm /tmp/miniconda.sh
fi

# We're going to replace this later once the environment is activated, but
# for now point at the absolute path.
CONDA=${CONDADIR}/bin/conda

$CONDA config --set auto_update_conda False

# Update conda before using it.
$CONDA update --yes -n base -c defaults conda

# Create the environment.
PY3ENV="py3web1"
PY3ENV_DIR="${CONDADIR}/envs/${PY3ENV}"
if [ ! -d $PY3ENV_DIR ]; then
    $CONDA create --yes python=3.7 -n $PY3ENV
fi

# There is an environment history file that keep track of all the explicit specs that
# you've ever requested installed.  This becomes problematic for us, because all our 
# spec tend to be explicit, and we eventually create a conflict.
# Since all of our dependencies are specified explicitly, we just want to always install
# to the exact set of specs we are currently specifying.
echo "" > $PY3ENV_DIR/conda-meta/history

# Add the 'web1' directory so that it can always be cleanly imported here.
PTH_PATH="${PY3ENV_DIR}/lib/python3.7/site-packages/web1.pth"
if [ ! -f ${PTH_PATH} ]; then
    echo "${HOME}/2xar/twoxar-demo/web1" > $PTH_PATH
fi

# Setup the shell to make this environment the default, if not already done
if ! grep ">>> conda initialize >>>" ~/.bashrc; then
    echo "Adding conda setup to bashrc"
    ${CONDA} init bash
fi

if ! grep ">>> conda default env >>>" ~/.bashrc; then
    echo "Setting the default environment in bashrc"
    echo "# >>> conda default env >>>" >> ~/.bashrc
    echo "conda activate ${PY3ENV}" >> ~/.bashrc
    echo "# <<< conda default env <<<" >> ~/.bashrc
fi

# Activate our environment
source ${CONDADIR}/etc/profile.d/conda.sh
# Replace the absolute path with the shell function that is now added.
CONDA='conda'
$CONDA activate ${PY3ENV}
    

# Assume we don't have this env activated right now, just point at the right
# pip directly.
PIP3="${PY3ENV_DIR}/bin/pip"


CONDAFREEZE="${SCRIPTDIR}/freeze.conda"
PIPFREEZE="${SCRIPTDIR}/freeze.pip"

# Update pip to the frozen version before using it
PIPVER=$(cat $PIPFREEZE | grep -w "pip")
echo "Pre-installing pip '$PIPVER'"
$PIP3 install $PIPVER


# numpy 1.20 introduces a binary incompatibility, all binary python packages need to be built against a version < or >= that, but not mixed.
# https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp
# https://numpy.org/doc/stable/release/1.20.0-notes.html#size-of-np-ndarray-and-np-void-changed
#
# We will want to switch over all at once, at some later point in time when everything is switched over.
#
# However, even with explicit pinning, this is causing problems.
# See this whole thread, and the linked comment in particular - https://github.com/scikit-learn-contrib/hdbscan/issues/457#issuecomment-773671043
# (Also see https://github.com/pypa/pip/issues/9542)
#
# The solution for now appears to be to (a) pin numpy all over the place and (b) force reinstall the fisher package at the end.
#
# Note that this may also look different on fresh installs vs incrementally upgraded ones - anything with native code
# depending on the numpy C API needs to rebuild when this gets upgraded, which doesn't happen automatically for incremental updates.
#
# You'll need to rerun tests to see if it worked, these failures show up at runtime.
#
# The libraries that seem to be most affected here are:
# - fisher
# - numba
# - tensorflow
# If you can import all of those, then probably things are OK.
NUMPY_VER="numpy==1.19.*"

pre_py_pkgs=(
        # This is required for installing the version of fisher's exact we use
        # and it needs to be installed first.
        "Cython"
        ${NUMPY_VER}
        )

$PIP3 install ${pre_py_pkgs[@]} || exit 1

CONDA_CHANNELS="-c rdkit -c plotly -c chembl -c conda-forge"

if [[ "$1" == "refresh" ]]; then
    echo "** Refreshing frozen environment files"
    echo "Installing conda packages"
    # Install rdkit & plotly into our env via conda.
    conda_pkgs=(
        "python==3.7.*"
        "rdkit==2020.09.*"
        "plotly"
        # plotly static image export backend
        "python-kaleido"
        "chembl_structure_pipeline"
        # chembl library for structural similarity searching.
        "fpsim2"
        # JIT compiles numpy & python to be much faster
        "numba"
        # Make sure conda doesn't try to update numpy on us.
        ${NUMPY_VER}
    )
    $CONDA install --yes -n $PY3ENV $CONDA_CHANNELS --update-all ${conda_pkgs[@]}

    echo "Installing pip packages"
    # Everything else install via pip.
    # We should probably move most of this to conda at some point, but there are a few that aren't
    # available, and conda takes longer for now (because it attempts to solve the dependency graph).
    py_pkgs=(
            "atomicwrites"
            "awscli"
            "boto3"
            "CrossMap"
            # This is required for installing the version of fisher's exact we use
            "Cython"
            # Used as a high performance django cache.
            "diskcache"
            "django==3.2.*"
            # Login ratelimiting
            "django-axes"
            "django-csp"
            # referrer-policy is baked into django 3.0+, remove it when we upgrade.
            "django-referrer-policy"
            "django-two-factor-auth==1.10.*"
            # 0.8.* is broken for now
            # https://github.com/Bouke/django-two-factor-auth/issues/335
            "django-otp==0.7.*"
            # Querying patents from the European Patent Office
            "python-epo-ops-client"
            # We should be able to update this, but need to fix up some tests.
            "eutils==0.3.1"
            "fisher>=0.1.9"
            "future"
            # Used for patent search
            "google-cloud-bigquery"
            # Intel's faster gzip implementation
            "isal"
            # Notebooks
            "jupyter"
            # Reads arff files
            "liac-arff"
            "lxml"
            # Old plots, new stuff uses plotly.
            "matplotlib"
            "matplotlib-venn"
            "mock"
            # In py2 we use mysql-python, which is no longer maintained and has no
            # python3 support.
            # mysqlclient is a fork with python3 support.
            "mysqlclient"
            # We use this for reactome data
            "neo4j"
            "networkx"
            "nevergrad"
	    # nltk has a new dependency in 3.7 that might be transient
	    # and shouldn't really be needed. Pin for now.
	    # https://github.com/nltk/nltk/issues/3024
            "nltk==3.6"
            # Make sure pip doesn't try to update numpy on us.
            ${NUMPY_VER}
            # Parsing Obo ontology files.
            "obonet"
            # Parses excel files (.xls), used for some ETL drug collections.
            # XXX Temporarily pinned to 3.0.5 due to an issue in
            # XXX openpyxl/worksheet/merge.py; this is fixed in the source
            # XXX tree, so should be ok in 3.0.7. See
            # XXX https://twoxar.slack.com/archives/C798RS33J/p1614624295016400
            "openpyxl==3.0.5"
            "pandas"
            # Used by 2factor, apparently not included as a dependency.
            "phonenumberslite"
            # Used by some ETL
            "peewee"
            "pip"
            # Parses .owl ontology files for EFO (otarg)
            "pronto"
            # Used by memory middleware
            "psutil"
            # For reading parquet files
            "pyarrow"
            # We don't use this directly, but I believe this prevents an intermittent testing edgecase with
            # vcrpy where it fails to unpatch with the error "'tornado.curl_httpclient' has no attribute 'CurlAsyncHTTPClient'"
            "pycurl"
            # Able to add ?profile and see page load time breakdown
            "pyinstrument"
            # Testing-related packages
            "pylint"
            "pylint-django"
            "pytest"
            "pytest-coverage"
            "pytest-django"
            "pytest-html"
            # Used for selenium-based unit tests.
            "PyVirtualDisplay"
            "ray[default]"
            "ray[tune]"
            "requests"
            # Used for rich text in terminal in qa scripts.
            "rich"
            "s3cmd"
            "scipy"
	    # selenium 4.3 removes an API method we use everywhere; pin for now
	    # https://stackoverflow.com/questions/72773206/selenium-python-attributeerror-webdriver-object-has-no-attribute-find-el
            "selenium==4.2.0"
            "sh"
            "sklearn"
            "statsmodels"
            # Currently used for optimizing WZS
            "tensorflow>=2.0"
            # Easy-to-use progress bars with time estimates.
            "tqdm"
            # Transliterates unicode to similar ASCII chars
            "unidecode"
            # Store and replay HTTP requests in tests.
            "vcrpy"
            # Store compressed numpy arrays in a file.
            "zarr"
            # python zstd bindings, faster and better compression than gzip
            "zstandard"
        )
    $PIP3 install ${py_pkgs[@]} --upgrade --upgrade-strategy=eager || exit 1

    # Despite the no-pip flag, we still have to manually grep out the pypi libs.
    $CONDA list --export --no-pip | egrep -v "pypi_0$" > $CONDAFREEZE
    # pip seems to know about some packages that are installed & built by conda
    # and can have issues with them, so filter out.
    # --all makes it include pip itself
    # setuptools seems to be appends a weird subversion, just exclude for now.
    $PIP3 freeze --all | grep -v "@ file:///" | egrep -v "^mkl-"  | grep -v "setuptools" > $PIPFREEZE
fi


echo "Installing frozen conda versions"
$CONDA install -y $CONDA_CHANNELS  --file $CONDAFREEZE
echo "Installing frozen pip versions"
$PIP3 install -r $PIPFREEZE

echo "Reinstall fisher to work around numpy binary compat issues"
$PIP3 install --force fisher


TMPDIR="$(${SCRIPTDIR}/../web1/path_helper.py bigtmp)"
echo "Setting py3web1 TMPDIR as $TMPDIR"
conda env config vars set -n $PY3ENV TMPDIR=${TMPDIR}
