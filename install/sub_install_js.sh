#!/bin/bash

set -e

# NOTE: This script sets up a conda environment, js-env, which is never
# used directly. Instead, the npm executable from the resulting environment
# is copied into py3env at the bottom of this file.
#
# This file was originally created as part of PLAT-3059, which added the
# pathway network view. Presumably the intent was to support access to
# the graph database, but that seems to run in a docker image, outside of
# our conda environment. So, this may not be necessary at all.
#
# XXX A reasonable experiment would be to remove the npm link, and verify
# XXX that both the pathway view and the reactome ETL still work. Since
# XXX there's no particular penalty in just leaving this here, and it might
# XXX be a useful example for something in the future, I'm leaving it here
# XXX with version updates frozen, rather than either:
# XXX - fixing updates, or
# XXX - removing it altogether

CONDADIR="$HOME/2xar/opt/conda"
. $CONDADIR/etc/profile.d/conda.sh

ENV="js-env"
ENV_DIR="${CONDADIR}/envs/${ENV}"

echo "Setting up ${ENV}"

if [ ! -d "${ENV_DIR}" ]; then
    conda create --yes -n $ENV 
fi

# Useful conda commands:
# conda search {-c channel} package # to see available versions
# conda env remove -n <env-name>
# conda info --envs

conda activate $ENV
# Install Node + NPM via conda.
# Conda has a big jump in available node versions (10.12.0 to 14.8.0).
# We install a package called deasync that appears to not have deasync.node
# binding files for linux for node versions after 13.x.x. So we freeze
# the version here explicitly. Since I'm not sure what this is even used
# for, it's not worth tracking down for now.
conda install --yes -n $ENV nodejs==10.13.0

echo "${ENV} upgrading npm"
# Upgrade NPM.
# Since we've frozen nodejs above, we freeze this as well
npm install -q -g npm@7.24.0

echo "${ENV} installing dependencies"
# Install our js dependencies.
cd $HOME/2xar/twoxar-demo/web1/js
npm install -q


echo "${ENV} building js file"
# Build our production JS file.
# If you're developing on JS you won't need it, but otherwise you want it.
npm run build


echo "${ENV} cleanup"
# Stick our version of npm into the py3web1 environment.
# This isn't the cleanest way to handle things, but it is more convenient to
# be able to call everything from that environment.
PY3ENV_DIR="${CONDADIR}/envs/py3web1"
rm -f ${PY3ENV_DIR}/bin/npm
ln -s ${ENV_DIR}/bin/npm ${PY3ENV_DIR}/bin/npm

# Check if we're in production; if so, we'll want to put the file there too.
if [[ -d '/home/www-data/' ]]; then
    JSDIR="/home/www-data/2xar/publish/js/"
    sudo mkdir -p $JSDIR
    sudo chown www-data:www-data $JSDIR
    sudo cp $HOME/2xar/publish/js/main.js* $JSDIR
fi

conda deactivate
