#!/bin/bash

set -e

if [[ -f /.dockerenv ]]; then
    # For now we're assuming that the host machine is going to be running the reactome container
    # already, which means we can access it as a sibling.
    echo "Not running reactome setup, inside a container"
    exit 0
fi


SCRIPTDIR="$(dirname $0)"
OPT="$(realpath ~/2xar/opt)"

echo "Using OPT=$OPT"

# This file used to install reactome as a docker container on all machines.
# Now it's just being used to install docker itself, the diagrams for the network page, and cleanup any old instances.

VERSION=v$(cat ~/2xar/twoxar-demo/databases/reactome/last_published_version)

if [[ ! -f "${OPT}/reactome/${VERSION}" ]]; then
    sudo rm -rf "${OPT}/reactome"
    mkdir -p "${OPT}/reactome"

    DIAGRAMS_DIR=${SCRIPTDIR}/../../publish/pw
    mkdir -p ${DIAGRAMS_DIR}
    s3cmd get s3://2xar-versioned-datasets/reactome/reactome.${VERSION}.diagrams.tgz ${DIAGRAMS_DIR}/
    tar xf ${DIAGRAMS_DIR}/reactome.${VERSION}.diagrams.tgz -C ${DIAGRAMS_DIR}

    if [[ -d /home/www-data/2xar/publish ]]; then
        sudo cp -R ${DIAGRAMS_DIR} /home/www-data/2xar/publish/pw
    fi

    touch "${OPT}/reactome/${VERSION}"
fi


# Install docker.
sudo apt-get install -y docker.io

# If we have a neo4j docker instance, remove it.
# We used to leave these running all the time, but not anymore.
if sudo docker ps -a | grep -q 'neo4j-reactome'; then
    echo "Found running persistent neo4j instance, removing"
    sudo docker rm -f neo4j-reactome
fi