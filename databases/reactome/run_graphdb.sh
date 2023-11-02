#!/bin/bash

set -e
set -o pipefail

GRAPHDB_TGZ="$1"
RCT_DIR="$2"

NEO4J_VER=4.4
# neo4j isn't automatically updated to the latest version. Instead, we
# should use the version recommended by reactome here:
#  https://reactome.org/dev/graph-database#GetStarted
#
# Since we now re-build the container every time we run this, changes
# to the version number above will happen automatically.
#
# The underlying docker resources are published by neo4j itself,
# and described here:
#  https://hub.docker.com/_/neo4j/

# start by forcing a clean slate
echo "Forcing clean rebuild"
sudo docker rm -f neo4j-reactome || true # ok if it doesn't exist
sudo rm -rf $RCT_DIR || true # ok if it doesn't exist

RCT_DATA=$RCT_DIR/dataset
echo "Installing reactome dataset"
mkdir -p $RCT_DATA/databases
tar xf $GRAPHDB_TGZ -C $RCT_DATA/databases

RCT_CONF=$RCT_DIR/conf
echo "Installing reactome configuration"
mkdir -p $RCT_CONF
cp neo4j.conf $RCT_CONF

echo "Creating docker container"
# Run neo4j's docker image.
sudo docker run \
        --publish 7474:7474 \
        --publish 7687:7687 \
        --volume=${RCT_DIR}/dataset:/data \
        --volume=${RCT_DIR}/conf:/var/lib/neo4j/conf \
        --env NEO4J_AUTH=none \
        -d \
        --rm \
        --name 'neo4j-reactome' \
        neo4j:${NEO4J_VER}

# Start up the container if it's not running.
# This shouldn't be needed, since the docker run now always happens.
#if ! sudo docker ps | grep -q 'neo4j-reactome'; then
#    sudo docker start neo4j-reactome
#fi

# NOTE: If the following appears to hang, it's probably the result of the
# container failing to start. Run the above 'sudo docker run' command by hand,
# leaving out the --rm. This will leave the container around after the
# failure, so you can use 'sudo docker logs' to examine its log files.
#
# An out-of-disk condition can leave the container volumes in a bad state.
# After arranging for more space, you should remove $RCT_DIR to force a
# rebuild of the container volumes.
#
# It also may be helpful to connect to the neo4j server:
#  http://localhost:7474/browser/
# (For this to work, you need to forward both 7474 and 7687 back to the host
# running the browser, e.g.
#  ssh etl -L 7474:localhost:7474 -L 7687:localhost:7687
# )
# From there, you can get some info on a database's status by doing:
#   :use system
#   show databases
#
# If the database looks ok, but this script hangs in the loop below, you
# can get useful information by executing the short polling script in
# an interactive python session.
#
# As of 2022-12-02, this wasn't working, with the python code throwing
# neo4j.exceptions.TransientError: {code: Neo.TransientError.Database.DatabaseUnavailable} {message: Database 'graph.db' is unavailable.}
# As a workaround, I:
# - placed an 'echo' in front of the sudo docker run command
# - ran a make build
# - when the loop below hung, I copied the echoed docker run command and
#   pasted it into another window. That caused the container to start
#   correctly, and the Makefile to proceed. (Actually, the first attempt
#   failed, but after doing a 'sudo docker rm -f neo4j-reactome' and then
#   re-pasting the command, it worked.)
# I'm not sure what the exact issue is (some timing thing?) but we can
# try again next time.

echo "Waiting for server to come up"
while ! python -c "from dtk.reactome import Reactome; r = Reactome(); r.get_pathways_with_prot('P29275')" >& /dev/null; do
    sleep 1
done
echo "Done"
