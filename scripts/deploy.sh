#!/bin/bash

# This script walks through the weekly sprint deploy process.
# It still requires someone at the keyboard to type in usernames.
# It also has no support for resumption if something goes wrong or
# anything fancy like that.
#
# This should be run from a machine that can ssh to platform and worker.
# It expects to be able to ssh to them as "platform" and "worker", so configure
# your .ssh/config appropriately.
#
# (If you are ssh'd with ssh-agent and forwarding enabled, and your ssh key has git read access,
# this will not require you to enter a password when it does the git fetch during deploy)
#
# Usage: deploy.sh <MCH> {<SPRINT_ID>}
#
# MCH is the machine you're deploying to (platform, worker)
# SPRINT_ID is the optional sprint number (e.g. 326). If not supplied, it
# defaults to the one after the one in the git tree you're deploying from.
# If you just tagged the sprint on github and haven't fetched it, this will
# be correct, but the manual override is available for special circumstances.
#
# The script will ask for confirmation before actually doing anything.
#
# Example deployment:
#   ./deploy.sh worker
#   ./deploy.sh platform

set -e
set -x

MCH=$1
NUM=$2
if [ -z "$MCH" ]; then
    echo "Please supply a machine name"
    exit 1
fi
if [ -z "$NUM" ]; then
    PREV_SPRINT_ID="$(git describe --tags | cut -f1 -d'-')"
    SPRINT_ID=$(python -c "x='${PREV_SPRINT_ID}'; num=x.split('sprint')[1]; print(int(num)+1);")
    NUM=$SPRINT_ID
fi

AWS_OP="$HOME/2xar/twoxar-demo/web1/aws_op.py"

read -p "Deploying sprint $NUM to $MCH, continue? [y/N]" CONTINUE
if [ "$CONTINUE" != "y" ]; then
    echo "Stopping"
    exit 0
fi


echo "*** Snapshotting and starting machines"
${AWS_OP} -m ${MCH} deploy${NUM}
${AWS_OP} -m ${MCH} start

read -p "Machine is imaged and has been restarted.  Do any manual steps then hit enter to continue" CONTINUE

echo "*** Updating $MCH"
ssh -A -tt $MCH "cd ~/2xar/twoxar-demo && git fetch 'git@github.com:twoXAR/twoxar-demo.git' sprint${NUM}:sprint${NUM}  && git checkout sprint${NUM} && install/install.sh"
echo "*** Done updating $MCH"
