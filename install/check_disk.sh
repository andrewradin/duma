#!/bin/bash

set -e

if [[ -z "$1" ]]; then
    echo "Expected a username to run as"
    exit 1
fi

SCRIPT_DIR="$(dirname $0)"
# check_disk.py has a -u command to switch user, but doesn't give the permissions
# of that user - use sudo for that.
sudo -H -u $1 $SCRIPT_DIR/check_disk.py -r -u $1
