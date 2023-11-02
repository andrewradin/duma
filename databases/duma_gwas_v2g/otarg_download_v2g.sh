#!/bin/bash

set -e
set -o pipefail

OTARG_GEN_VER=$1
OUT_DIR=$2

if [[ -z "$OTARG_GEN_VER" ]]; then
    echo "Provide an opentargets genetics version"
    exit 1
fi

BASE=http://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/$OTARG_GEN_VER/v2g/
BASE_FTP=ftp://ftp.ebi.ac.uk/pub/databases/opentargets/genetics/$OTARG_GEN_VER/v2g/


mkdir -p $OUT_DIR
cd $OUT_DIR

# Grabbing the index page, parsing out links to the parts json files.
ALL_PARTS=$(curl $BASE | sed 's/.*href="\(.*.json\)".*/\1/g' | grep "part-" | grep ".json")


# Download 4 at a time (note the -P4 in xargs below)
# This seems like a nice balance between getting things fast, but also not overloading EBI.
# (This data is also available via HTTP rather than FTP, but that is much slower and often hangs).
cnt=$(echo "$ALL_PARTS" | wc -w)
echo "$ALL_PARTS" | xargs -n1 -P4 -I{} bash -c "wget -q $BASE_FTP/{} && zstd {} && rm {}"