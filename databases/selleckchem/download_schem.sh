#!/bin/bash

set -e

FLAG_FILE=$1
IN_DIR=$2

echo "Downloading to $IN_DIR"

mkdir -p $IN_DIR

for input in $(cat urls.txt); do
    OUT="$IN_DIR/$(basename ${input})"
    if [[ ! -f $OUT ]]; then
        echo " Downloading $input"
        wget -O $OUT.tmp $input
        mv $OUT.tmp $OUT
    fi
done

touch $FLAG_FILE
