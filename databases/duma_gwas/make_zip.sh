#!/bin/bash

# Coordinates the other scripts to build a zip file from an unordered gwas data TSV.
# We pull the data out, split it into a bunch of manageable files, sort them, and
# pipe them into build_archive which sticks them in a zip file indexed by their key.

set -e

SCRIPT_DIR=$(dirname $0)
echo "Using scriptdir $SCRIPT_DIR"

echo "Computing file length (this is slow, but much faster than the rest!)"
IN=$1
OUT=$2


LEN=$(zcat $1 | wc -l)
# Make sure we use a local tmpdir, the real one might not have tons of space
# and sort uses lots of disk
rm -rf ./archive_tmp
mkdir -p ./archive_tmp

echo "Splitting inputs"
zcat $1 | tqdm --total $LEN | ${SCRIPT_DIR}/split_data.py ./archive_tmp

echo "Archiving"

grouped_input() {
    for x in ./archive_tmp/*; do
        # LC_ALL makes it faster, and also avoids weird locale issues
        # where sort ignores certain characters like '-'.
        zcat $x | LC_ALL=C sort -T ./sort_tmp/ -S 25% --parallel $(nproc)
    done
}

grouped_input | ${SCRIPT_DIR}/build_archive.py -o $OUT -t $LEN


# This is an alternative without the intermediate splitting, but it takes several hours to run and a ton of extra
# disk space (the sort uses uncompressed intermediate files).
#zcat $1 | tqdm --total $LEN | sort -S 50% --parallel $(nproc) -T ./sort_tmp/ | ./build_archive.py -o $OUT -t $LEN

rm -rf ./archive_tmp
