#!/bin/bash

# This should take roughly 10-20m, depending on the number of missing rsids that need to be found.
# If it's taking wildly longer than that, something may have gone wrong.
#
# If this is currently running, you can run "./rs_to_vcf_monitor.sh" to print out the current progress.

set -e

RS=$1
VCF=$2
OUT=$3

if [[ -z "$VCF" ]] || [[ -z "$RS" ]] || [[ -z "$OUT" ]]; then
    echo "Usage: $0 <rs file> <in vcf file> <out vcf file>"
    exit 1
fi

# This is supposed to be a performance improvement, but strangely seems to make
# our "grep -w" wildly (>100x) slower.  So don't uncomment the line below.
#export LC_ALL=C

echo "Pulling header from $VCF"
zcat $VCF | head -1000 | egrep "^#" > $OUT
echo "Pulling a bunch of IDs out of $VCF"
# - pigz only gets mild parallelization benefits during decompression, but even single threaded
#   it is meaningfully faster than zcat.
# - We previously used parallel here, but it isn't actually any faster here anymore.
# - If grep doesn't find anything it will exit non-zero, but that could legitimately happen, so ignore via " || true"
((unpigz -p4 -c $VCF | grep -wFf $RS) >> $OUT) || true

echo "Done"

