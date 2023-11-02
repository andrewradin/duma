#!/bin/bash

# This should take roughly 10-20m, depending on the number of missing rsids that need to be found.
# If it's taking wildly longer than that, something may have gone wrong.
#
# This does benefit from having more cores, so if it's slow consider bumping machine size.
#
# If this is currently running, you can run "./rs_to_vcf_monitor.sh" to print out the current progress.

set -e

# This should contain rsids (prefixed with 'rs'), or can also contain coordinates - really anything
# that will match the grep.
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

# The VCF file is actually block-compressed-gzip (bgzip).  Tools that understand that can
# actually decompress in parallel.
# There is also a tabix file that we could use in theory, but it can't be used for rsids, and in practice
# isn't that fast for the number of things we're looking up.
((bgzip --threads $(nproc) -d -c $VCF | parallel --will-cite -j100% --pipe --block 100M -- grep -wFf $RS) >> $OUT) || true

echo "Done"

