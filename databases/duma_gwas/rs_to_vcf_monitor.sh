#!/bin/bash
#
# Prints out how far along we are in the rs_to_vcf.sh process.
# This was more useful when it took many hours, but still useful for tracking.
#
# It looks at how far along in the input file the decompressor is currently reading.

set -e

PID=$(pgrep "unpigz")
echo "Tracking PID $PID"
FN=$(ls -l /proc/$PID/fd/ | grep "GCF.*gz" | awk '{print $11}')
FD=$(ls -l /proc/$PID/fd/ | grep "GCF.*gz" | awk '{print $9}')

echo "Tracking FD $FD for $FN"

POS=$(cat /proc/$PID/fdinfo/$FD | grep pos | awk '{print $2}')
SZ=$(stat --printf='%s' $FN)
echo "At POS $POS / $SZ = $(echo "scale=2;$POS*100/$SZ" | bc)%"

