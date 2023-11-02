#!/bin/bash
#
# Prints out how far along we are in the vep process.
# It looks at how far along in the input file the decompressor is currently reading.

set -e

while ! pgrep -f "gzip.*for_vep.gz"; do
	echo "Waiting for start..."
	sleep 10
done

PID=$(pgrep -f "gzip.*for_vep.gz")
echo "Tracking PID $PID"
FN=$(ls -l /proc/$PID/fd/ | grep "for_vep.gz" | awk '{print $11}')
FD=$(ls -l /proc/$PID/fd/ | grep "for_vep.gz" | awk '{print $9}')

echo "Tracking FD $FD for $FN"

echo "NOTE: Progress will freeze for several minutes at the start, while VEP reads in its massive .fasta index"

SZ=$(stat --printf='%s' $FN)

while ps -p $PID > /dev/null; do
	POS=$(cat /proc/$PID/fdinfo/$FD | grep pos | awk '{print $2}')
	echo "At POS $POS / $SZ = $(echo "scale=2;$POS*100/$SZ" | bc)%"
	sleep 10
done
