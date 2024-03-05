#!/bin/sh

# param is GeoID(_tissue_id)

JOB=$1
cd ~/2xar/publish
mkdir -p $JOB
rmdir $JOB || exit 1
# See instructions in grab_job if the 'platform' below causes problems
scp -r platform:/home/www-data/2xar/publish/$JOB $JOB

