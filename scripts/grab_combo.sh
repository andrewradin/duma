#!/bin/sh

# param is ws_id

WS=$1
DIR=$WS/combo
cd ~/2xar/ws
mkdir -p $DIR
rmdir $DIR || exit 1
# See instructions in grab_job if the 'platform' below causes problems
scp -r platform:/home/www-data/2xar/ws/$DIR $DIR

