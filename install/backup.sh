#!/bin/sh

set -e

DATE=`date "+20%y.%m.%d.%H.%M"`
WD="/tmp"
S3CMD=$HOME/2xar/opt/conda/envs/py3web1/bin/s3cmd

mysqldump -u root web1 | gzip > $WD/$DATE.sql.gz
$S3CMD put $WD/$DATE.sql.gz s3://2xar-backups/production/database/
rm -f $WD/$DATE.sql.gz

