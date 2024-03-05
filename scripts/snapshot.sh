#!/bin/sh

DATE=`date "+20%y.%m.%d.%H.%M"`
WD=$HOME/2xar/snapshots
FILE=$WD/$DATE.sql.gz

echo creating $FILE
mkdir -p $WD
mysqldump -u root web1 | gzip > $FILE

# to restore:
# zcat <filename> | mysql -u root web1


