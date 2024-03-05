#!/bin/bash

set -e

SCRIPTDIR="$(readlink -m $(dirname $0))"
WEB1="$SCRIPTDIR/../web1"
S3_DB="s3://2xar-backups/production/database"

LATEST_FILE="$(aws s3 ls ${S3_DB}/ | tail -1 | awk '{print $4}')"

echo "Latest production backup: $LATEST_FILE"


read -p "This will erase any local data updates, overwriting them with the production database.  Continue? [y/N] " CONTINUE
if [ "$CONTINUE" != "y" ]; then
    echo "Stopping"
    exit 0
fi

echo "** Cleaning up LTS"
cd $WEB1
./scripts/lts_status.py --fix --remove check

echo "** Fetching DB snapshot"
aws s3 cp ${S3_DB}/${LATEST_FILE} ./

echo "** Importing DB"
LINES="$(zcat ${LATEST_FILE} | wc -l)"
# the pre- and post- load files implement the load optimization described
# here: https://serverfault.com/a/568465
zcat ${LATEST_FILE} | cat ${SCRIPTDIR}/pre-load.sql - ${SCRIPTDIR}/post-load.sql | tqdm --total ${LINES} | mysql -u root web1

NEW_BRANCH="$(hostname)_before_${LATEST_FILE/.sql.gz/}"
echo "** Updating settings file to ${NEW_BRANCH}"
if grep -q "lts_branch" ./local_settings.py; then
    sed -i "s/lts_branch=.*/lts_branch='${NEW_BRANCH}'/" ./local_settings.py
else
    echo "lts_branch='${NEW_BRANCH}'" >> ./local_settings.py
fi

echo "** Deleting django cache (could have jobid-based cache values)"
rm -rf "${SCRIPTDIR}/../../ws/django_cache/"

echo "** All done!"
