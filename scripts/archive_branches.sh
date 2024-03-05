#!/bin/bash

set -e

git fetch -p

git checkout master

git pull

# -r does remote only
# --merged only gives merged branches
# Both HEAD and master are technically fully merged... but let's not try to delete them.
TO_DELETE=$(git branch -r --merged | cut -f1 | grep "origin/" | egrep -vw "HEAD|master|sprintqa" | sed 's:origin/::')

if [[ -n "$TO_DELETE" ]]; then
    echo -e "Branches to be tagged and removed: \n$TO_DELETE"
    read -p "Confirm (yes/[no])? " RSP
    if [[ "$RSP" == "yes" ]]; then  
        for BRANCH in $TO_DELETE; do
            echo "Tagging and Deleting $BRANCH"
            git push origin refs/remotes/origin/$BRANCH:refs/tags/archived/$BRANCH
            git push --delete origin $BRANCH
        done
    else
        echo "Not deleting anything"
    fi
else
    echo "No merged branches to be removed"
fi
