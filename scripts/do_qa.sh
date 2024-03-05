#!/bin/bash

set -e
set -x

banner() {
    set +x
    echo ""
    echo "*******************************"
    echo "$1"
    echo "*******************************"
    echo ""
    set -x
}


update_deps=false
check_diff=true
check_unmerged=true
# the manual prev_sprint setting is a hack to get around a non-standard
# sprint name (needed for sprint341b)
prev_sprint=
while [ "$#" -gt 0 ]
do
	case $1 in
		--update-deps)
			update_deps=true
			;;
		--ignore-diff)
			check_diff=false
			;;
		--ignore-unmerged)
			check_unmerged=false
			;;
		--prev-sprint)
			prev_sprint=$2
			shift
			;;
		*)
			echo "unknown option: $1"
			exit 1
			;;
	esac
	shift
done

if [[ "$STY" == "" ]]; then
    banner "Run this inside a screen session, it takes a while."
    exit 1
fi


SCRIPTDIR=$(dirname $0)
TXAR=${SCRIPTDIR}/../
cd $TXAR

# Make sure there are no pending changes
if ! git diff --exit-code; then
    banner "Uncommitted changes"
    if $check_diff; then
        exit 1
    fi
fi

# Grab the latest QA branch.
git checkout sprintqa
git pull


# Make sure everything has been merged for this sprint.
if git branch -a --no-merge | grep "origin/plat"; then
    banner "Unmerged branches found"
    if $check_unmerged; then
        exit 1
    fi
fi

# do this after pulling sprintqa branch, in case something else was initially
# checked out
PREV_SPRINT_ID="${prev_sprint:-$(git describe --tags | cut -f1 -d'-')}"
SPRINT_ID=$(python -c "x='${PREV_SPRINT_ID}'; num=x.split('sprint')[1]; print(f'sprint{int(num)+1}');")
banner "Doing sprint '${SPRINT_ID}'"

# Pause any pending CI stuff, so it doesn't slow us down - shouldn't be anything since all merged.
# (Ignore failures here, in case CI isn't running)
ci/concourse/freeze_pipelines.py || true


if $update_deps; then
    # Refresh dependencies
    install/sub_install_py3.sh refresh
fi
# Run the latest install
install/install.sh

# Push the updated frozen dependencies.
git diff
if ! git diff --quiet; then
    banner "Committing updated deps"
    git commit -a -m 'sprintqa: Updated frozen dependencies'
    git push -u origin sprintqa
fi

# Update worker-qa.
web1/aws_op.py -m worker-qa start
ssh -tt worker-qa 'cd ~/2xar/twoxar-demo/ && git clean -fd && git reset --hard && git fetch && git checkout origin/sprintqa && install/install.sh'

# Make sure the local webserver is running; this will exit non-0 if not.
curl localhost:8000


# Kick off the qa process.
banner "Setup complete, starting main qa process"
cd experiments/dashboard
./run_sprint_qa.py -l ${SPRINT_ID} -o data/${SPRINT_ID}.json

banner "Done"
