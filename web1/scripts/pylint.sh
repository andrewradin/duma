#!/bin/bash
set -e

SCRIPT_DIR=$(dirname $0)

cd $SCRIPT_DIR/..

echo "Linting *.py"
pylint *.py

for dir in *; do
    if [[ ! -d "${dir}" ]]; then
        continue
    fi

    if [[ "${dir}" == "js" ]]; then
        continue
    fi

    # ML has issues and does some import *'s which breaks analysis.
    if [[ "${dir}" == "ML" ]]; then
        continue
    fi

    # Similar issues with tox, and we don't use it right now.
    if [[ "${dir}" == "tox" ]]; then
        continue
    fi

    echo "Linting ${dir}"
    pylint ${dir}
done
