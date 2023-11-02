#!/bin/bash

set -e
set -o pipefail

PW=$(cat ~/.my.cnf  | grep password | cut -f2 -d"=")
# Chop off the first two lines, which are the password prompt
expect peewee.expect $PW $1 | tail -n+3
