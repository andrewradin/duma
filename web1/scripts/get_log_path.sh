#!/bin/sh
echo $(./path_helper.py lts)log$(echo ${1} | tail -c 2)/$(echo ${1} | tail -c 4)/${1}/publish/bg_log.txt
