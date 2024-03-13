#!/bin/sh

resp=`curl -s -X POST -H 'Content-type: application/json' --data '{"text":"'"$1"'"}'  https://hooks.slack.com/services/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`
#echo $resp
test "$resp" = "ok"
