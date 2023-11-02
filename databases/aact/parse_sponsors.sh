#!/bin/bash
cat ${1} | tr '|' '\t' | cut -f 2,4,5 | perl -lane 'print if $F[1] ~~ "lead";' | cut -f1,3
