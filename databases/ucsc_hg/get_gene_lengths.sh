#!/bin/bash
mysql --user=genome --host=genome-mysql.cse.ucsc.edu -A -D hg${1} -e 'select X.displayID,K.txEnd-K.txStart as transcriptLen from kgProtAlias as X,knownGene as K where X.kgId=K.name and X.displayID=X.alias'
