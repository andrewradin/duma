#!/bin/bash

echo -e "dpimerge_id\tuniprot_id\tevidence\tdirection"

for uniprot in $(cat $1 | tail -n +2 | cut -f1 | uniq); do
    echo -e "Mol${uniprot}\t${uniprot}\t0.9\t1"
done
