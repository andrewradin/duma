#~/bin/bash

scoreCol=$1
actualScore=` expr $scoreCol - 2`

outFile="drugbankTani_score${actualScore}.arff"
echo "Using score${actualScore}"
echo "Results are being written to ${outFile}"

zcat drugbankTani.tsv.gz | cut -f 1,2,"${scoreCol}" | python drugBankToArff.py > "${outFile}"
