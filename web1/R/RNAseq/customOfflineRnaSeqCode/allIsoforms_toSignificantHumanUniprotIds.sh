#!/bin/bash

### XXX Homologene has subsequently been updated to use version data files and so the path called here
### XXX no longer works. The current approach can be seen in algorithms/run_sig.py and subseqent R code.
### XXX Basically we have human uniport - other species Entrez Gene ID files to convert directly

# this will take RNA-seq output from a non-human species for which we have a uniprot code
# and convert it to human genes while maintaining the states associated with the original transcript
# it does that by running the homolgene code

if [ $# != 2 ]
then
    echo "usage: $0 <rnaSeqOutputFile> <currentTaxonomyID>"
    exit 1
fi

inputFile=${1}
currentTaxID=${2}
if [ -s ${inputFile} ]
then
    : all good
else
    echo "${inputFile} not found"
fi

# I'll also hard code some variables here, though these could easily be passed in
minFdr=1 #"5e-2"
CLEAN=true

outDir=$(dirname ${inputFile})
prefix=$(basename ${inputFile} | sed 's/\.tsv//g')
significantEnsemblSorted=${outDir}/${prefix}_onlySignif_sortedByEnsembl.txt
justEnsembl=${outDir}/${prefix}_justEnsembl.txt
convertedEnsembl=${outDir}/${prefix}_justEnsemblConvertedToHumanUniprot.tsv
sortedConvertedEnsembl=${outDir}/${prefix}_justEnsemblConvertedToHumanUniprot.srt.tsv
finalOut=${outDir}/${prefix}_convertedToHumanUniprot.tsv

tail -n +2 ${inputFile} | awk -v fdr="${minFdr}" 'BEGIN { OFS="\t" }{if($5 < fdr){print $NF,$0}}' | sort | cut -f 2- > ${significantEnsemblSorted}
cut -f 6 ${significantEnsemblSorted} > ${justEnsembl}

bash ../../../homoloGene/convertEnsemblGeneIDsToHumanGenes.sh ${justEnsembl} ${currentTaxID} ${convertedEnsembl}

awk 'BEGIN { OFS="\t" }{print $NF,$0}' ${convertedEnsembl} | sort | cut -f 2- > ${sortedConvertedEnsembl}
join -1 6 -2 5 <(sort -k6,6 ${significantEnsemblSorted}) <(sort -k 5,5 ${sortedConvertedEnsembl}) > temp
echo -e "inputEnsembl\tlogFC\tlogCPM\tLR\tPValue\tFDR\tdirection\tHUMAN_uniprot\thumanEntrezGeneID\thomologeneID\tinputEntrezGeneID" > ${finalOut}
cat temp | tr " " "\t" >> ${finalOut}

"${CLEAN}" && rm -f ${convertedEnsembl} ${justEnsembl} ${sortedConvertedEnsembl} temp
