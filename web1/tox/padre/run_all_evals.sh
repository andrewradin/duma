valDir=${1}/validationStats/
python extract_validation_stats.py -i ${valDir}
for f in ${valDir}/*tsv; do Rscript plotAggregatedValidationStats.R $f; done
mv *dictedProb.pdf ${valDir}

tail -n+2 ${1}/*__testSet_AUCs.tsv | grep -v "==>" - | grep -v '^$' - > ${1}/all_testSet_AUCs.tsv
Rscript plotAUCBoxes.R ${1}/all_testSet_AUCs.tsv
mv *aucBoxplots.pdf ${1}
