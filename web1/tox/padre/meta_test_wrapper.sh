#full_drug_set=allDBs.txt
dpi_file=~/2xar/ws/dpi/dpi.drugbank.justDrugbank.tsv
mb_file=drugbank_mbs_morganFP2.tsv
sa_file=justSAs.tsv
test_set_size=1
adr_file=~/2xar/ws/tox/adr.sider.default.tsv
full_drug_set=dbs_with_withADRs_from_$(basename ${adr_file}).txt
test_name=${1}

totaladrs=$(tail -n+2 ${adr_file} | cut -f2 | sort -u | wc -l)
if [[ ! -e ${full_drug_set} ]]
then
    tail -n+2 ${adr_file}| cut -f 1 | sort -u > relevantStitch
    python convert_stitch_to_drugbank.py -i relevantStitch | cut -f 1 | sort -u > ${full_drug_set}
    rm relevantStitch
fi

mkdir -p ${test_name}
mkdir -p ${test_name}/validationStats

#for i in {1..25}; do # this should, on average catch all the drugs (there are ~480, so 25 iterations x 20 drugs per = 500 which is > 480)
for i in {1..1}; do
    mkdir -p ${test_name}/validationStats/${i}
    echo "Iteration ${i}..."
    dir_prefix="${test_name}/${i}_"
# pull out X number of drugs from DPI file -> these are now the test set and will not be used to build a new SE-P network
    Ds="${dir_prefix}drugsUsed.txt"
    stitched="${dir_prefix}drugsUsed_stitch.txt"
    converter="${dir_prefix}drugbank_to_stitch.tsv"
#    shuf -n ${test_set_size} ${full_drug_set} > ${Ds}
    # go ahead and convert this to a stitchID as well b/c we'll need that
#    python convert_drugBank_to_stitch.py -i ${Ds} > ${converter}
#    cut -f 1 ${converter} > ${Ds} # just use the drugs we have stitch IDs for, b/c those are the keys for ADRs
#    cut -f 2 ${converter} > ${stitched}
    newDpi="${dir_prefix}dpi.txt"
    newMb="${dir_prefix}Mb.txt"
    newSa="${dir_prefix}Sa.txt"
#    grep -v -f ${Ds} ${dpi_file} > ${newDpi}
#    grep -v -f ${Ds} ${mb_file} > ${newMb}
#    grep -v -f ${Ds} ${sa_file} > ${newSa}
#
    selectAdrFile="${dir_prefix}ADRsForPredDrugs.tsv"
#    head -1 ${adr_file} > ${selectAdrFile}
#    grep -w -f ${stitched} ${adr_file} >> ${selectAdrFile}

# build the ADR networks
# first for the dpi
    dpi_prefix="${dir_prefix}_dpi"
    adrDpi_file="${dpi_prefix}_path_stats.tsv"
#    echo "building DPI"
#    python napa_build.py --d_a_file ${newDpi} --d_a_type protein --adr ${adr_file} -o ${dpi_prefix} --adr_to_check all
# we use the full DPI file now to ensure the heldout drugs have DPI
    dpiPredPrefix="${dpi_prefix}predictions"
    dpiPredFile="${dpiPredPrefix}_path_stats.tsv"
 #   echo "predicting DPI"
 #   python napa_predict.py --d_a_file ${dpi_file} --d_a_type protein --adr ${adrDpi_file} -o ${dpiPredPrefix} --drugs_to_check ${full_drug_set} --path_summary_stat max

# now repeat for mol bits
    mb_prefix="${dir_prefix}_mb"
    adrmb_file="${mb_prefix}_path_stats.tsv"
#    echo "building MB"
#    python napa_build.py --d_a_file ${newMb} --d_a_type molec_bits --adr ${adr_file} -o ${mb_prefix} --adr_to_check all
# we use the full mb file now to ensure the heldout drugs have mb
    mbPredPrefix="${mb_prefix}predictions"
    mbPredFile="${mbPredPrefix}_path_stats.tsv"
#    echo "predicting MB"
#    python napa_predict.py --d_a_file ${mb_file} --d_a_type molec_bits --adr ${adrmb_file} --drugs_to_check ${full_drug_set} --path_summary_stat max -o ${mbPredPrefix}

# and for SAs
    sa_prefix="${dir_prefix}_sa"
    adrsa_file="${sa_prefix}_path_stats.tsv"
#    echo "building SA"
#    python napa_build.py --d_a_file ${newSa} --d_a_type struc_alerts --adr ${adr_file} -o ${sa_prefix} --adr_to_check all
# we use the full sa file now to ensure the heldout drugs have sa
    saPredPrefix="${sa_prefix}predictions"
    saPredFile="${saPredPrefix}_path_stats.tsv"
#    echo "predicting SA"
#    python napa_predict.py --d_a_file ${sa_file} --d_a_type struc_alerts --adr ${adrsa_file} --drugs_to_check ${full_drug_set} --path_summary_stat max -o ${saPredPrefix}

# now we need to run the meta method to combine the results
#    python padre_meta_balance.py -i ${saPredFile} ${mbPredFile} ${dpiPredFile} --adrs ${adr_file} --drugs ${full_drug_set} -o ${dir_prefix}BalancedRF --ml_method RF --predict ${Ds} --pred_train_stats
    Rscript plot_eval_stats.R ${dir_prefix}BalancedRF_testSet_eval_stats.tsv ${dir_prefix}BalancedRF_testSet_eval_stats.pdf
    Rscript plot_eval_stats.R ${dir_prefix}BalancedRF_predTrain_stats.tsv ${dir_prefix}BalancedRF_predTrain_stats.pdf
    rm ${dir_prefix}*png ${dir_prefix}*stingStats.txt
    python padre_meta.py -i ${saPredFile} ${mbPredFile} ${dpiPredFile} --adrs ${adr_file} --drugs ${full_drug_set} -o ${dir_prefix}RF --ml_method RF --predict ${Ds} --pred_train_stats
    Rscript plot_eval_stats.R ${dir_prefix}RF_testSet_eval_stats.tsv ${dir_prefix}RF_testSet_eval_stats.pdf
    Rscript plot_eval_stats.R ${dir_prefix}RF_predTrain_stats.tsv ${dir_prefix}RF_predTrain_stats.pdf
    rm ${dir_prefix}*png ${dir_prefix}*stingStats.txt
    python padre_meta.py -i ${saPredFile} ${mbPredFile} ${dpiPredFile} --adrs ${adr_file} --drugs ${full_drug_set} -o ${dir_prefix}RFwt --ml_method RF_weight --predict ${Ds} --pred_train_stats
    Rscript plot_eval_stats.R ${dir_prefix}RFwt_testSet_eval_stats.tsv ${dir_prefix}RFwt_testSet_eval_stats.pdf
    Rscript plot_eval_stats.R ${dir_prefix}RFwt_predTrain_stats.tsv ${dir_prefix}RFwt_predTrain_stats.pdf
    # I don't actually look at these plots and they just create a mess...it's easier just to delete than not make in this case
    rm ${dir_prefix}*png ${dir_prefix}*stingStats.txt

# compare to actual ADRs for those drugs
#    for j in 01 05 {1..9}
#    do
#        Rscript padre_meta_predictions.R ${converter} ${selectAdrFile} "${dir_prefix}_final_predictions.tsv" ${totaladrs} "0.${j}" ${dir_prefix} > ${test_name}/validationStats/${i}/0.${j}_confusionMatrix.tsv
#    done
done
