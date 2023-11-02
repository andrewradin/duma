dpi_file=drugbank_unfiltered_mbs.tsv
allRelevantDrugsFile="allRelevantDrugBanks_forTestingMB.txt"
test_set_size=1
se_file=${2}
test_name=${1}

mkdir -p ${test_name}

for i in {1..50}; do
    echo "Iteration ${i}..."
    gen_prefix="${test_name}/${i}_"
# pull out X number of drugs from DPI file -> these are now the test set and will not be used to build a new SE-P network
    Ds="${gen_prefix}drugsUsed.txt"
    stitched="${gen_prefix}drugsUsed_stitch.txt"
    shuf -n ${test_set_size} ${allRelevantDrugsFile} > ${Ds}
    # go ahead and convert this to a stitchID as well b/c we'll need that
    python convert_drugBank_to_stitch.py -i ${Ds} > ${stitched}
    newDpi="${gen_prefix}dpi.txt"
    grep -v -f ${Ds} ${dpi_file} > ${newDpi}
#
    selectSeFile="${gen_prefix}sideEffectsForPredDrugs.tsv"
    head -1 ${se_file} > ${selectSeFile}
    grep -w -f ${stitched} ${se_file} >> ${selectSeFile}
# check different values of Q and path number
    for path in 1 2 3
    do
        mkdir -p "${test_name}_path/direct_min${path}"
        dir_prefix="${test_name}_path/direct_min${path}/${i}_"
# build the SE-P network
# first for the direct
        dir_sep_file="${dir_prefix}_path_sums.tsv"
        python napa_build.py \
        --d_a_file ${newDpi} --d_a_type molec_bits --adr ${se_file} -o ${dir_prefix} \
        --adr_to_check all --pbcid_to_db pubchemcid_to_drugbankid.tsv \
        --min_paths ${path}
# predict SEs for the heldout drugs
# we use the full DPI file now to ensure the heldout drugs have DPI
        dirPredPrefix="${dir_prefix}predictions"
        dirPredFile="${dirPredPrefix}_path_sums.tsv"
        python napa_predict.py --d_a_file ${dpi_file} --d_a_type molec_bits\
        --adr ${dir_sep_file} --drugs_to_check ${Ds} -o ${dirPredPrefix} \
        --pbcid_to_db pubchemcid_to_drugbankid.tsv
# compare to actual SEs for those drugs
        if [[ $(wc -l <${dirPredFile}) -ge 1 ]]; then
            Rscript compare_padres.R ${dirPredFile} ${selectSeFile} ${dirPredPrefix} $(cut -f1 ${dir_sep_file} | sort -u | wc -l)
        fi
    done
    for q in 0.05 0.1 0.15 0.2
    do
        mkdir -p "${test_name}_q/direct_min${q}"
        dir_prefix="${test_name}_q/direct_min${q}/${i}_"
# build the SE-P network
# first for the direct
        dir_sep_file="${dir_prefix}_path_sums.tsv"
        python napa_build.py \
        --d_a_file ${newDpi} --d_a_type molec_bits --adr ${se_file} -o ${dir_prefix} \
        --adr_to_check all --pbcid_to_db pubchemcid_to_drugbankid.tsv \
        --min_q ${q}
# predict SEs for the heldout drugs
# we use the full DPI file now to ensure the heldout drugs have DPI
        dirPredPrefix="${dir_prefix}predictions"
        dirPredFile="${dirPredPrefix}_path_sums.tsv"
        python napa_predict.py --d_a_file ${dpi_file} --d_a_type molec_bits\
        --adr ${dir_sep_file} --drugs_to_check ${Ds} -o ${dirPredPrefix} \
        --pbcid_to_db pubchemcid_to_drugbankid.tsv
# compare to actual SEs for those drugs
        if [[ $(wc -l <${dirPredFile}) -ge 1 ]]; then
            Rscript compare_padres.R ${dirPredFile} ${selectSeFile} ${dirPredPrefix} $(cut -f1 ${dir_sep_file} | sort -u | wc -l)
        fi
    done
    for or in 1 1.2 1.5 1.8 2 4
    do
        mkdir -p "${test_name}_or/direct_min${or}"
        dir_prefix="${test_name}_or/direct_min${or}/${i}_"
# build the SE-P network
# first for the direct
        dir_sep_file="${dir_prefix}_path_sums.tsv"
        python napa_build.py \
        --d_a_file ${newDpi} --d_a_type molec_bits --adr ${se_file} -o ${dir_prefix} \
        --adr_to_check all --pbcid_to_db pubchemcid_to_drugbankid.tsv \
        --min_or ${or}
# predict SEs for the heldout drugs
# we use the full DPI file now to ensure the heldout drugs have DPI
        dirPredPrefix="${dir_prefix}predictions"
        dirPredFile="${dirPredPrefix}_path_sums.tsv"
        python napa_predict.py --d_a_file ${dpi_file} --d_a_type molec_bits\
        --adr ${dir_sep_file} --drugs_to_check ${Ds} -o ${dirPredPrefix} \
        --pbcid_to_db pubchemcid_to_drugbankid.tsv
# compare to actual SEs for those drugs
        if [[ $(wc -l <${dirPredFile}) -ge 1 ]]; then
            Rscript compare_padres.R ${dirPredFile} ${selectSeFile} ${dirPredPrefix} $(cut -f1 ${dir_sep_file} | sort -u | wc -l)
        fi
    done
done

Rscript processPredictions.R ${test_name}_path 50 paths
Rscript processPredictions.R ${test_name}_q 50 q
Rscript processPredictions.R ${test_name}_or 50 oddsRatio
