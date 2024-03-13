#!/bin/bash

# for running salmon
# first we need to detect whether the reads are >31 bp, as that is the default kmer.
# if they're <32bp long, we'll make the Kmer 23

if [[ "$#" -lt 2 ]]
then
    echo "$(basename $0) [FASTQ_DIR] [oDir]"  1>&2
    echo "   [FASTQ_DIR]: directory containing FASTQ files to be used" 1>&2
    echo "   [oDir]: output directory" 1>&2
    exit 1
fi

FASTQ_DIR=$(echo $1 | sed 's:/$::g')
oDir=$(echo $2 | sed 's:/$::g')
# make output directory if it doesnt exist
[[ ! -d "${oDir}" ]] && mkdir "${oDir}"

if [[ "$#" -gt 2 ]]
then
    NumberOfThreads=$3
else
    # since this is a one off thing I would like to use as many threads as we can
    NumberOfThreads=$(grep -c ^processor /proc/cpuinfo)
fi

#====================================================================================================
# settings and paths
#====================================================================================================
# that is, the directory this script is saved in, which is where the other scripts should be as well
SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
storagePath=$(${SCRIPTS_PATH}/../../path_helper.py storage)
indexDirAndPrefix="${storagePath}/forSalmon/Homo_sapiens.GRCh38.cdna_kmer"

# what is the minimum length to use a kmer of 31, which is the default,
# and our other option is 23, which works b/c I throw out anything shorter than 25bp
kmerBreakPoint=32

#====================================================================================================
# Run salmon
#====================================================================================================

function runsalmon {
    FASTQ1="${1}"
    kmerBreakPoint="${2}"
    salmonIndexDir="${3}"
    SCRIPTS_PATH="${4}"
    readType="${5}"
    oDir="${6}"
    # this may not be the best way to do this
    # right now I'm checking to see what the biggest read length is,
    # assuming that the max will be the real, or very close to the original read length
    # which is what I care about.
    # Another option is to take the median read length
    maxReadLength=$(zcat ${FASTQ1} | head -n10000 |\
    awk 'NR%4 == 2 {lengths[length($0)]++} END {for (l in lengths) {print l, lengths[l]}}' |\
    sort -nr | uniq -c | head -n1 | perl -lane 'print $F[1];')

    if [ "${maxReadLength}" -le "${kmerBreakPoint}" ]
    then
       kmer=23
    else
        kmer=31
    fi

    if [ "${readType}" -eq 1 ]
    then
        # the double sed is b/c not all have the _1
        prefix=$(basename "${FASTQ1}" | sed 's/_trimmed\.fq\.gz//g' | sed 's/_1//g')
        mkdir ${oDir}/${prefix}
        ${SCRIPTS_PATH}/Salmon/bin/salmon quant -i ${salmonIndexDir}${kmer} -o ${oDir}/${prefix} -l A -r "${FASTQ1}" -p 1
#        rm "${FASTQ1}"
    elif [ "${readType}" -eq 2 ]
    then
        prefix=$(basename "${FASTQ1}" | sed 's/R1_001_val_1\.fq\.gz//g')
        FASTQ2Name=$(basename "${FASTQ1}" | sed 's/R1_001_val_1\.fq\.gz/R2_001_val_2\.fq\.gz/g')
        FQ_DIR=$(dirname $FASTQ1)
        FASTQ2="${FQ_DIR}/${FASTQ2Name}"
        mkdir ${oDir}/${prefix}
        ${SCRIPTS_PATH}/Salmon/bin/salmon quant -i ${salmonIndexDir}${kmer} -o ${oDir}/${prefix} -l A -1 "${FASTQ1}" -2 "${FASTQ2}" -p 1
#        rm "${FASTQ2}" "${FASTQ1}"
    else
        echo "Unrecognized read type. Only SE or PE is acceptable"
    fi
}

export -f runsalmon

myarray=(`find "${FASTQ_DIR}" -maxdepth 1 -name "*R1_001_val_1.fq.gz"`)
if [ ${#myarray[@]} -ne 0 ]
then
    echo "Running PE with:"
    echo $(find "${FASTQ_DIR}" -name '*R1_001_val_1.fq.gz')
    parallel "-j${NumberOfThreads}" 'runsalmon {}' ::: $(find "${FASTQ_DIR}" -name '*R1_001_val_1.fq.gz') ::: "${kmerBreakPoint}" ::: "${indexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: 2 ::: "${oDir}"
else
    echo "Running SE with:"
    echo $(find "${FASTQ_DIR}" -name '*_trimmed.fq.gz')
    parallel "-j${NumberOfThreads}" 'runsalmon {}' ::: $(find "${FASTQ_DIR}" -name '*_trimmed.fq.gz') ::: "${kmerBreakPoint}" ::: "${indexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: 1 ::: "${oDir}"
fi


#====================================================================================================================
# Combine the counts data into one file now, and give a header to allow for identification
#====================================================================================================================
cd "${oDir}"
Rscript "${SCRIPTS_PATH}/condense_SalmonResults.R"
