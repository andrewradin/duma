#!/bin/bash

set -e
set -x

if [[ "$#" -lt 4 ]]
then
    echo "$(basename $0) [ADDRESSES_FILE] [RAW_FASTQ_DIR] [OUT_DIR]"  1>&2
    echo "   [ADDRESSES_FILE]: A file where each line is a raw data file to download. Name must be of form geoID_<'FASTQ' or 'SRA'>sToDownload.csv" 1>&2
    echo "   [RAW_FASTQ_DIR]: directory containing SRA files (the SRA files can be in subdirectories)" 1>&2
    echo "   [KMER_DIR_PREF]: Versioned Salmon Kmer directory prefix" 1>&2
    echo "   [OUT_DIR]: output directory" 1>&2
    echo "   [N_THREADS]: How many threads can parallel run on" 1>&2
    exit 1
fi

ADDRESSES_FILE=$(echo $1 | sed 's:/$::g')
RAW_FASTQ_DIR=$(echo $2 | sed 's:/$::g')
KMER_DIR_PREF=$(echo $3 | sed 's:/$::g')
OUT_DIR=$(echo $4 | sed 's:/$::g')
NumberOfThreads=$5
# This counts the number of spaces in the to download file, which is the same as the number of samples
numberOfSamples=$(grep -o " " ${ADDRESSES_FILE} | wc -l )

SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # that is, the directory this script is saved in, which is where the other scripts should be as well

# make output directory if it doesnt exist
[[ ! -d "${OUT_DIR}" ]] && mkdir "${OUT_DIR}"

#================================================================
# Process the read files
#================================================================
TRIMMED_FASTQ_DIR="${RAW_FASTQ_DIR}/trimmed"
bash "${SCRIPTS_PATH}/qcFastqs_parallel.sh" "${RAW_FASTQ_DIR}" "${TRIMMED_FASTQ_DIR}" "${NumberOfThreads}"

#================================================================
# Quantify
#================================================================
bash "${SCRIPTS_PATH}/runSalmon.sh" "${TRIMMED_FASTQ_DIR}" "${KMER_DIR_PREF}" "${OUT_DIR}" "${NumberOfThreads}"


#================================================================
# Get the list of SRRs that successfully processed
#================================================================
# TODO: It would probably be better to just check this against the output uniprot_expression_data.rds,
#       but condense_Salmon just checks directories, so copy that.
# NOTE: This will also include a non-SRR directory or two, but shouldn't matter.
ls -1d ${OUT_DIR}/*/ | rev | cut -d'/' -f2 | rev > ${OUT_DIR}/../SRRs_processed.tsv

# files to keep (for each sample)
#${TRIMMED_FASTQ_DIR}/*_trimming_report.txt
#${OUT_DIR}/uniprot_expression_data.rds
# also:
#tar ${TRIMMED_FASTQ_DIR}/fastqc # and keep that
