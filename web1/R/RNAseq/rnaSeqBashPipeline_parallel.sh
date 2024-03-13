#!/bin/bash

if [[ "$#" -lt 2 ]]
then
    echo "$(basename $0) [ADDRESSES_FILE] [RAW_FASTQ_DIR]"  1>&2
    echo "   [ADDRESSES_FILE]: A file where each line is a raw data file to download. Name must be of form geoID_<'FASTQ' or 'SRA'>sToDownload.csv" 1>&2
    echo "   [RAW_FASTQ_DIR]: directory containing SRA files (the SRA files can be in subdirectories)" 1>&2
    echo "   [CORES]: how many ocres have been allotted" 1>&2
    exit 1
fi

ADDRESSES_FILE=$(echo $1 | sed 's:/$::g')
RAW_FASTQ_DIR=$(echo $2 | sed 's:/$::g')
NumberOfThreads=$(echo $3 | sed 's:/$::g')

SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # that is, the directory this script is saved in, which is where the other scripts should be as well

bash -x "${SCRIPTS_PATH}/getRawData_parallel.sh" "${ADDRESSES_FILE}" "${RAW_FASTQ_DIR}" "${NumberOfThreads}"

