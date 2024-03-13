#!/bin/bash

set -e
set -x

if [[ "$#" -lt 4 ]]
then
    echo "$(basename $0) [ADDRESSES_FILE] [RAW_FASTQ_DIR] [KMER_DIR_PREF] [OUT_DIR] [N_THREADS]"  1>&2
    echo "   [ADDRESSES_FILE]: A file where each line is a raw data file to download. Name must be of form geoID_<'FASTQ' or 'SRA'>sToDownload.csv" 1>&2
    echo "   [RAW_FASTQ_DIR]: directory containing SRA files (the SRA files can be in subdirectories)" 1>&2
    echo "   [KMER_DIR_PREF]: Versioned Salmon Kmer directory prefix" 1>&2
    echo "   [OUT_DIR]: output directory" 1>&2
    echo "   [N_THREADS]: How many threads can parallel run on" 1>&2
    exit 1
fi

#============================================================================================================================================
# parameters
#============================================================================================================================================

ADDRESSES_FILE=$(echo $1 | sed 's:/$::g')
COMBINE_FILE=$(dirname "${ADDRESSES_FILE}")/$(basename "${ADDRESSES_FILE}" | sed 's/_SRR_SRAsToDownload\.csv/_SRRsToCombine\.tsv/g')

RAW_FASTQ_DIR=$(echo $2 | sed 's:/$::g')
KMER_DIR_PREF=$(echo $3 | sed 's:/$::g')
OUT_DIR=$(echo $4 | sed 's:/$::g')
NumberOfThreads=$5
TRIMMED_FASTQ_DIR="${RAW_FASTQ_DIR}/trimmed"
# This counts the number of spaces in the to download file, which is the same as the number of samples and adds one b/c the last entry doesn't have a space
numberOfSamples=$(grep -o " " ${ADDRESSES_FILE} | wc -l | xargs -n 1 bash -c 'echo $(($1 + 1))' args)

# that is, the directory this script is saved in, which is where the other scripts should be as well
SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIGTMP=$(${SCRIPTS_PATH}/../../path_helper.py bigtmp)

### Download parms
if [[ "${ADDRESSES_FILE}" == *_SRAsToDownload.csv ]]
then
    DNLD_TYPE='SRA'
else
    DNLD_TYPE='AE'
fi


#### qc parms
phredQualityCutoff=15
adaptorMatchingStringency=5
minimumReadLength=25

### salmon parms
# Based on my testing, salmon does make good use of lots of physical cores,
# but virtual/HT cores are not helpful.
# It also adds non-trivial overhead as thread count rises, which makes it more efficient
# to parallelize with multiple jobs.  However, if we have fewer jobs than cores, we're wasting
# resources, and similarly if they aren't similarly sized jobs, we often have stragglers.
ParallelRuns=$NumberOfThreads
ThreadsPerRun=$(($ParallelRuns/$numberOfSamples + 2))

echo "Using $ThreadsPerRun threads per run over $ParallelRuns parallel runs, to process $numberOfSamples samples"
# what is the minimum length to use a kmer of 31, which is the default,
# and our other option is 23, which works b/c I throw out anything shorter than 25bp
kmerBreakPoint=32

#============================================================================================================================================
# setup dirs
#============================================================================================================================================
mkdir -p "${RAW_FASTQ_DIR}"
[[ ! -d "${TRIMMED_FASTQ_DIR}" ]] && mkdir "${TRIMMED_FASTQ_DIR}"
[[ ! -d "${TRIMMED_FASTQ_DIR}/fastqc" ]] && mkdir "${TRIMMED_FASTQ_DIR}/fastqc"
[[ ! -d "${OUT_DIR}" ]] && mkdir "${OUT_DIR}"
for k in 23 31
do
        if [[ ! -d "${KMER_DIR_PREF}${k}" ]]
        then
                s3cmd get s3://2xar-versioned-datasets/salmon/$(basename ${KMER_DIR_PREF})${k}.tgz $(dirname ${KMER_DIR_PREF})
                mkdir ${KMER_DIR_PREF}TEMP && tar -xzvf ${KMER_DIR_PREF}${k}.tgz -C ${KMER_DIR_PREF}TEMP --strip-components=2
                rm -f ${KMER_DIR_PREF}${k}.tgz
                mv ${KMER_DIR_PREF}TEMP ${KMER_DIR_PREF}${k}
        fi
done

#============================================================================================================================================
# set up functions
#============================================================================================================================================
### Download functions

function wgetFiles {
    echo "Downloading ${1}"
    # Use an explicit retry loop in addition to wget retries - sometimes we get a file not found on their FTP
    # spuriously, which wget won't retry itself.
    for i in {1..20}; do
        if wget --retry-connrefused --waitretry=1 --tries 3 --read-timeout 60 --timeout 60 --continue -a downloadLogFile.txt -r "${1}"; then
            break
        fi
        if [[ "$i" == "20" ]]; then
            echo "Giving up on ${1}"
        else
            echo "Download $i for ${1} failed, trying again in a sec"
            sleep 5
        fi
    done
    echo "Done downloading ${1} ($?)"
}
export -f wgetFiles

# only comressing to level 4 w/pigz, not 6 (as is default)
# in order to make this faster since these files are ultimately deleted anyway
### Per https://github.com/ncbi/sra-tools/issues/391 if we are having failure to download
### we may want to move to a 2 step, prefetch then fasterq dump approach
function recursiveFasterqDumping {
    local ADDRESS="${1}"
    local SCRIPTS_PATH="${2}"
    local RAW_FASTQ_DIR="${3}"
    local BIGTMP="${4}"
    local ThreadsPerRun="${5}"
    echo "Downloading ${ADDRESS}"
    for i in {1..20}; do
        if "${SCRIPTS_PATH}/sratoolkit/bin/fasterq-dump" -e ${ThreadsPerRun} --split-3 -t "${BIGTMP}" -O "${RAW_FASTQ_DIR}" "${ADDRESS}"; then
            pigz -4 "${RAW_FASTQ_DIR}"/"${ADDRESS}"*.fastq -p ${ThreadsPerRun}
            rm -f "${RAW_FASTQ_DIR}"/"${ADDRESS}"*.fastq
            break
        fi
        if [[ "$i" == "20" ]]; then
            echo "!!!FAILURE to download using fasterq-dump. If this happens a lot complain to the platform team to use prefetch"
            exit 1
        else
            echo "Download $i for ${ADDRESS} failed, trying again in 5 seconds"
            sleep 5
        fi
    done
    echo "Done downloading ${ADDRESS} ($?)"
}
export -f recursiveFasterqDumping

### Processing functions

# It would be good if we could improve the adaptor matching.
# Currently it is just using the Illumina standard adaptor (which is fine,
# but could be improved since we should know the sequencing approach
function qcFastqs {
    local FASTQ1="${1}"
    local SCRIPTS_PATH="${2}"
    local phredQualityCutoff="${3}"
    local adaptorMatchingStringency="${4}"
    local minimumReadLength="${5}"
    local is_paired="${6}"
    local TRIMMED_FASTQ_DIR="${7}"

    FQ_DIR=$(dirname $FASTQ1)
    # the phred score setting is what format the read quality is in, options are (almost always) 33 or 64. This automates the guessing
    phredGuess=$(awk 'NR % 4 == 0' <(zcat "${FASTQ1}") | head -n 100000 | python "${SCRIPTS_PATH}/guessPhred.py" | perl -lane 'if ($F[0]=~/Sanger/) {print "33";} else {print "64";}')
    PHRED_SCORE_SETTING="--phred${phredGuess}"

    programCall="${SCRIPTS_PATH}/trim_galore/TrimGalore-*/trim_galore"
    # I tried putting these together and it didn't work
    qualitySetting="--quality ${phredQualityCutoff}"
    stringencySetting="--stringency ${adaptorMatchingStringency}"
    lengthSetting="--length ${minimumReadLength}"

    # trim galore's documentation suggests 4 cores as a sweet spot, beyond which there are diminishing returns.
    # The cutadapt part will also use more than $CORES, as it's also separately zipping things.
    # We don't specify CORES for fastqc, but as it won't use more than 1 thread per file.
    # cutadapt takes a lot more time, though, so fastqc shouldn't matter much.
    CORES=4

   # Trim off the adaptors and the low quality reads
    if [[ "${is_paired}" = true ]]
    then
        echo "Running ${programCall} as paired}"
        FASTQ2Name=$(basename "${FASTQ1}" | sed 's/_1\.fastq\.gz/_2\.fastq\.gz/g')
        FASTQ2="${FQ_DIR}/${FASTQ2Name}"
        ${programCall} ${PHRED_SCORE_SETTING} ${qualitySetting} -O ${TRIMMED_FASTQ_DIR} -j $CORES --paired --fastqc_args "--outdir ${TRIMMED_FASTQ_DIR}/fastqc" ${FASTQ1} ${FASTQ2}
        rm "${FASTQ2}" "${FASTQ1}"
    else
        echo "Running ${programCall} as single end}"
        ${programCall} ${PHRED_SCORE_SETTING} ${qualitySetting} -O ${TRIMMED_FASTQ_DIR} -j $CORES --fastqc_args "--outdir ${TRIMMED_FASTQ_DIR}/fastqc" ${FASTQ1}
        rm "${FASTQ1}"
    fi
}

export -f qcFastqs

### quant/Salmon functions

function runsalmon {
    local FASTQ1="${1}"
    local kmerBreakPoint="${2}"
    local KMER_DIR_PREF="${3}"
    local SCRIPTS_PATH="${4}"
    local readType="${5}"
    local OUT_DIR="${6}"
    local ThreadsPerRun="${7}"

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

    echo "Using ${FASTQ1} will run something like: ${SCRIPTS_PATH}/Salmon/bin/salmon quant -i ${KMER_DIR_PREF}${kmer} -o ${OUT_DIR}/${prefix} -l A -r ${FASTQ1} -p ${ThreadsPerRun}"

    if [ "${readType}" -eq 1 ]
    then
        # the double sed is b/c not all have the _1
        prefix=$(basename "${FASTQ1}" | sed 's/_trimmed\.fq\.gz//g' | sed 's/_1//g')
        mkdir ${OUT_DIR}/${prefix}
        ${SCRIPTS_PATH}/Salmon/bin/salmon quant -i ${KMER_DIR_PREF}${kmer} -o ${OUT_DIR}/${prefix} -l A -r "${FASTQ1}" -p ${ThreadsPerRun}
        rm "${FASTQ1}"
    elif [ "${readType}" -eq 2 ]
    then
        prefix=$(basename "${FASTQ1}" | sed 's/_1_val_1\.fq\.gz//g')
        FASTQ2Name=$(basename "${FASTQ1}" | sed 's/_1_val_1\.fq\.gz/_2_val_2\.fq\.gz/g')
        FQ_DIR=$(dirname $FASTQ1)
        FASTQ2="${FQ_DIR}/${FASTQ2Name}"
        mkdir ${OUT_DIR}/${prefix}
        ${SCRIPTS_PATH}/Salmon/bin/salmon quant -i ${KMER_DIR_PREF}${kmer} -o ${OUT_DIR}/${prefix} -l A -1 "${FASTQ1}" -2 "${FASTQ2}" -p ${ThreadsPerRun}
        rm "${FASTQ2}" "${FASTQ1}"
    else
        echo "Unrecognized read type. Only SE or PE is acceptable"
    fi
}

export -f runsalmon

function download_wrapper {
    local DNLD_TYPE="${1}"
    local RECORD="${2}"
    local SCRIPTS_PATH="${3}"
    local RAW_FASTQ_DIR="${4}"
    local BIGTMP="${5}"
    local ThreadsPerRun="${6}"

    if [ "${DNLD_TYPE}" == "SRA" ]; then
        echo "running recursiveFasterqDumping ${RECORD} ${SCRIPTS_PATH} ${RAW_FASTQ_DIR} ${BIGTMP} ${ThreadsPerRun}"
        recursiveFasterqDumping ${RECORD} ${SCRIPTS_PATH} ${RAW_FASTQ_DIR} ${BIGTMP} ${ThreadsPerRun}
    else
        echo "running wgetFiles ${RECORD}"
        wgetFiles ${RECORD}
    fi
}
export -f download_wrapper

function wrapper {
    local DNLD_TYPE="${1}"
    local RECORD="${2}"
    local SCRIPTS_PATH="${3}"
    local RAW_FASTQ_DIR="${4}"
    local BIGTMP="${5}"
    local phredQualityCutoff="${6}"
    local adaptorMatchingStringency="${7}"
    local minimumReadLength="${8}"
    local kmerBreakPoint="${9}"
    local KMER_DIR_PREF="${10}"
    local ThreadsPerRun="${11}"
    local OUT_DIR="${12}"
    local OTHERS="${13}"
    local COMBINE_FLAG="${14}"
    local TRIMMED_FASTQ_DIR="${15}"

# download
    if [ "${COMBINE_FLAG}" == true ]; then
        # first get the main record and then a list of the others
        download_wrapper ${DNLD_TYPE} ${RECORD} ${SCRIPTS_PATH} ${RAW_FASTQ_DIR} ${BIGTMP} ${ThreadsPerRun}
# This is meant to accomodate the cases where the whole experiment may have samples that need to be combined,
# but this particular sample does not
        NEED_COMBINING=false
        arrRemainRecords=(${OTHERS//,/ })
        for X in "${arrRemainRecords[@]}"
        do
            if [ "${X}" != None ]; then
                echo "downloading the following record that will be combined with ${RECORD}: ${X}"
                download_wrapper ${DNLD_TYPE} ${X} ${SCRIPTS_PATH} ${RAW_FASTQ_DIR} ${BIGTMP} ${ThreadsPerRun}
                NEED_COMBINING=true
            fi
        done
        if [ "${NEED_COMBINING}" == true ]; then
            echo "for combining, running the following ${SCRIPTS_PATH}/combine_redundant_fqs.py ${RAW_FASTQ_DIR} --record ${RECORD}"
            ${SCRIPTS_PATH}/combine_redundant_fqs.py ${RAW_FASTQ_DIR} --record ${RECORD}
        else
            echo "No combining needed for ${RECORD}"
        fi
    else
        echo "${RECORD} starting RNA-seq download..."
        download_wrapper ${DNLD_TYPE} ${RECORD} ${SCRIPTS_PATH} ${RAW_FASTQ_DIR} ${BIGTMP} ${ThreadsPerRun}
    fi

    echo "${RECORD}...done with download, starting processing..."

#qcFastqs & call salmon, depending on single or paired end
    myarray=(`find "${RAW_FASTQ_DIR}" -name "*_2.fastq.gz"`)
    if [ ${#myarray[@]} -eq 0 ]
    then
        echo "Did not find paired end reads, processing as single end reads"
        ENDTYPE='single'
    else
        echo "Found paired end reads. Processing as such"
        ENDTYPE='paired'
    fi

    if [ "${ENDTYPE}" == 'single' ]
    then
        FQ=${RAW_FASTQ_DIR}/${RECORD}.fastq.gz
        echo "running qcFastqs ${FQ} ${SCRIPTS_PATH} ${phredQualityCutoff} ${adaptorMatchingStringency} ${minimumReadLength} false ${TRIMMED_FASTQ_DIR}"
        qcFastqs "${FQ}" "${SCRIPTS_PATH}" "${phredQualityCutoff}" "${adaptorMatchingStringency}" "${minimumReadLength}" false "${TRIMMED_FASTQ_DIR}"
        TRIMMED_FQ=${TRIMMED_FASTQ_DIR}/${RECORD}_trimmed.fq.gz
        echo "runsalmon ${TRIMMED_FQ} ${kmerBreakPoint} ${KMER_DIR_PREF} ${SCRIPTS_PATH} 1 ${OUT_DIR} ${ThreadsPerRun}"
        runsalmon  "${TRIMMED_FQ}" "${kmerBreakPoint}" "${KMER_DIR_PREF}" "${SCRIPTS_PATH}" 1 "${OUT_DIR}" "${ThreadsPerRun}"
    else
        FQ=${RAW_FASTQ_DIR}/${RECORD}_1.fastq.gz
        echo "qcFastqs ${FQ} ${SCRIPTS_PATH} ${phredQualityCutoff} ${adaptorMatchingStringency} ${minimumReadLength} true ${TRIMMED_FASTQ_DIR}"
        qcFastqs "${FQ}" "${SCRIPTS_PATH}" "${phredQualityCutoff}" "${adaptorMatchingStringency}" "${minimumReadLength}" true "${TRIMMED_FASTQ_DIR}"
        TRIMMED_FQ=${TRIMMED_FASTQ_DIR}/${RECORD}_1_val_1.fq.gz
        echo "runsalmon ${TRIMMED_FQ} ${kmerBreakPoint} ${KMER_DIR_PREF} ${SCRIPTS_PATH} 2 ${OUT_DIR} ${ThreadsPerRun}"
        runsalmon "${TRIMMED_FQ}" "${kmerBreakPoint}" "${KMER_DIR_PREF}" "${SCRIPTS_PATH}" 2 "${OUT_DIR}" "${ThreadsPerRun}"
    fi
    echo "${RECORD}...all finished"
}

export -f wrapper


#================================================================
# Execute
#================================================================

# if there are lines in the _SRRsToCombine.tsv file then we need to merge them
# in the wrapper the COMBINE_CSV is ignored if the COMBINE_FLAG is false
if [[ $(wc -l <${COMBINE_FILE}) -ge 1 ]]; then
    COMBINE_FLAG=true
    RCRDS=$(cut -f1 ${COMBINE_FILE})
    COMBINE_CSV=$(cut -f2 ${COMBINE_FILE})
else
    COMBINE_FLAG=false
    RCRDS=$(cat ${ADDRESSES_FILE})
    COMBINE_CSV=null
fi

echo "Per ${COMBINE_FILE} the combine status: ${COMBINE_FLAG}"

# the link ensures that if we're running the combine version, the two arrays are essentially zipped together versuses combinatorially
TMPDIR=${BIGTMP} parallel -k --link "-j${NumberOfThreads}" 'wrapper {}' ::: \
    ${DNLD_TYPE} ::: \
    ${RCRDS} ::: \
    "${SCRIPTS_PATH}" ::: \
    "${RAW_FASTQ_DIR}" ::: \
    "${BIGTMP}" ::: \
    "${phredQualityCutoff}" ::: \
    "${adaptorMatchingStringency}" ::: \
    "${minimumReadLength}" ::: \
    "${kmerBreakPoint}" ::: \
    "${KMER_DIR_PREF}" ::: \
    "${ThreadsPerRun}" ::: \
    "${OUT_DIR}" ::: \
    ${COMBINE_CSV} ::: \
    "${COMBINE_FLAG}" ::: \
    "${TRIMMED_FASTQ_DIR}"


#====================================================================================================================
# Combine the counts data into one file now, and give a header to allow for identification
#====================================================================================================================
cd "${OUT_DIR}"
Rscript "${SCRIPTS_PATH}/../condense_SalmonResults.R"

#================================================================
# Get the list of SRRs that successfully processed
#================================================================
# TODO: It would probably be better to just check this against the output uniprot_expression_data.rds,
#       but condense_Salmon just checks directories, so copy that.
# NOTE: This will also include a non-SRR directory or two, but shouldn't matter.
ls -1d ${OUT_DIR}/*/ | rev | cut -d'/' -f2 | rev > ${OUT_DIR}/../SRRs_processed.tsv
