#!/bin/bash

# for running salmon
# first we need to detect whether the reads are >31 bp, as that is the default kmer.
# if they're <32bp long, we'll make the Kmer 23

if [[ "$#" -lt 2 ]]
then
    echo "$(basename $0) [FASTQ_DIR] [oDir]"  1>&2
    echo "   [FASTQ_DIR]: directory containing FASTQ files to be used" 1>&2
    echo "   [KMER_DIR_PREF]: The versioned salmon directory prefix" 1>&2
    echo "   [oDir]: output directory" 1>&2
    exit 1
fi

FASTQ_DIR=$(echo $1 | sed 's:/$::g')
indexDirAndPrefix=$(echo $2 | sed 's:/$::g')
oDir=$(echo $3 | sed 's:/$::g')
# make output directory if it doesnt exist
[[ ! -d "${oDir}" ]] && mkdir "${oDir}"

CPUS=$(nproc)
if [[ "$#" -gt 3 ]]
then
    NumberOfThreads=$4
else
    # since this is a one off thing I would like to use as many threads as we can
    NumberOfThreads=$CPUS
fi

# Based on my testing, salmon does make good use of lots of physical cores,
# but virtual/HT cores are not helpful.
# It also adds non-trivial overhead as thread count rises, which makes it more efficient
# to parallelize with multiple jobs.  However, if we have fewer jobs than cores, we're wasting
# resources, and similarly if they aren't similarly sized jobs, we often have stragglers.
#
# We will try to strike a balance here, but will also oversubscribe CPUs a little bit,
# to guard against stragglers lingering using only a few CPUs.
NSamples=$(ls -1 $FASTQ_DIR/*.fq.gz | grep -v "_val_2" | wc -l)
ParallelRuns=$NumberOfThreads
ThreadsPerRun=$(($ParallelRuns/$NSamples + 2))

echo "Using $ThreadsPerRun threads per run over $ParallelRuns parallel runs, to process $NSamples samples"


for k in 23 31
do
	if [[ ! -d "${indexDirAndPrefix}${k}" ]]
	then
		s3cmd get s3://2xar-versioned-datasets/salmon/$(basename ${indexDirAndPrefix})${k}.tgz $(dirname ${indexDirAndPrefix})
		mkdir ${indexDirAndPrefix}TEMP && tar -xzvf ${indexDirAndPrefix}${k}.tgz -C ${indexDirAndPrefix}TEMP --strip-components=2
                rm -f ${indexDirAndPrefix}${k}.tgz
                mv ${indexDirAndPrefix}TEMP ${indexDirAndPrefix}${k}
	fi
done

#====================================================================================================
# settings and paths
#====================================================================================================
# that is, the directory this script is saved in, which is where the other scripts should be as well
SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
    ThreadsPerRun="${7}"
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
        ${SCRIPTS_PATH}/Salmon/bin/salmon quant -i ${salmonIndexDir}${kmer} -o ${oDir}/${prefix} -l A -r "${FASTQ1}" -p ${ThreadsPerRun}
        rm "${FASTQ1}"
    elif [ "${readType}" -eq 2 ]
    then
        prefix=$(basename "${FASTQ1}" | sed 's/_1_val_1\.fq\.gz//g')
        FASTQ2Name=$(basename "${FASTQ1}" | sed 's/_1_val_1\.fq\.gz/_2_val_2\.fq\.gz/g')
        FQ_DIR=$(dirname $FASTQ1)
        FASTQ2="${FQ_DIR}/${FASTQ2Name}"
        mkdir ${oDir}/${prefix}
        ${SCRIPTS_PATH}/Salmon/bin/salmon quant -i ${salmonIndexDir}${kmer} -o ${oDir}/${prefix} -l A -1 "${FASTQ1}" -2 "${FASTQ2}" -p ${ThreadsPerRun}
        rm "${FASTQ2}" "${FASTQ1}"
    else
        echo "Unrecognized read type. Only SE or PE is acceptable"
    fi
}

export -f runsalmon

myarray=(`find "${FASTQ_DIR}" -maxdepth 1 -name "*1_val_1.fq.gz"`)
if [ ${#myarray[@]} -ne 0 ]
then
    echo "Running PE with:"
    echo $(find "${FASTQ_DIR}" -name '*1_val_1.fq.gz')
    echo "RUNNING:"
    echo "-j${ParallelRuns}" 'runsalmon {}' ::: $(find "${FASTQ_DIR}" -name '*1_val_1.fq.gz') ::: "${kmerBreakPoint}" ::: "${indexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: 2 ::: "${oDir}" ::: "${ThreadsPerRun}"
    parallel "-j${ParallelRuns}" 'runsalmon {}' ::: $(find "${FASTQ_DIR}" -name '*1_val_1.fq.gz') ::: "${kmerBreakPoint}" ::: "${indexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: 2 ::: "${oDir}" ::: "${ThreadsPerRun}"
else
    echo "Running SE with:"
    echo $(find "${FASTQ_DIR}" -name '*_trimmed.fq.gz')
    echo "RUNNING:"
    echo "-j${ParallelRuns}" 'runsalmon {}' ::: $(find "${FASTQ_DIR}" -name '*_trimmed.fq.gz') ::: "${kmerBreakPoint}" ::: "${indexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: 1 ::: "${oDir}" ::: "${ThreadsPerRun}"
    parallel "-j${ParallelRuns}" 'runsalmon {}' ::: $(find "${FASTQ_DIR}" -name '*_trimmed.fq.gz') ::: "${kmerBreakPoint}" ::: "${indexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: 1 ::: "${oDir}" ::: "${ThreadsPerRun}"
fi


#====================================================================================================================
# Combine the counts data into one file now, and give a header to allow for identification
#====================================================================================================================
cd "${oDir}"
Rscript "${SCRIPTS_PATH}/../condense_SalmonResults.R"
