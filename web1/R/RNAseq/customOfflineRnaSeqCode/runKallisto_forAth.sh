#!/bin/bash

# for running kallisto
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
    NumberOfThreads=$(grep -c ^processor /proc/cpuinfo) # since this is a one off thing I would like to use as many threads as we can
fi

#====================================================================================================
# settings and paths
#====================================================================================================
SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # that is, the directory this script is saved in, which is where the other scripts should be as well
storagePath=$(${SCRIPTS_PATH}/../../path_helper.py storage)
kallistoIndexDirAndPrefix="${storagePath}/forKallisto/Homo_sapiens.GRCh38.rel79.cdna_kmer"

singleEndReadFragmentSize=200 # since we don't have that information, I'm going to just give a best guess. This is supposedly a normal size for Illumina reads and that's what most people do
kmerBreakPoint=32 # what is the minimum length to use a kmer of 31, which is the default, and our other option is 23, which works b/c I throw out anything shorter than 25bp

#====================================================================================================
# Run Kallisto
#====================================================================================================

function runKallisto {
    FASTQ1="${1}"
    kmerBreakPoint="${2}"
    kallistoIndexDir="${3}"
    SCRIPTS_PATH="${4}"
    singleEndReadFragmentSize="${5}"
    readType="${6}"
    oDir="${7}"
    # this may not be the best way to do this
    # right now I'm checking to see what the biggest read length is,
    # assuming that the max will be the real, or very close to the original read length
    # which is what I care about.
    # Another option is to take the median read length
    maxReadLength=$(zcat ${FASTQ1} | head -n10000 | awk 'NR%4 == 2 {lengths[length($0)]++} END {for (l in lengths) {print l, lengths[l]}}' | sort -nr | uniq -c | head -n1 | perl -lane 'print $F[1];')
    if [ "${maxReadLength}" -le "${kmerBreakPoint}" ]
    then
       kmer=23
    else
        kmer=31
    fi
    
    if [ "${readType}" -eq 1 ]
    then
        prefix=$(basename "${FASTQ1}" | sed 's/_trimmed\.fq\.gz//g' | sed 's/_1//g') # the double sed is b/c not all have the _1
        ${SCRIPTS_PATH}/kallisto/build/src/kallisto quant -i ${kallistoIndexDir}${kmer} -o ${oDir}/${prefix} --single -l ${singleEndReadFragmentSize} "${FASTQ1}"
        #rm "${FASTQ1}"
    elif [ "${readType}" -eq 2 ]
    then
        prefix=$(basename "${FASTQ1}" | sed 's/_R1_val_1\.fq\.gz//g')
        FASTQ2Name=$(basename "${FASTQ1}" | sed 's/_R1_val_1\.fq\.gz/_R2_val_2\.fq\.gz/g')
        FQ_DIR=$(dirname $FASTQ1)
        FASTQ2="${FQ_DIR}/${FASTQ2Name}"
        ${SCRIPTS_PATH}/kallisto/build/src/kallisto quant -i ${kallistoIndexDir}${kmer} -o ${oDir}/${prefix}  "${FASTQ1}" "${FASTQ2}"
        #rm "${FASTQ2}" "${FASTQ1}"
    else
        echo "Unrecognized read type. Only SE or PE is acceptable"
    fi
}

export -f runKallisto

myarray=(`find "${FASTQ_DIR}" -maxdepth 1 -name "*_R1_val_1.fq.gz"`)
if [ ${#myarray[@]} -ne 0 ]
then
    echo "Running PE with:"
    echo $(find "${FASTQ_DIR}" -name '*_R1_val_1.fq.gz')
    parallel "-j${NumberOfThreads}" 'runKallisto {}' ::: $(find "${FASTQ_DIR}" -name '*_R1_val_1.fq.gz') ::: "${kmerBreakPoint}" ::: "${kallistoIndexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: "${singleEndReadFragmentSize}" ::: 2 ::: "${oDir}"
else
    echo "Running SE with:"
    echo $(find "${FASTQ_DIR}" -name '*_trimmed.fq.gz')
    parallel "-j${NumberOfThreads}" 'runKallisto {}' ::: $(find "${FASTQ_DIR}" -name '*_trimmed.fq.gz') ::: "${kmerBreakPoint}" ::: "${kallistoIndexDirAndPrefix}" ::: "${SCRIPTS_PATH}" ::: "${singleEndReadFragmentSize}" ::: 1 ::: "${oDir}"
fi


#====================================================================================================================
# Combine the counts data into one file now, and give a header to allow for identification
#====================================================================================================================
# I need to combine these into one file, headed by their SRX numbers
# I want the 4th line which as the coutns
cd "${oDir}"
# This is some awk magic I found online, but it works!
awk 'NF > 0 { a[$1] = a[$1] " " $4 } END { for (i in a) { print i a[i]; } }' */abundance.tsv > temp.tsv

# create a file with the srx prefixes
if [[ -e header.txt ]]
then
   rm header.txt
fi

for file in */abundance.tsv ; do echo "${file}" | perl -lane '$F[0]=~s/\/abundance\.tsv$//; print $F[0];' | tr '\n' ' ' >> header.txt; done
echo "" >> header.txt
echo "genes" > remainingHeader.txt
paste -d" " remainingHeader.txt header.txt > header2.txt

# Now put the header on and clean up
# the temp file does still have the original header though, and I want to remove that
cat header2.txt temp.tsv | perl -lane 'print unless $F[0]~~"target_id";' > allMerged.txt
#tr ' ' '\t' < allMerged.txt > allMerged.tsv # I should keep things as tsv, but the sigGEO file is already expecting space separated, and I don't want to deal with it
rm remainingHeader.txt header.txt temp.tsv

