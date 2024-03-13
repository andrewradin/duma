# hardcoded to only work w/SE

phredQualityCutoff=15
adaptorMatchingStringency=5
minimumReadLength=25

if [[ "$#" -lt 2 ]]
then
    echo "$(basename $0) [FASTQ_DIR] [oDir]"  1>&2
    echo "   [FASTQ_DIR]: directory containing FASTQ files to be processed" 1>&2
    echo "   [oDir]: output directory" 1>&2
    exit 1
fi

FASTQ_DIR=$(echo $1 | sed 's:/$::g')
oDir=$(echo $2 | sed 's:/$::g')
SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # that is, the directory this script is saved in, which is where the other scripts should be as well

if [[ "$#" -gt 2 ]]
then
    NumberOfThreads=$3
else
    NumberOfThreads=$(grep -c ^processor /proc/cpuinfo) # since this is a one off thing I would like to use as many threads as we can
fi


# make output directory if it doesnt exist
[[ ! -d "${oDir}" ]] && mkdir "${oDir}"
[[ ! -d "${oDir}/fastqc" ]] && mkdir "${oDir}/fastqc"

currentDir=$(pwd)
cd "${oDir}"

# It would be good if we could improve the adaptor matching.  Currently it is just using the Illumina standard adaptor (which is fine, but could be improved since we should know the sequencing approach
function qcFastqs {
    FASTQ1="${1}"
    SCRIPTS_PATH="${2}"
    phredQualityCutoff="${3}"
    adaptorMatchingStringency="${4}"
    minLength="${5}"

    FQ_DIR=$(dirname $FASTQ1)
    # the phred score setting is what format the read quality is in, options are (almost always) 33 or 64. This automates the guessing
    phredGuess=$(awk 'NR % 4 == 0' <(zcat "${FASTQ1}") | head -n 100000 | python "${SCRIPTS_PATH}/guessPhred.py" | perl -lane 'if ($F[0]=~/Sanger/) {print "33";} else {print "64";}')
    PHRED_SCORE_SETTING="--phred${phredGuess}"

    programCall="${SCRIPTS_PATH}/trim_galore/TrimGalore-*/trim_galore"
    # I tried putting these together and it didn't work
    qualitySetting="--quality ${phredQualityCutoff}"
    stringencySetting="--stringency ${adaptorMatchingStringency}"
    lengthSetting="--length ${minLength}"

   # Trim off the adaptors and the low quality reads
   ${programCall} ${PHRED_SCORE_SETTING} ${qualitySetting} --fastqc_args "--outdir ./fastqc" ${FASTQ1}
}

export -f qcFastqs

parallel "-j${NumberOfThreads}" 'qcFastqs {}' ::: $(find "${FASTQ_DIR}" -name '*.fastq.gz') ::: "${SCRIPTS_PATH}" ::: "${phredQualityCutoff}" ::: "${adaptorMatchingStringency}" ::: "${minimumReadLength}"

cd "${currentDir}"