set -x

if [[ "$#" -lt 2 ]]
then
    echo "$(basename $0) [ADDRESSES_FILE] [OUT_DIR]"  1>&2
    echo "   [ADDRESSES_FILE]: A file where each line is a raw data file to download. Name must be of form geoID_<'FASTQ' or 'SRA'>sToDownload.csv" 1>&2
    echo "   [OUT_DIR]: output directory" 1>&2
    echo "   [CORES]: number of cores allotted" 1>&2
    exit 1
fi

#============================================================================================================================================
# parameters
#============================================================================================================================================
ADDRESSES_FILE=$(echo "${1}" | sed 's:/$::g')
OUT_DIR=$(echo "${2}" | sed 's:/$::g')
NumberOfThreads=$(echo "${3}" | sed 's:/$::g')

# This counts the number of spaces in the to download file, which is the same as the number of samples
numberOfSamples=$(grep -o " " "${ADDRESSES_FILE}" | wc -l )

mkdir -p "${OUT_DIR}"
# that is, the directory this script is saved in, which is where the other scripts should be as well
SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIG_TMP=$(${SCRIPTS_PATH}/../../path_helper.py bigtmp)

#============================================================================================================================================
# set up functions
#============================================================================================================================================

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
    echo "Downloading ${1}"
    for i in {1..20}; do
        if "${2}/sratoolkit/bin/fasterq-dump" -e 2 --split-3 -t "${4}" -O "${3}" "${1}"; then
            pigz -4 "${3}"/"${1}"*.fastq -p 2
            rm -f "${3}"/"${1}"*.fastq
            break
        fi
        if [[ "$i" == "20" ]]; then
            echo "!!!FAILURE to download using fasterq-dump. If this happens a lot complain to the platform team to use prefetch"
            exit 1
        else
            echo "Download $i for ${1} failed, trying again in 5 seconds"
            sleep 5
        fi
    done
    echo "Done downloading ${1} ($?)"
}
export -f recursiveFasterqDumping

if [[ "${ADDRESSES_FILE}" == *_SRAsToDownload.csv ]]
then
    TMPDIR=${BIG_TMP} parallel "-j${NumberOfThreads}" 'recursiveFasterqDumping {}' ::: $(cat "${ADDRESSES_FILE}") ::: "${SCRIPTS_PATH}" ::: "${OUT_DIR}" ::: "${BIG_TMP}"
    echo "Finished fasterq-dumping"
else
    # we're working with ae data, we just need to download the FASTQs
    cd "${OUT_DIR}"
    echo "Downloading"
    parallel "-j${NumberOfThreads}" -v 'wgetFiles {}' ::: $(cat "${ADDRESSES_FILE}")
    echo "Done"
fi

"${SCRIPTS_PATH}"/combine_redundant_fqs.py "${OUT_DIR}" --cores "${NumberOfThreads}"

