if [[ "$#" -lt 2 ]]
then
    echo "$(basename $0) [BAM_DIR] [OUT_DIR]"  1>&2
    echo "   [BAM_DIR]: directory containing BAM files to be quantified" 1>&2
    echo "   [OUT_DIR]: output directory" 1>&2
    exit 1
fi

BAM_DIR=$(echo $1 | sed 's:/$::g')
OUT_DIR=$(echo $2 | sed 's:/$::g')

if [[ "$#" -gt 2 ]]
then
    NumberOfThreads=$3
else
    NumberOfThreads=$(grep -c ^processor /proc/cpuinfo) # since this is a one off thing I would like to use as many threads as we can
fi


SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # that is, the directory this script is saved in, which is where the other scripts should be as well
# this is the file that contains the gene annotations/definitions
storagePath=$(cd "${SCRIPTS_PATH}" && ../../path_helper.py storage)
referenceFilesDir="${storagePath}/forStar/"
GTF="${referenceFilesDir}/Homo_sapiens.GRCh38.78.gtf"
# minimum score for considering a read to be properly mapped
minMAPQ=30

# make output directory if it doesnt exist
[[ ! -d "${OUT_DIR}" ]] && mkdir "${OUT_DIR}"

# create the function to quantify the alignment files
function quantBams {
    prefix=$(basename "${1}" | sed 's/\.bam//g')
    countsFile="${4}/${prefix}_counts.txt"
    # in a single step filter out the low confidence mappers and quantify over exons
    samtools view -q "${2}" "${1}" | htseq-count -r pos - "${3}" > "${countsFile}"
    #rm "${1}"
}
export -f quantBams
parallel "-j${NumberOfThreads}" 'quantBams {}' ::: $(find "${BAM_DIR}" -name '*.bam') ::: "${minMAPQ}" ::: "${GTF}" ::: "${OUT_DIR}"

#====================================================================================================================
# Combine the counts data into one file now, and give a header to allow for identification
#====================================================================================================================
# I need to combine these into one file, headed by their SRR numbers
cd "${OUT_DIR}"
# This is some awk magic I found online, but it works!
awk 'NF > 0 { a[$1] = a[$1] " " $2 } END { for (i in a) { print i a[i]; } }' *Aligned.sortedByCoord.out_counts.txt > temp.txt
# column two is the first file if you do ls *_counts.txt, so I could add a header with srr prefixes
# create a file with the srr prefixes
if [[ -e header.txt ]]
then
   rm header.txt
fi

for file in *Aligned.sortedByCoord.out_counts.txt ; do echo "${file}" >> header.txt; done
cat header.txt | perl -ne 's/Aligned\.sortedByCoord\.out_counts\.txt\n/ /g; print;' > header2.txt
echo "" >> header2.txt
echo "genes" > remainingHeader.txt
paste -d" " remainingHeader.txt header2.txt > header.txt

# Now put the header on and clean up
cat header.txt temp.txt > allMerged.txt
rm remainingHeader.txt header2.txt temp.txt
