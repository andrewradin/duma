# this accepts 3 arguments:
# 1. a list of Uniprots that this drug indirectly interacts with
# 2. the ppi file used to get 1
# 3. the output file handle (to which the GO term type will be appended)

# I also need to know where this script is, because the other scripts to run will be with it
SCRIPTS_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# first let's clean up this file which is full of duplicates
sort -u $1 > "${1}.srt"
mv "${1}.srt" $1

# now let's get the background, which is all uniprots in the PPI file
# the 1st and thrid columns are the uniprots pull them all out, and uniq sort them
allUniqPPIUniprots="${2}_allUniqUniprots.txt"
if [[ ! -e $allUniqPPIUniprots ]]; then
    cut -f1,3 $2 | tr "\t" "\n" | sort -u > $allUniqPPIUniprots
fi

# complete the background by getting the uniprots not in 1, but in the ppi
notIn1="${1}_OtherUniprots"
comm -23 $allUniqPPIUniprots $1 > $notIn1

# now construct the files needed for the R script
allUnisWithScores="${1}_allUniprotsWithScores.csv"
cat $notIn1 | perl -lane 'print join(",", ($F[0],1));' > $allUnisWithScores
cat $1 | perl -lane 'print join(",", ($F[0],0));' >> $allUnisWithScores

# and run the Rscript
storageDir=$(${SCRIPTS_PATH}/../../path_helper.py storage)
for ontologyType in BP # it was taking too long, so I eliminated MF CC  for now
do
    annotFile="${storageDir}${ontologyType}GOTerms_allEvidence_Query.txt"
    Rscript "${SCRIPTS_PATH}/runningTopGoForSingleDrug.R" $allUnisWithScores 0.1 $annotFile $ontologyType $3
done

rm $1 $allUnisWithScores $notIn1

echo "GO analysis worked"

