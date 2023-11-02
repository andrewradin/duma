inFile=${1}
attributeName=${2}

cut -f1,4 ${inFile} | awk -v atr=${attributeName} '{print $1 "\t" atr "\t" $2};'
