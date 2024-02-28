INDI_ACCUM=${1}
DRUG_ACCUM=${2}
DEMO_ACCUM=${3}
DATE_ACCUM=${4}
DATA=${5}

cd $DATA

INDI_FILE=report_drug_indication.txt

DRUG_FILE=report_drug.txt

DEMO_FILE=reports.txt

# remove the header, and process,
# then for the drug files take the primary id and the drug name
tail -n+1 ${DRUG_FILE} |\
    tr '[:upper:]' '[:lower:]' |\
    cut -d $ -f 2,4 |\
    tr "$" "\t" |\
    sed -e 's/\"//g' |\
    uniq >> "${DRUG_ACCUM}"

# for the indications data take primary id and the disease
tail -n+1 ${INDI_FILE} |\
    tr '[:upper:]' '[:lower:]' |\
    cut -d $ -f 2,5  |\
    tr "$" "\t" |\
    sed -e 's/\"//g' |\
    uniq >> "${INDI_ACCUM}"

tail -n+1 ${DEMO_FILE} |\
    tr '[:upper:]' '[:lower:]' |\
    cut -d $ -f 1,10,14,20,21 |\
    tr "$" "\t" |\
    sed -e 's/\"//g' |\
    uniq >> "${DEMO_ACCUM}"

tail -n+1 ${DEMO_FILE} |\
    tr '[:upper:]' '[:lower:]' |\
    cut -d $ -f 1,4 |\
    tr "$" "\t" |\
    sed -e 's/\"//g' |\
    uniq >> "${DATE_ACCUM}"

cd ../..
