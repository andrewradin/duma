#!/bin/bash

SCRIPTS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "${SCRIPTS_DIR}"

#===================================================================================================================
# Set up
#===================================================================================================================
ontologyType=${1/$//}
if [ "${ontologyType}" != "BP" -a "${ontologyType}" != "CC" -a "${ontologyType}" != "MF" ]; then
    echo "The only acceptable GO term types are BP, CC or MP."
    exit 1
fi
evidenceLevel=${2/$//}
if [ "${evidenceLevel}" != "all" ]; then
    echo "Currently filtering by evidence level is not supported. Please set evidence level to 'all'"
    exit 1
fi
outputDir=${3/$//}

# create files containing the GO-term information
storageDir=$(../../path_helper.py storage)
allGOTermsFile="${storageDir}/uniprotToGOTerms.tsv"
if [[ ! -e "${allGOTermsFile}" ]] ; then
    Rscript uniprotToGO.R "${allGOTermsFile}"
    # Ideally we would want to also check that the table exists
    mysql -u root < importUniprotToGO.sql
    mysqlimport --local --ignore-lines=0 --delete -u root goTerms "${allGOTermsFile}"
fi

#===================================================================================================================
# Extract data from tables using mysql
#===================================================================================================================
# create and run the query file
sqlResults="${outputDir}/${ontologyType}GOTerms_${evidenceLevel}Evidence_Query.txt"

if [[ ! -e "${sqlResults}" ]]; then
    sqlQuery="sqlQuery.sql"

    if [ "${evidenceLevel}" != "all" ]; then
        echo "select u.UniprotID, GROUP_CONCAT(u.GeneOntology) from uniprotToGOTerms as u where u.GOType = \"${ontologyType}\" and u.EvidenceCode = \"${evidenceLevel}\" group by u.UniprotID order by u.UniprotID desc ;" > "${sqlQuery}"
    else
        echo "select u.UniprotID, GROUP_CONCAT(u.GeneOntology) from uniprotToGOTerms as u where u.GOType = \"${ontologyType}\" group by u.UniprotID order by u.UniprotID desc ;" > "${sqlQuery}"
    fi
    # this is what this looks like in a more readable manner
    # select 
        # u.UniprotID, 
        # u.GeneOntology
    # from 
        # uniprotToGOTerms as u
    # where
            # u.GOType = "${ontologyType}"
        # and
            # u.EvidenceCode = "${evidenceLevel}"
    # group by u.UniprotID
    # order by u.UniprotID desc;

    # now run that command
    mysql -u root goTerms < "${sqlQuery}" > "${sqlResults}" # this prints out, (with a header) Uniprot\tGO:Term1,GO:Term2...GO:TermN
    rm "${sqlQuery}"
fi
