#!/bin/bash

echo "Testing sigGEO.R with existing data. Good luck."

# I know what this should create, so let's run it and compare
Rscript /home/ubuntu/2xar/twoxar-demo/web1/R/sigGEO.R GDS3690 10 testing

# This is where the output should have gone
logFile="/tmp/sigGEO_GDS3690_10.test.log"

if [ -f "${logFile}" ];
then
	# Before doing anything else make sure the very first line is right
	firstLine=$(head -n 1 "${logFile}")
	if [ "${firstLine}" != '[1] "Loading existing GDS for testing"' ]
	then
		echo "FAIL: The first line of the log file is already wrong"
		exit 1
	fi

	# Check the  conversion, which should have worked well
	symbolToEntrezLine=$(grep "converting from symbol to Entrez Gene ID" "${logFile}")
	if [ "${symbolToEntrezLine}" != '[1] "In converting from symbol to Entrez Gene ID 0.989266112972947 were successfully converted."' ]
	then
		echo "FAIL: The conversion from symbol to Entrez did not go as expected"
		exit 1
	fi

	# And the final conversion done after SAM
	entrezToUniprotSummaryLine=$(grep " significant probes successfully converted to UniProt." "${logFile}")
	if [ "${entrezToUniprotSummaryLine}" != '[1] "0.97719087635054 portion of significant probes successfully converted to UniProt."' ]
	then
		echo "FAIL: The conversion from significant probe to UniProt did not go as expected"
		exit 1
	fi

	# And finally the very last line the log, which should only be this if everything worked
	lastLine=$(tail -n 1 "${logFile}")
	if [ "${lastLine}" != '[1] "insert into web1.browse_significantprotein values (NULL, \"Q9UKL3\", \"ILMN_24212\", 0.5832, 1, 10)"' ]
	then
		echo "FAIL: The last line of the log file which is the last insert into the database is not as expected"
		exit 1
	fi

	echo "SUCCESS! Everything tested looks good."

else
	echo "FAIL: Something went horribly wrong.  There is no log file at: ${logFile}"
	exit 1
fi

