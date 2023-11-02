#!/bin/bash

# Check if the name of the ascii directory in the unzipped file is properly formatted
ASCII_DIR_NAME=$(ls ${1} | grep -i as.*i.*)
if [[ ${ASCII_DIR_NAME} != "ascii" ]]
then
    echo "${ASCII_DIR_NAME} directory found. Renaming directory to ascii"
    mv ${1}/${ASCII_DIR_NAME} ${1}/ascii
fi

echo "Ensuring that all .txt filenames are lowercase"
# Downcases all the file extensions in the ascii directory
cd ${1}/ascii
for i in $( ls | grep [A-Z] ); do mv -i $i `echo $i | tr 'A-Z' 'a-z'`; done

FILE_EXT=$(ls | grep drug.*\.txt | sed 's/.*\.//')
cd ../../..

# Testing if everything was completed as we expect for the Makefile
CORRECT_ASCII_DIR_NAME=$(ls ${1} | grep -i as.*i.*)
echo "${FILE_EXT} should be txt"
echo "${CORRECT_ASCII_DIR_NAME} should be ascii"
if [[ ${CORRECT_ASCII_DIR_NAME} == "ascii" && ${FILE_EXT} == "txt" ]]
then
	echo "1" > tmp/complete
fi
