ptcsDir="/home/ubuntu/2xar/twoxar-demo/web1/tox/ptc"
cd $ptcsDir
# already done
#bash getPTCData.sh

python parse_PTC_testingData.py -t data/testing_smiles.tsv -o data/PTC_testing_attributes.tsv -r data/testing_results.tsv
python parse_PTC_trainingData.py -t data/training_cas_name_tr.tsv -s data/training_corrected_smiles.txt -o data/PTC_training_attributes.tsv -r data/training_corrected_toxResults.txt

rm -f data/allPTC_bits
for f in data/PTC_training_attributes.tsv data/PTC_testing_attributes.tsv
do
    python match_with_chembl.py -i ${f} -o temp -v
    cd ../../../moleculeSimilarity/fingerprint/
    python molecularSubstructureFromAttributes.py -c -i ${ptcsDir}/${f} -b ${ptcsDir}/${f}_bits
    cd $ptcsDir
    tail -n+2 temp ${f}_bits >> ${f}
    cat ${f}_bits >> data/allPTC_bits
    rm temp ${f}_bits
done

cd ../../../moleculeSimilarity/fingerprint/
python filterMolecularSubstructuresToFeatureCsv.py -i ${ptcsDir}/data/allPTC_bits --verbose
cd $ptcsDir
rm -f data/allPTC_bits
