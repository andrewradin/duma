bwDir="/home/ubuntu/2xar/twoxar-demo/web1/tox/boxedWarnings"
drugsetsDir="/home/ubuntu/2xar/ws/drugsets/"
chemblFile="${drugsetsDir}create.chembl.full.tsv"
cd $bwDir

cat ${chemblFile} | perl -lane 'print $F[0] if $F[1] == "max_phase" && $F[2] == "4";' > chembl_ids_max_phase_4.txt

python parse_chembl_data.py -i chembl_ids_max_phase_4.txt -c ${chemblFile} > chembl_atrs_max_phase_4.tsv

cd ../../../moleculeSimilarity/fingerprint/
python molecularSubstructureFromAttributes.py -c -i ${bwDir}/chembl_atrs_max_phase_4.tsv -b ${bwDir}/chembl_atrs_max_phase_4_bits.tsv
python filterMolecularSubstructuresToFeatureCsv.py -i ${bwDir}/chembl_atrs_max_phase_4_bits.tsv --verbose
cd $bwDir

# Put the real tox (black box warnings) in the appropriate file
grep "blackbox_warning" ${chemblFile} | perl -lane 'print join("\t", ($F[0], 'rt_boxed_warnings', $F[2]));' >> "${drugsetsDir}rt.chembl.tsv"
