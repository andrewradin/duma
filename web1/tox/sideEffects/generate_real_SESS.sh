python calculate_side_effect_prevalence_severity.py \
-p ~/2xar/ws/tox/sider_data_matrix_allSEs_sideEffectFreq_placeboNormd.tsv \
-s ~/2xar/ws/tox/gottliebSideEffectsWithUMLS.tsv \
-o temp \
--verbose > ./data/calculate_side_effect_prevalence_severity.log 2>&1

echo -e "pubchem_cid\tattribute\tvalue" > /home/ubuntu/2xar/ws/drugsets/rt.pub_chem.full.test.tsv
cat temp_real_sums.tsv | perl -lane 'print join("\t", ($F[0], "rt_SE_prevalence_severity", $F[1]));' >> /home/ubuntu/2xar/ws/drugsets/rt.pub_chem.full.test.tsv
rm temp_real_sums.tsv

python calculate_side_effect_likelihood_severity.py \
-p ~/2xar/ws/tox/offsides_from_2012Paper.tsv \
-s ~/2xar/ws/tox/gottliebSideEffectsWithUMLS.tsv \
-o temp \
--verbose > ./data/calculate_side_effect_likelihood_severity.log 2>&1

cat temp_real_sums.tsv | perl -lane 'print join("\t", ($F[0], "rt_SE_likelihood_severity", $F[1]));' >> /home/ubuntu/2xar/ws/drugsets/rt.pub_chem.full.test.tsv
rm temp_real_sums.tsv

