cd /home/ubuntu/2xar/twoxar-demo/web1/tox/ptc
mkdir -p data
cd data
curl 'http://www.predictive-toxicology.org/data/ntp/corrected_smiles.txt' > training_corrected_smiles.txt
curl 'http://www.predictive-toxicology.org/data/ntp/corrected_results.txt' > training_corrected_toxResults.txt
curl 'http://www.predictive-toxicology.org/data/fda/fda_results.tab' > testing_results.tsv
curl 'http://www.predictive-toxicology.org/data/fda/fda_smiles.tab' > testing_smiles.tsv
curl 'http://www.predictive-toxicology.org/data/ntp/cas_name_tr.tab' > training_cas_name_tr.tsv
