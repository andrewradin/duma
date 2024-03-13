cd /home/ubuntu/2xar/twoxar-demo/web1/R/
ln -s RNAseq/customOfflineRnaSeqCode/nephRnaSeq_sig2.R ./
Rscript nephRnaSeq_sig2.R  > /mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto/nephRnaSeq_sig2.log 2>&1
cd RNAseq/customOfflineRnaSeqCode
for f in /mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto/*_allIsoforms.tsv;
do
    prefix=$(echo ${f} | sed 's/_allIsoforms\.tsv//g')
    bash allIsoforms_toSignificantHumanUniprotIds.sh $f 10090 > ${prefix}_conversion.log 2>&1
done

cd ../../
ln -s RNAseq/customOfflineRnaSeqCode/finish_nephRnaSeq.R ./
Rscript finish_nephRnaSeq.R > /mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto/finish_nephRnaSeq.log 2>&1
rm finish_nephRnaSeq.R nephRnaSeq_sig2.R

s3cmd put /mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto/Podo_out.tsv s3://2xar-duma-sigprot/dn_podo_sigHumProts.tsv --force
s3cmd put /mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto/Endo_out.tsv s3://2xar-duma-sigprot/dn_endo_sigHumProts.tsv --force
s3cmd put /mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto/Glom_out.tsv s3://2xar-duma-sigprot/dn_glom_sigHumProts.tsv --force
