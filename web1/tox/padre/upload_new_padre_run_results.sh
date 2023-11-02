s3cmd del s3://2xar-duma-padre/padre*png
for f in padre_cv_stats_*png; do s3cmd put ${f} s3://2xar-duma-padre/${f} ; done
s3cmd put padre_attrs_selected.tsv s3://2xar-duma-padre/padre_attrs_selected.tsv
s3cmd put ${1} s3://2xar-duma-tox/padre_final_predictions.tsv --force
