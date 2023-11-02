#mkdir -p plat833
#python napa_build.py --d_a_file drugbank.default --d_a_type protein --adr ~/2xar/ws/tox/adr.sider.default_pt.tsv -o plat833/dpi --adr_to_check all
#python napa_predict.py --d_a_file drugbank.default --d_a_type protein --adr plat833/dpi_path_stats.tsv -o plat833/dpi_summary
#python napa_predict.py --d_a_file drugbank.default --d_a_type protein --adr plat833/dpi_path_stats.tsv -o plat833/dpi_ftlist --to_report ft_list
#
#python napa_build.py --d_a_file /home/ubuntu/2xar/ws/49/struct/9619/output/moleculeBits.csv --d_a_type molec_bits --adr ~/2xar/ws/tox/adr.sider.default_pt.tsv -o plat833/mb --adr_to_check all
#python napa_predict.py --d_a_file /home/ubuntu/2xar/ws/49/struct/9619/output/moleculeBits.csv --d_a_type molec_bits --adr plat833/mb_path_stats.tsv -o plat833/mb_summary
#python napa_predict.py --d_a_file /home/ubuntu/2xar/ws/49/struct/9619/output/moleculeBits.csv --d_a_type molec_bits --adr plat833/mb_path_stats.tsv -o plat833/mb_ftlist --to_report ft_list
#
#python napa_build.py --d_a_file justSAs.tsv --d_a_type struc_alerts --adr ~/2xar/ws/tox/adr.sider.default_pt.tsv -o plat833/sa --adr_to_check all
#python napa_predict.py --d_a_file justSAs.tsv --d_a_type struc_alerts --adr plat833/sa_path_stats.tsv -o plat833/sa_summary
#python napa_predict.py --d_a_file justSAs.tsv --d_a_type struc_alerts --adr plat833/sa_path_stats.tsv -o plat833/sa_ftlist --to_report ft_list
#exit 0
#
#
#python padre_meta.py --summary_fts plat833/mb_summary_path_stats/ \
#                     --indiv_fts plat833/mb_ftlist_ft_list/ \
mkdir -p plat833_out
python padre_meta.py --summary_fts plat833/dpi_summary/ plat833/mb_summary/ plat833/sa_summary/ \
                     --indiv_fts plat833/dpi_ftlist/ plat833/mb_ftlist/ plat833/sa_ftlist/ \
                     -o plat833_out/ \
                     --predict all \
                     --pred_train_stats \
                     --ml_method attrSel_RF \
                     --balance \
                     --outer_iters 10 \
                     --no_test_eval \
                     --verbose \
                     --ws_ds plat833/49_wsas.txt \
                     --adrs plat833/49wsas_adr.sider.default_pt.tsv
#                     --adrs ~/2xar/ws/tox/adr.sider.default_pt.tsv
