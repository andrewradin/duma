#!/usr/bin/env Rscript

library(tidyverse)
library(readr)
library(stringr)
library(optparse)
library(parallel)

option_list = list(
                   make_option(c("-p","--phars"), type="character", default=NULL,
                               help="3D drugset input pharmacophore file", metavar="character"),
                   make_option(c("-t","--input_treat"), type="character", default=NULL,
                               help="treatment input file (treatment smiles)", metavar="character"),
                   make_option(c("-g","--graph"), type='character',default=NULL,
                               help="output alignment score tsv in graphable format", metavar="character"),
                   make_option(c("-i","--input_dir"), type='character',default=NULL,
                               help="directory for intermediate files",
                               metavar="character"),
                   make_option(c("-c","--num_cores"), type='numeric',default=NULL,
                               help="number of available cores",
                               metavar="numeric"),
                   make_option(c("-w","--db_to_wsa"), type='character',default=NULL,
                               help="map from drugbank id to wsa id",
                               metavar="character"),
                   make_option(c("-o","--scores"), type='character',default=NULL,
                               help="output alignment score csv in format for get_data_code_groups()",
                               metavar="character"),
                   make_option(c("-v","--score_type"), type='character',default=NULL,
                               help="allowable values include 'tversky' and 'tanimoto'",
                               metavar="character")
                   )
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser);

#{{{  Get treatment 3D structures
#===========================================================================
phars <- read_file(args$phars)
phars_vector <- read_file(args$phars) %>% str_split('\\$\\$\\$\\$\n', simplify = TRUE) %>% .[1,] %>%
	paste0('$$$$')
phar_id <- phars_vector %>% str_split('\n', simplify = TRUE) %>% .[,1]
tre_id <- read_lines(args$input_treat) # all treatment ids including those not in the drugset (if froze in openbabel)
indices_to_keep = phar_id %in% tre_id
treatments <- phars_vector[indices_to_keep]
treatments_id <- phar_id[indices_to_keep] # all treatment ids that are in the drugset

align <- function(treatment_id, treatment_phar, my_phars) {
	print(paste0('Starting alignment of [[', treatment_id,']]'))
	score_file <- paste0(args$input_dir, 'scores_', treatment_id, '.txt')
	treatment_path <- paste0(args$input_dir, 'treatment_', treatment_id,'.phar')
	write_lines(treatment_phar, path = treatment_path)

	temp_score_file <- paste0(args$input_dir,'scores_temp_',treatment_id,'.txt')
	temp_drug_file <- paste0(args$input_dir,'temp_drug_',treatment_id,'.phar')
    args <- c('.04s', #1s messed it up?
              'align-it',
              paste0('-r ', treatment_path),
              '--refType PHAR',
			  paste0('-d ', temp_drug_file),
              '--dbType PHAR',
              paste0('-s ', temp_score_file)
			  )
    for(i in 1:length(my_phars)) { # XXX use for non-debugging (comment below line)
    #for(i in 1:10) { # XXX use this for debugging (comment above line)
        write_file(my_phars[i], temp_drug_file)
        system2('timeout', args, stdout = FALSE, stderr = FALSE) #added timeout to skip comparisons that freeze
		system2('timeout', c('.5s', 'cat', temp_score_file, '>>' ,score_file))
		#print(system2('wc', c('-l', score_file), stdout = TRUE)) # for debugging to track progress
    }
	print(paste0('Finishing alignment of [[', treatment_id,']]'))
	return(read_tsv(score_file, col_names = c('ref_id', 'ref_max_vol', 'db_id', 'db_max_vol', 'overlap_max', 'overlap_exclude_spheres', 'overlap_corrected', 'num_phar', 'score_tanimoto', 'score_tversky_ref', 'score_tversky_db')))
}
c1 <- makeCluster(args$num_cores-2, type="FORK", outfile = "") #outfile = "" allows printing from within a clusterMap call so debugging actually prints out
#clusterMap(c1, align, treatments_id, treatments, MoreArgs = list(my_phars = phars_vector))
scores <- clusterMap(c1, align, treatments_id, treatments, MoreArgs = list(my_phars = phars_vector))
stopCluster(c1)
db_to_wsa <- read_csv(args$db_to_wsa, col_names = c('db_id','wsa_id'))
if(args$score_type == 'tversky') {
tidy_scores <- scores %>% bind_rows() %>%
	select(RefID=ref_id, DsID=db_id, score=score_tversky_db)
} else if(args$score_type == 'tanimoto') {
tidy_scores <- scores %>% bind_rows() %>%
	select(RefID=ref_id, DsID=db_id, score=score_tanimoto)
}
tidy_scores %>% write_csv(args$graph)
tidy_scores %>% left_join(db_to_wsa, by = c('RefID' = 'db_id')) %>%
	mutate(ref_wsa_id = paste0('like_',wsa_id)) %>%
	select(-RefID, -wsa_id) %>%
	spread(ref_wsa_id, score, fill = '0') %>%
	left_join(db_to_wsa, by = c('DsID' = 'db_id')) %>%
	select(-DsID) %>%
	select(drug_id = wsa_id, everything()) %>% # puts columns in right order
    write_csv(args$scores)
#}}}
