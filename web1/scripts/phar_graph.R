#!/usr/bin/env Rscript
library(optparse)
library(stringr)
library(readr)
library(ggrepel)
library(tidyverse)

option_list = list(
                   make_option(c("-i","--input"), type="character", default=NULL,
                               help="score file to graph", metavar="character")
                 , make_option(c("-o","--output"), type='character',default=NULL,
                               help="graph jpeg output file", metavar="character")
                 , make_option(c("-a","--atc"), type='character',default=NULL,
                               help="wsa to atc .csv", metavar="character")
                 , make_option(c("-g","--graph"), type='character',default=NULL,
                               help="file to save atc enrichment graph to", metavar="character")
                   )
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser);

scores <- read_csv(args$input) %>% gather(ref_id, score, contains('like_')) %>%
    mutate(ref_id = str_sub(ref_id,6,-1))

#{{{ graph enrichment of ATC codes:
tre_id <- scores$ref_id %>% unique()
atc <- read_csv(args$atc, col_names = c('wsa','atc'))
tre_atc <- atc %>% filter(wsa %in% tre_id) %>% .$atc
tre_atc_truncated <- tre_atc %>% str_sub(1,-3)

background_plot <- scores %>% group_by(ref_id) %>% arrange(ref_id, score) %>%
    mutate(rank = 1:n()) %>% filter(score !=1)
known_treat_plot <- background_plot %>% filter(drug_id %in% tre_id, score != 1)
atc_group_4 <- function(tre_wsa, tre_atc_trunc) {
    atc_group_4 <- atc %>% filter(str_detect(atc, tre_atc_trunc)) %>% .$wsa
    foreground_plot <- background_plot %>% filter(ref_id == tre_wsa, drug_id %in% atc_group_4) %>%
        filter(score != 1)
    return(foreground_plot)
}
atc_group_plot <- map2(tre_id, tre_atc_truncated, atc_group_4) %>% bind_rows()
print(background_plot)
p <- background_plot %>% ggplot(aes(ref_id,rank), alpha = .5) + geom_point(color = 'black') + 
    geom_point(data = atc_group_plot, aes(ref_id, rank), color ='gray') +
    geom_point(data = known_treat_plot, aes(ref_id, rank, color = as.character(drug_id))) 
ggsave(args$graph,p)

#}}}

#{{{ plot geometric mean with all other datapoints (but x axis properly ordered by geometri mean, so x axis represents same drug)
sgm <- scores %>% group_by(drug_id) %>% summarize(med = median(score)) %>%
                                                #prod(as.numeric(ScoreTanimoto)+1e-2)^(1/n())) %>%
  arrange(med)

db_levels_gm <- sgm$drug_id

sf_gm <- sgm %>% mutate(drug_id = factor(drug_id, levels = db_levels_gm))
sf <- scores %>% mutate(drug_id = factor(drug_id, levels = db_levels_gm)) #order points in ggplot2 by geometric mean

p <- sf %>% ggplot(aes(drug_id, score)) +
  geom_point(aes(color = ref_id), alpha = .3, size = .1) +
  geom_point(data = sf_gm, aes(drug_id, med), size = .1) + 
  theme_minimal() +
  theme(axis.text.x = element_blank()) +
  theme(legend.position = "none")
ggsave(args$output,p)
#}}}
