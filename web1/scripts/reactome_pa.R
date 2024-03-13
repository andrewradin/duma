#!/usr/bin/env Rscript

library(tidyverse)
library(readr)
library(ReactomePA)
library(optparse)
library(stringr)
library(clusterProfiler) #for bitr function (biological id translator)


option_list = list(
                   make_option(c("-i","--input"), type="character", default=NULL,
                               help="Gene Expresssion Signature File", metavar="character")
                 , make_option(c("-o","--output"), type="character", default=NULL,
                               help="Gene Expresssion Signature File", metavar="character")
                   )
opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser);

gesig_df <- read_csv(args$input, col_names = c('gene_name','value'))

entrez_id <- bitr(gesig_df$gene_name, from='UNIPROT',to='ENTREZID', annoDb="org.Hs.eg.db")

# change gesig from dataframe to named list b.c. that is what ReactomePA requires.
gesig <- gesig_df$value
names(gesig) <- entrez_id$ENTREZID

Sys.time() %>% print()
y <- gsePathway(gesig, nPerm=1000,
                minGSSize=120, pvalueCutoff=0.2,
                pAdjustMethod="BH", verbose=FALSE)

y@geneList %>% print()
print('HERE')
y@permScores %>% print()
y@result %>% as.tibble() %>% arrange(enrichmentScore) %>%
    write_tsv(args$output)


#other 'slots' of the object y are as follows:
# y@setType
# y@geneSets
# y@geneList
# y@permScores
