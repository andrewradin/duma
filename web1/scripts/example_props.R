#!/usr/bin/env Rscript

library(tidyverse)
library(PROPS)
library(readr)
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

gesig_df <- read_csv(args$input, col_names = FALSE)#c('gene_name','value'))
print(gesig_df)

# XXX format ge data for fucnction

## WWW put into function
#props_features <- props(example_healthy, example_data)
## batch corrections
#healthy_batches = c(rep(1, 25), rep(2, 25))
#dat_batches = c(rep(1, 20), rep(2, 30))
#
#props_features_batchcorrected <- props(example_healthy, example_data, batch_correct =
#                                       TRUE, healthy_batches = healthy_batches,
#                                       dat_batches = dat_batches)

# XXX graph results, and put into same format as grep.

# then instead of using KEGG default, try adding reactoe? Or just leave it. Ask Aaron at
# next sync up.:

#entrez_id <- bitr(gesig_df$gene_name, from='UNIPROT',to='ENTREZID', annoDb="org.Hs.eg.db")
