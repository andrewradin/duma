library(dplyr)
setwd("/home/ubuntu/2xar/twoxar-demo/web1/R")
source("/home/ubuntu/2xar/twoxar-demo/web1/R/sigGEO_supportFunctions.R")

logdir <- microarrayDir <- "/mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto/"
geoID <- 'dnRnaSeq'
#Biological replicates of samples are performed and sequenced at different batches labeled by the `batch` column in samples_meta.csv.
#CTRL: normal mice
#STZ: streptozotocin-induced diabetic mice model
#Endo: Endothelial cells separated from glomerulus
#Podo: Podocytes separated from glomerulus
#Glom: The whole glomerulus sample
#print(colnames(countsData))
# [1] "CTRL_Endo_0" "CTRL_Endo_2" "CTRL_Endo_3" "CTRL_Endo_4" "CTRL_Endo_5"
# [6] "CTRL_Endo_6" "CTRL_Endo_7" "CTRL_Endo"   "CTRL_Glom_0" "CTRL_Glom"
#[11] "CTRL_Podo_0" "CTRL_Podo_2" "CTRL_Podo"   "STZ_Endo_0"  "STZ_Endo_2"
#[16] "STZ_Endo"    "STZ_Glom_0"  "STZ_Glom_2"  "STZ_Glom"    "STZ_Podo_0"
#[21] "STZ_Podo_2"  "STZ_Podo"

for (cellType in c('Endo', "Podo", "Glom")){
    infile <- paste0(logdir, cellType, '_allIsoforms_convertedToHumanUniprot.tsv')
    tisID <- cellType
    allUniprots <- read.csv(infile, sep = "\t", header = TRUE, stringsAsFactors = FALSE, row.names=NULL)
    allUniprots$UNIPROTKB <- allUniprots$HUMAN_uniprot
    allUniprots$direction <- sign(allUniprots$logFC)
    allUniprots$logFC <- abs(allUniprots$logFC)
    significantUniprots <- condenseAndFilterGenes(allUniprots, rnaseq = TRUE)
# There are some unmerged changes that will make this the appropriate call, but not yet
#    temp <- condenseAndFilterGenes(allUniprots, rnaseq = TRUE)
#    significantUniprots <- temp$significantUniprots
    # write out in the proper format for putting into the table
    toPrintOut <- as.data.frame(matrix(nrow = nrow(significantUniprots), ncol = 7))
    toPrintOut[,1] <- NULL
    toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
    # this isn't really applicable for RNA-seq
    toPrintOut[,3] <- '1;1'
    toPrintOut[,4] <- round((1 - asNum(significantUniprots[,"q-value"])), digits = 4)
    toPrintOut[,5] <- significantUniprots[,"direction"]
    toPrintOut[,6] <- tisID
    toPrintOut[,7] <- significantUniprots[,"Fold Change"]
    outFile <- paste0(logdir, cellType, "_out.tsv")
    write.table(toPrintOut, file = outFile, quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t")
}
