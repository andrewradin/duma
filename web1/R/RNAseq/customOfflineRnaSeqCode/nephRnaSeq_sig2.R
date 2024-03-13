library(pheatmap)
setwd("/home/ubuntu/2xar/twoxar-demo/web1/R")
source("/home/ubuntu/2xar/twoxar-demo/web1/R/sigGEO_supportFunctions.R")
source("/home/ubuntu/2xar/twoxar-demo/web1/R/sigGEO_rnaSeq.R")

logdir <- microarrayDir <- "/mnt2/ubuntu/rnaSeqFromAvi/quantifiedWithKallisto"
geoID <- 'dnRnaSeq'
countsData <- read.csv(paste0(microarrayDir,"/allMerged.txt"), sep = " ", row.names = 1, header = TRUE)
# for some reason there is an extra space at the end of line, lazily I'm not going to worry about it, and instead remove it here
if(all(is.na(countsData[,ncol(countsData)]))){
    countsData <- countsData[,-ncol(countsData)]
}

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
    if(cellType == 'Endo'){
        caseColumns <- grep("STZ_Endo", colnames(countsData))
        controlColumns <- grep("CTRL_Endo", colnames(countsData))
    }else if(cellType == 'Glom'){
        caseColumns <- grep("STZ_Glom", colnames(countsData))
        controlColumns <- grep("CTRL_Glom", colnames(countsData))
    }else if(cellType == 'Podo'){
        caseColumns <- grep("STZ_Podo", colnames(countsData))
        controlColumns <- grep("CTRL_Podo", colnames(countsData))
    }
    tisID <- cellType
    print(paste0("caseColumns: ", caseColumns))
    print(paste0("controlColumns: ", controlColumns))
    
    x <- buildX(countsData, caseColumns,controlColumns)
    x <- filterNonexpressedGenes(x, length(controlColumns))
    # these are the labels necessary for edgeR
    results <- run_edgeR(x, list(cont = controlColumns, case = caseColumns))
    # in converting to human we lose the direction, but we keep the FC, so just make the FC directional
    results$logFC <- results$logFC * results$direction
    geneOutFile <- paste0(microarrayDir, "/", cellType, "_allIsoforms.tsv")
    write.table(results, file = geneOutFile, quote = F, row.names = F, col.names = T, sep = "\t")
}
