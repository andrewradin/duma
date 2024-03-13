library(pheatmap)
setwd("/home/ubuntu/2xar/twoxar-demo/web1/R")
source("/home/ubuntu/2xar/twoxar-demo/web1/R/sigGEO_supportFunctions.R")
calcAndPlotMDS <- function(data, title="Metric  MDS"){
  d <- Dist(data, nbproc = ncores) # euclidean distances between the rows
  fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
  fit # view results

  # plot solution
  x <- fit$points[,1]
  y <- fit$points[,2]
  plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
       main=title,  type="n")
  text(x, y, labels = row.names(data), cex=.7)
}

geoID <- tisID <- 'LM'
logdir <- microarrayDir <- "/mnt2/ubuntu/LM_RNAseq/quantified"
txi <- readRDS(paste0(microarrayDir,"/uniprot_expression_data.rds"))

caseColumns <- grep('^LM[1-9]', colnames(txi$counts), perl = TRUE)
controlColumns <- grep('^C[1-9]', colnames(txi$counts), perl = TRUE)

# Now reorganize the countsData so case and controls are grouped
x <- buildRNASeqX(txi$counts, caseColumns,controlColumns)
x <- filterNonexpressedGenes(x, length(controlColumns))
xForPlotting <- thoroughLog2ing(x)
plotDiagnostics(xForPlotting, controlColumns, caseColumns)
significantUniprots <- run_edgeR_tximport(x, txi$length,
                                         list(cont = controlColumns, case = caseColumns)
                                         )
if (nrow(significantUniprots)>0){
  # add direction
  toPrintOut <- as.data.frame(matrix(nrow=nrow(significantUniprots), ncol=7))
  toPrintOut[,1] <- NULL
  toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
  # Wasn't sure what to put here, so I put the portion of all reads mapped to this specific gene: log Counts per million mapped, higher means more confidence because more reads
  toPrintOut[,3] <- significantUniprots$logCPM
  toPrintOut[,4] <- round((1 - asNum(significantUniprots[,"q-value"])), digits = 4)
  toPrintOut[,5]<- significantUniprots[,"direction"]
  toPrintOut[,6]<- "LM"
  toPrintOut[,7] <- significantUniprots[,"Fold Change"]
  outFile <- paste0(microarrayDir, "/lm_pbmc_sigProts.tsv")
  write.table(toPrintOut, file=outFile,quote=F, row.names=F, col.names=F, sep="\t")
  
  # the above is for us, I'll also print out more standard/redable results for them
  toPrintOut <- as.data.frame(matrix(nrow=nrow(significantUniprots), ncol=3))
  toPrintOut[,1] <- significantUniprots[,"UNIPROTKB"]
  # Wasn't sure what to put here, so I put the portion of all reads mapped to this specific gene: log Counts per million mapped, higher means more confidence because more reads
  toPrintOut[,2] <- significantUniprots[,'Fold Change']
  toPrintOut[,3] <- round((as.numeric(as.character(significantUniprots[,"q-value"]))), digits=4)
  outFile2 <- paste0(microarrayDir, "/significantUniprotsForJoyce.tsv")
  write.table(toPrintOut, file=outFile2, quote=F, row.names=F, col.names=F, sep="\t")

}else{
  print("uh oh")
}
