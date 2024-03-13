library(edgeR)
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


microarrayDir <- "/mnt2/ubuntu/athRnaSeq/kallistoResults"
countsData <- read.csv(paste(microarrayDir,"/allMerged.txt", sep=""), sep=" ", row.names=1, header=TRUE)
# for some reason there is an extra space at the end of line, lazily I'm not going to worry about it, and instead remove it here
if(all(is.na(countsData[,ncol(countsData)]))){
    countsData <- countsData[,-ncol(countsData)]
}
# atherosclerosis-protective (YF-2 and YF-4) and atherosclerosis-susceptible (YF-1 and YF-3)

caseColumns <- c("YF1", "YF3")
controlColumns <- c("YF2", "YF4")
x <- buildX(countsData, caseColumns,controlColumns)
# these are the labels necessary for edgeR
groups<-c(rep("control",length(controlColumns)), rep("case", length(caseColumns)))

#================================================================================
# Run edgeR
#================================================================================
# get the data in the edgeR format
dgeList.raw <- DGEList(counts=x,group=groups)

# filter away lowly expressed genes
# Get the indexes of the rows that have at least the minimal number of samples with at least the minimal number of counts per million reads mapped
keep <- rowSums(cpm(dgeList.raw) >= minCPM) >= minSamplePortionWithReads*ncol(x)
dgeList.countFiltered <- dgeList.raw[keep,,keep.lib.sizes=FALSE]
print("Number of genes passing cpm and sample filtering")
nrow(dgeList.countFiltered)

# Apply TMM normalization, which accounts for differences between libraries/batches:
dgeList.countFiltered.normd <- calcNormFactors(dgeList.countFiltered)

# Now run a general linear model analysis 
# first create a design matrix
design <- model.matrix(~groups)
rownames(design) <- colnames(dgeList.countFiltered.normd)

# Next we estimate the overall dispersion for the dataset, to get an idea of the overall level of biological variability:
print("estimateGLMCommonDisp : estimate overall dispersion for the dataset, to get an idea of the overall level of biological variability")
dgeList.countFiltered.normd.glmDisp <- estimateGLMCommonDisp(dgeList.countFiltered.normd, design, verbose=TRUE)

# The square root of the common dispersion gives the coefficient of variation of biological variation.
# Then we estimate gene-wise dispersion estimates, allowing a possible trend with average count
dgeList.countFiltered.normd.glmDisp <- estimateGLMTrendedDisp(dgeList.countFiltered.normd.glmDisp, design)
dgeList.countFiltered.normd.glmDisp <- estimateGLMTagwiseDisp(dgeList.countFiltered.normd.glmDisp, design)

# Now proceed to determine differentially expressed genes. Fit genewise glms:
fit <- glmFit(dgeList.countFiltered.normd.glmDisp, design)
# Finally get the differential genes, with p-values
d <- dgeList.countFiltered.normd.glmDisp
options(max.print=1e7) # just to make sure everything gets printed
comparisonsName <- "control_vs_case"
# actually run the glm
lrt <- glmLRT(fit,coef=2)
results<-as.data.frame(topTags(lrt, n=nrow(lrt))) # the p-value and fold change for all genes, as a df for easy printing
results$Ensembl <- rownames(results) # make the gene names a column for easy merging below
geneOutFile <- paste0(microarrayDir, "/significantisoforms.tsv")
write.table(results, file=geneOutFile, quote=F, row.names=F, col.names=T, sep="\t")
#================================================================================
# convert to UniProt from Ensembl, and prepare genes to be written out
#================================================================================
EnsemblToUniprotConverter <- read.csv(EnsemblToUniprotMap, sep="\t", header=FALSE)#it's uniprotID \t Ensembl
colnames(EnsemblToUniprotConverter) <- c("UNIPROTKB","Ensembl")
allUniprots <- merge(results, EnsemblToUniprotConverter, by="Ensembl")

# filter on p, and FC
significantUniprots <- subset(allUniprots, allUniprots$FDR < (qmax/100) & abs(allUniprots$logFC) > abs(log2(minFC)))

# write out in the proper format for putting into the table
if (nrow(significantUniprots)>0){
  # add direction
  significantUniprots$direction <- 1
  significantUniprots[which(significantUniprots$logFC < 0),"direction"] <- -1
  toPrintOut <- as.data.frame(matrix(nrow=nrow(significantUniprots), ncol=6))
  toPrintOut[,1] <- NULL
  toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
  # Wasn't sure what to put here, so I put the portion of all reads mapped to this specific gene: log Counts per million mapped, higher means more confidence because more reads
  toPrintOut[,3] <- significantUniprots$logCPM
  toPrintOut[,4] <- round((1 - as.numeric(as.character(significantUniprots[,"FDR"]))), digits=4)
  toPrintOut[,5]<- significantUniprots[,"direction"]
  toPrintOut[,6]<- "Yun_atherosclerosis"
  outFile <- paste0(microarrayDir, "/significantUniprots.tsv")
  write.table(toPrintOut, file=outFile,quote=F, row.names=F, col.names=F, sep="\t")
  
  # the above is for us, I'll also print out more standard/redable results for them
  toPrintOut <- as.data.frame(matrix(nrow=nrow(significantUniprots), ncol=3))
  toPrintOut[,1] <- significantUniprots[,"UNIPROTKB"]
  # Wasn't sure what to put here, so I put the portion of all reads mapped to this specific gene: log Counts per million mapped, higher means more confidence because more reads
  toPrintOut[,2] <- significantUniprots$logFC
  toPrintOut[,3] <- round((as.numeric(as.character(significantUniprots[,"FDR"]))), digits=4)
  outFile2 <- paste0(microarrayDir, "/significantUniprotsForYunAndMatt.tsv")
  write.table(toPrintOut, file=outFile2, quote=F, row.names=F, col.names=F, sep="\t")

}else{
  print("uh oh")
}
#================================================================================
# Make a lot of qc plots for the edgeR approach
#================================================================================
# First, how the case and controls cluster on a multi. dim. plot (i.e. a PCA down to 2axis)

colForPlots<-c(rep("black", length(controlColumns)),rep('red', length(caseColumns)))

png(paste0(microarrayDir,'/dgeList.raw_MDSplot.png'))
    plotMDS(dgeList.raw, col=colForPlots)
    legend('topright',,c('Control','Case'),c('black', 'red'))
dev.off()

png(paste0(microarrayDir,'/dgeList.countFiltered_MDSplot.png'))
    plotMDS(dgeList.countFiltered, col=colForPlots)
    legend('topright',,c('Control','Case'),c('black', 'red'))
dev.off()

png(paste0(microarrayDir,'/dgeList.countFiltered.normd_MDSplot.png'))
    plotMDS(dgeList.countFiltered.normd, col=colForPlots)
    legend('topright',,c('Control','Case'),c('black', 'red'))
dev.off()


# Now also create a plot showing all of the pairwise correlations of the samples. Hope to see the cases and controls cluster together
corrs <- matrix(nrow=ncol(dgeList.countFiltered.normd),ncol=ncol(dgeList.countFiltered.normd))
colnames(corrs) <- colnames(dgeList.countFiltered.normd$counts)
rownames(corrs) <- colnames(dgeList.countFiltered.normd$counts)
for(i in 1:ncol(dgeList.countFiltered.normd)){
  for(j in 1:ncol(dgeList.countFiltered.normd)){
    corrs[i,j]=cor(dgeList.countFiltered.normd$counts[,i], y=dgeList.countFiltered.normd$counts[,j])
  }
}
    
library(pheatmap)    

png(paste0(microarrayDir,'/sampleCorrelationHeatmap.png'))
    pheatmap(corrs)
dev.off()

# And finally an MA plot, which shows the average count (across all samples) for each gene (X-axis), and the differential for that gene between cases and controls (y-axis)
pval <- 0.05 # what value to consider significant when plotting in MA plot (has no effect on what's output to summary file), hence why it's hard coded in, and not a parameter
de <- decideTestsDGE(lrt, p=pval, adjust="BH")
detags <- rownames(d)[as.logical(de)]
png(paste0(microarrayDir,'/MAplot.png'))
    plotSmear(lrt, de.tags=detags, main=paste("Genes below q of:", pval, sep=" "))
dev.off()
 
