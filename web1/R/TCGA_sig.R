library(edgeR)
setwd("/home/ubuntu/2xar/twoxar-demo/web1/R")
source("/home/ubuntu/2xar/twoxar-demo/web1/R/sigGEO_supportFunctions.R")

# loading settings that are normally set by the user
minCPM <- 1
minSamplePortionWithReads <- 0.5
EntrezToUniprotMap <- paste0(storageDir,'HUMAN_9606_Protein_Entrez.tsv')

calcAndPlotMDS <- function(data, title = "Metric  MDS"){
  d <- Dist(data, nbproc = ncores) # euclidean distances between the rows
  fit <- cmdscale(d, eig = TRUE, k = 2) # k is the number of dim
#  fit # view results

  # plot solution
  x <- fit$points[,1]
  y <- fit$points[,2]
  plot(x, y, xlab = "Coordinate 1", ylab = "Coordinate 2",
       main = title,  type = "n")
  text(x, y, labels = row.names(data), cex = 0.7)
}

findColAssignments <- function(v){
    # the 14th and 15th characters denote tumor or control status
    # all tumors are single digits ( I don't think 10 is actually used)
    vals <- as.numeric(substr(v, 14, 15))
    controlColInds <- vals > 10
    caseColInds <- vals < 10
    return(list(case = v[caseColInds], control = v[controlColInds]))
}    


#================================================================================
# read in data
#================================================================================
args <- commandArgs(trailingOnly = TRUE)
microarrayDir <- args[1]
infile <- args[2]

countsData <- read.table(gzfile(paste(microarrayDir, infile, sep = "/")),
                       sep = "\t",
                       row.names = 1,
                       header = TRUE,
                       stringsAsFactors = FALSE
                      )

# the 1st row is useless, so we drop that
countsData <- countsData[-1,]
rownames(countsData) <- unlist(sapply(strsplit(rownames(countsData),
                                               "|", fixed = TRUE),
                                      '[[', 2)
                              )
colAssignments <- findColAssignments(colnames(countsData))

if (length(colAssignments$control) == 0){
    print('WARNING: No controls found. Quiting')
    quit('no')
}

x <- buildRNASeqX(countsData, colAssignments$case, colAssignments$control)

# these are the labels necessary for edgeR
groups <- c(rep("control",length(colAssignments$control)),
            rep("case", length(colAssignments$case))
            )

#================================================================================
# Run edgeR
#================================================================================
# get the data in the edgeR format
dgeList.raw <- DGEList(counts = x,group = groups)

# filter away lowly expressed genes
# Get the indexes of the rows that have at least the minimal number of samples
# with at least the minimal number of counts per million reads mapped
keep <- rowSums(cpm(dgeList.raw) >= minCPM) >= minSamplePortionWithReads*ncol(x)
dgeList.countFiltered <- dgeList.raw[keep,,keep.lib.sizes = FALSE]
print("Number of genes passing cpm and sample filtering")
nrow(dgeList.countFiltered)
results <- run_core_edgeR(groups, dgeList.countFiltered)
# make the gene names a column for easy merging below
results$Entrez <- rownames(results)

#================================================================================
# convert to UniProt from Entrez, and prepare genes to be written out
#================================================================================
EntrezToUniprotConverter <- read.csv(EntrezToUniprotMap,
                                      sep = "\t",
                                      header = FALSE)
colnames(EntrezToUniprotConverter) <- c("UNIPROTKB","Entrez")
allUniprots <- merge(results, EntrezToUniprotConverter, by = "Entrez")

# filter on p, and FC
significantUniprots <- subset(allUniprots,
                              allUniprots$FDR < (qmax/100) &
                              abs(allUniprots$logFC) > abs(log2(minFC))
                             )

# write out in the proper format for putting into the table
if (nrow(significantUniprots)>0){
    significantUniprots$direction <- 1
    significantUniprots[which(significantUniprots$logFC < 0),"direction"] <- -1
    toPrintOut <- as.data.frame(matrix(nrow = nrow(significantUniprots), ncol = 7))
    toPrintOut[,1] <- NULL
    toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
    # this isn't really applicable for RNA-seq
    toPrintOut[,3] <- '1;1'
    toPrintOut[,4] <- formatC((1 - asNum(significantUniprots[,"FDR"])),
                              format = 'e',
                              digits = 3
                              )
    toPrintOut[,5] <- significantUniprots[,"direction"]
    toPrintOut[,6] <- infile
    toPrintOut[,7] <- significantUniprots[,"logFC"]
    outFile <- paste0(microarrayDir, "/", databaseTable, ".tsv")
    write.table(toPrintOut, file = outFile, quote = FALSE,
                row.names = FALSE, col.names = FALSE, sep = "\t")
}else{
    print(sprintf("%s has no significant expression", infile))
}

#================================================================================
# Make a lot of qc plots for the edgeR approach
#================================================================================
# First, how the case and controls cluster on a multi. dim. plot (i.e. a PCA down to 2axis)

colForPlots<-c(rep("black", length(colAssignments$control)),rep('red', length(colAssignments$case)))

png(paste0(microarrayDir,'/dgeList.raw_MDSplot.png'))
    plotMDS(dgeList.raw, col = colForPlots)
    legend('topright',,c('Control','Case'),c('black', 'red'))
dev.off()

png(paste0(microarrayDir,'/dgeList.countFiltered_MDSplot.png'))
    plotMDS(dgeList.countFiltered, col = colForPlots)
    legend('topright',,c('Control','Case'),c('black', 'red'))
dev.off()

png(paste0(microarrayDir,'/dgeList.countFiltered_MDSplot.png'))
    plotMDS(dgeList.countFiltered, col = colForPlots)
    legend('topright',,c('Control','Case'),c('black', 'red'))
dev.off()


# Now also create a plot showing all of the pairwise correlations of the samples.
# Hope to see the cases and controls cluster together
corrs <- matrix(nrow = ncol(dgeList.countFiltered),
                ncol = ncol(dgeList.countFiltered)
               )
colnames(corrs) <- colnames(dgeList.countFiltered$counts)
rownames(corrs) <- colnames(dgeList.countFiltered$counts)
for(i in 1:ncol(dgeList.countFiltered)){
  for(j in 1:ncol(dgeList.countFiltered)){
    corrs[i,j] <- cor(dgeList.countFiltered$counts[,i],
                      y = dgeList.countFiltered$counts[,j]
                     )
  }
}
    
library(pheatmap)    

png(paste0(microarrayDir,'/sampleCorrelationHeatmap.png'))
    pheatmap(corrs)
dev.off()

### I took this out since we weren't looking at them anyhow and the lrt object is no longer in this script
### it's in the EdgeR call above

# And finally an MA plot, which shows the average count (across all samples) for each gene (X-axis),
# and the differential for that gene between cases and controls (y-axis)
# what value to consider significant when plotting in MA plot
# (has no effect on what's output to summary file), hence why it's hard coded in, and not a parameter
#pval <- 0.05
#de <- decideTestsDGE(lrt, p = pval, adjust = "BH")
#detags <- rownames(d)[as.logical(de)]
#png(paste0(microarrayDir,'/MAplot.png'))
#    plotSmear(lrt,
#              de.tags = detags,
#              main = paste("Genes below q of:", pval)
#             )
#dev.off()

