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


### note I changed this manually for each cohort
##
# COHORT 2
#geoID <- tisID <- 'MSA_cohort2'
#logdir <- microarrayDir <- "/mnt2/ubuntu/MSA_RNAseq/cohort2_Output"

##
# Cohort 1, NSW...
#logdir <- microarrayDir <- "/mnt2/ubuntu/MSA_RNAseq/cohort1_NSW_Output"

# ...LCM samples
#geoID <- tisID <- 'MSA_cohort1_NSW_LCM'

# ...nonLCM samples
#geoID <- tisID <- 'MSA_cohort1_NSW_whole'

##
# Cohort 1, BANNER
logdir <- microarrayDir <- "/mnt2/ubuntu/MSA_RNAseq/cohort1_BANNER_Output"

# ...LCM samples
geoID <- tisID <- 'MSA_cohort1_BANNER_LCM'

# ...nonLCM samples
#geoID <- tisID <- 'MSA_cohort1_BANNER_whole'


pubdir <- logdir
txi <- readRDS(paste0(microarrayDir,"/uniprot_expression_data.rds"))


# COHORT 2
#caseColumns <- colnames(txi$counts)[grep('_MSA_', colnames(txi$counts))]
#controlColumns <- colnames(txi$counts)[grep('_CTRL_', colnames(txi$counts))]

# Cohort 1, NSW, LCM
#caseColumns <- colnames(txi$counts)[grep("^22[0-9]*_S", colnames(txi$counts))]
#controlColumns <- colnames(txi$counts)[grep("^SU[0-9]*_S", colnames(txi$counts))]

# Cohort 1, NSW, non-LCM
#caseColumns <- colnames(txi$counts)[grep("^22[0-9]*_MSA_S", colnames(txi$counts))]
#controlColumns <- colnames(txi$counts)[grep("^SU[0-9]*_control_S", colnames(txi$counts))]

# Cohort 1, BANNER, LCM
caseColumns <- c("04_56_S10", "12_18_S12")
controlColumns <- c("03_63_S3", "06_21_S1", "98_34_S11", "99_02_S9")

# Cohort 1, BANNER, non-LCM
#caseColumns <- colnames(txi$counts)[grep("_MSA_", colnames(txi$counts))]
#controlColumns <- colnames(txi$counts)[grep("_control_", colnames(txi$counts))]


# Now reorganize the countsData so case and controls are grouped
x <- buildRNASeqX(txi$counts, caseColumns,controlColumns)
tmp <- filterNonexpressedGenes(x)
x <- tmp$x
xForPlotting <- thoroughLog2ing(x)
plotDiagnostics(xForPlotting, controlColumns, caseColumns)


# these are the labels necessary for edgeR
groups<-rep("filler",ncol(x))
final_controls <- list()
final_cases <- list()
for (i in 1:ncol(x)){
    samp <- colnames(x)[i]
    if (samp %in% controlColumns){
        groups[i] <- 'control'
        final_controls[[length(final_controls)+1]]=samp
    }else if (samp %in% caseColumns){
        groups[i] <- 'case'
        final_cases[[length(final_cases)+1]]=samp
    }else{
        print('There was an error in the case/control assignments. Quitting.')
        quit('no', as.numeric(exitCodes['usageError']))
    }
}
names(groups) <- colnames(x)
final_columnAssignments <- list(case=unlist(final_cases), cont=unlist(final_controls))


significantUniprots <- run_edgeR_tximport(x, txi$length, final_columnAssignments, groups)

if (nrow(significantUniprots)>0){
  # add direction
  toPrintOut <- as.data.frame(matrix(nrow=nrow(significantUniprots), ncol=7))
  toPrintOut[,1] <- NULL
  toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
  # Wasn't sure what to put here, so I put the portion of all reads mapped to this specific gene: log Counts per million mapped, higher means more confidence because more reads
  toPrintOut[,3] <- significantUniprots$logCPM
  toPrintOut[,4] <- round((1 - asNum(significantUniprots[,"q-value"])), digits = 4)
  toPrintOut[,5]<- significantUniprots[,"direction"]
  toPrintOut[,6] <- geoID
  toPrintOut[,7] <- significantUniprots[,"Fold Change"]
  outFile <- paste0(microarrayDir, "/", geoID, "_sigProts.tsv")
  write.table(toPrintOut, file=outFile,quote=F, row.names=F, col.names=F, sep="\t")
  
}else{
  print("uh oh")
}
