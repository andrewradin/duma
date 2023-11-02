# This is for testing
#proteinFile <- "../Downloads/out.csv"

source('../R/supportFunctionsForAllRCode.R')
library(scales)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  warning("proteinsNumberOfTissuesVsNumberOfDrugs.R <3col csv> <output dir>")
  quit('no', as.numeric(exitCodes['usageError']))
}
proteinFile <- args[1]
outDir <- args[2]

proteinCountsData <- read.csv(proteinFile, header=FALSE, row.names=1)

setwd(outDir)
pdf("proteinsNumberOfTissuesVsNumberOfDrugs.pdf")
plot(jitter(proteinCountsData[,1], 1.5)
     , jitter(proteinCountsData[,2], 0.9)
     , pch=16
     , col=alpha("darkblue",0.1)
     , xlab="Number of tissues differentially expressed in"
     , ylab="Number of treatments known to interact")
hist(proteinCountsData[,1], col='darkblue', main="Number of tissues differentially expressed in")
hist(proteinCountsData[,2], col='darkblue',main="Number of treatments known to interact")
dev.off()
