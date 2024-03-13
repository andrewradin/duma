getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) return('None')
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}


thisScriptsPath <- getScriptPath()
if (thisScriptsPath != 'None') source(paste0(thisScriptsPath, '/../R/supportFunctionsForAllRCode.R'))

library(pheatmap)
library(grid)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  warning("plotKTSimilarityHeatmap.R <forKTSimilarityHeatmap.csv> <output.png>")
  quit('no', as.numeric(exitCodes['usageError']))
}
heatmapFile <- args[1]
outputPng <- args[2]

heatmapData <- read.csv(heatmapFile, header=TRUE, row.names=1)

png(outputPng)
  pheatmap(heatmapData)
dev.off()
