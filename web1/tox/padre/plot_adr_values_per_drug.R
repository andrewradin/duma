# this takes an adr.*tsv file and for every drug (col1) plots the value (col3) for each ADR (col2)
# ToDo: 
# Color code the bars by the ADR severity
# convert the UMLS code to a name - this may be better done in python

#=============================
# Get set up
#=============================
library(pheatmap)
getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) stop("can't determine script dir: please call the script with Rscript")
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}


thisScriptsPath <- getScriptPath()
source(paste0(thisScriptsPath, './../R/supportFunctionsForAllRCode.R'))

#=============================
# Hardcoded params
#=============================
drug_col <- 1
adr_col <- 2
score_col <- 3


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  warning("Rscript plot_adr_values_per_drug.R <adr_file.tsv> <output directory>")
  quit('no', as.numeric(exitCodes['usageError']))
}
adrs <- read.table(args[1], header=T, sep="\t", stringsAsFactors=F)
outputDir <- args[2]
dir.create(outputDir)

#===============================================================================================
# For each drug, plot it's score for each ADR for which it as a score >0
#===============================================================================================
allRes <- list()
for (drug in unique(adrs[,drug_col])){
    adr_scores <- adrs[which(adrs[,drug_col] == drug & asNum(adrs[,score_col]) > 0 ), c(adr_col, score_col)]
    if (nrow(adr_scores) == 0 ){next}
    adr_scores <- adr_scores[order(adr_scores[,2]),]
    pdf(file.path(outputDir, paste0(drug, ".pdf")), height = (nrow(adr_scores)/4), width = 8)
        par(mar=c(0,6,0,1)+0.1)
        xrange <- range(c(0,1, adr_scores[,2])) # I want to atleast be b/t 0 and 1
        yticks <- barplot(adr_scores[,2]
                , col='deepskyblue3'
                , horiz = TRUE
                , xaxt = 'n'
                , yaxt = 'n'
                , xlim = xrange
               )
        xticks <- pretty(xrange)
        axis(3, at = xticks, labels = TRUE, pos = (max(yticks) + (yticks[2,1] - yticks[1,1])))
        axis(2, at = yticks, labels = FALSE)
        text(-0.1*max(xrange),(yticks + 0.8), labels = as.character(adr_scores[,1]), srt = 0, pos = 1, xpd = TRUE)
    dev.off()
}
