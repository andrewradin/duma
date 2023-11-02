#=============================
# Get set up
#=============================
#library(matrixStats)
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

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  quit('no', as.numeric(exitCodes['usageError']))
}else if(length(args) == 3){
    paramType = args[3]
}else{
    paramType = ""
}
dataDir <- args[1]
iterNum <- as.numeric(args[2])

setwd(dataDir)

# get the directories in predictions, each one should be its own value of the value which was searched over 
# e.g. minQ or min number of paths
subDirs <- dir()[file.info(dir())$isdir]

#===============================================================================================
# Run through each iteration and pull out the relevant information as well as quality metrics
#===============================================================================================
allVals <- allRocs <- allAucs <- list()
for (x in 1:length(subDirs)){
    accuracy <- balanced_accuracy <- NegLog10_accPval <- sensitivity <- specificity <- roc <- auc <- list()
    subdir <- subDirs[x]
    for (i in 1:iterNum){
        if (file.exists(paste0("./", subdir, "/", i, '_predictions_predicted_vs_real_SEs_stats.tsv'))){
            res <- read.table(paste0("./", subdir, "/", i, '_predictions_predicted_vs_real_SEs_stats.tsv'), sep="\t", header=F, row.names=1)
            accuracy[[length(accuracy) + 1]] <- asNum(res['Accuracy',1])
            balanced_accuracy[[length(balanced_accuracy) + 1]] <- asNum(res['Balanced Accuracy',1])
            NegLog10_accPval[[length(NegLog10_accPval) + 1]] <- -1*log10(asNum(res['AccuracyPValue',1]))
            sensitivity[[length(sensitivity) + 1]] <- asNum(res['Sensitivity',1])
            specificity[[length(specificity) + 1]] <- asNum(res['Specificity',1])
            # also get the ROC and AUC from ROCR
 #           temp <- readRDS(paste0("./", subdir, "/", i, "_rocAUC.rds"))
 #           roc[[length(roc) + 1 ]] <- asNum(temp$roc)
 #           auc[[length(auc) + 1 ]] <- asNum(temp$auc)
        }
    }
    allVals[[x]] <- list(accuracy = unlist(accuracy), balanced_accuracy = unlist(balanced_accuracy)
                                , NegLog10_accPval = unlist(NegLog10_accPval)
                                , sensitivity = unlist(sensitivity), specificity = unlist(specificity)
                                )
  #  allRocs[[x]] <- rowMedians(combinedRocs)
  #  allAucs[[x]] <- median(unlist(auc))
}

# for each measurement type
for (j in 1:length(allVals[[1]])){
    toPlot <- list()
    for (k in 1:length(allVals)){
        toPlot[[k]] <- allVals[[k]][[j]]
    }
    
    pdf(paste0(names(allVals[[1]])[[j]], "_asFuncOf_", paramType, ".pdf"))
        boxplot(toPlot, outpch = NA
                , names = subDirs
                , main = names(allVals[[1]])[[j]]
                , ylim = c(0, max(c(1, unlist(toPlot))))
                , notch = TRUE
                , xlab = paramType
                )
        stripchart(toPlot, vertical = TRUE, method = "jitter", pch = 21, col = "maroon", bg = "bisque", add = TRUE) 
    dev.off()
}
#pdf(paste0('roc_auc_asFuncOf_', paramType, '.pdf'))
#  plot(rowMedians(combinedRocs), main='Receiver Operating Characteristic', xlim=c(0,1), ylim=c(0,1))
#  abline(0,1)
#  legend(0.7, 0.3, paste("Median AUC:", round(median(unlist(aucs)))), bty='n')
