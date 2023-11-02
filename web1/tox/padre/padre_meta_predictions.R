#=============================
# Get set up
#=============================
library(caret)
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
if (length(args) < 4) {
  quit('no', as.numeric(exitCodes['usageError']))
}
converter <- read.table(args[1], header=F, sep="\t",stringsAsFactors=F)
realADRs <- read.table(args[2], header = T, sep="\t", stringsAsFactors=F)
predADRs <- read.table(args[3], header = T, sep="\t", stringsAsFactors=F)
totalADRs <- asNum(args[4])
minPredVal <- asNum(args[5])
outputFH <- args[6]

minRealVal <- 0

#===============================================================================================
# Run through each iteration and pull out the relevant information as well as quality metrics
#===============================================================================================
allRes <- list()
for (x in 1:nrow(converter)){
    stitchID <- converter[x,2]
    dbID <- converter[x,1]
    real <- realADRs[which(realADRs$stitch_id == stitchID), ]
    # if we don't have values for the real ADRs, we'll just go with binary (1's for having the ADR)
    if (ncol(real) < 3){
        real$flag <- rep(1, nrow(real))
        real_metric_col <- 'flag'
    }else{
        real_metric_col <- colnames(real)[3]
        real[,3] <- asNum(real[,3])
    }
    pred <- predADRs[which(predADRs$drug == dbID),]
    together <- merge(real[,-1*grep('stitch_id', colnames(real))], pred, by.x='umls_id', by.y='adr', all=T)
    together$probability <- asNum(together$probability)
    together[is.na(together)] <- 0
    truefalse <- together[,c(real_metric_col, 'probability')]
    truefalse[,1] <- truefalse[,1] > minRealVal
    truefalse[,2] <- truefalse[,2] >= minPredVal
    truefalse <- truefalse*1 # convert to numeric
    colnames(truefalse) <- c('v1', 'v2')
    filler <- data.frame(v1=rep(0, totalADRs - nrow(truefalse)), v2=rep(0, totalADRs - nrow(truefalse)))
    fortest <- rbind(truefalse, filler)
    fortest[,1] <- factor(fortest[,1], levels=0:1)
    fortest[,2] <- factor(fortest[,2], levels=0:1)
    res <- confusionMatrix(fortest[,2], fortest[,1], positive = '1')
    vals <- cbind(t(res$overall),t(res$byClass))
    print(dbID)
    print(res)
#    toreport <- data.frame(names=colnames(vals), values = vals[1,])
#    write.table(toreport, file=paste(outputFH, dbID, "predicted_vs_real_SEs_stats.tsv", sep="_"), quote=F, row.names=F, sep="\t", col.names=F)
    allRes[[x]] <- data.frame(names=colnames(vals), values = vals[1,])
}

allVals <- list()
for (i in 1:length(allRes)){
    res <- allRes[[i]]$values
    names(res) <- allRes[[i]]$names
    allVals[[i]] <- list(accuracy = asNum(res['Accuracy']), balanced_accuracy = asNum(res['Balanced Accuracy'])
                                , NegLog10_accPval = -1*log10(asNum(res['AccuracyPValue']))
                                , sensitivity = asNum(res['Sensitivity']), specificity = asNum(res['Specificity'])
                                )
}
# for each measurement type
for (j in 1:length(allVals[[1]])){
    toPlot <- list()
    for (k in 1:length(allVals)){
        toPlot[[k]] <- allVals[[k]][[j]]
    }
    pdf(paste0(outputFH, "_", names(allVals[[1]])[[j]], "_asFuncOf", ".pdf"))
        boxplot(unlist(toPlot), outpch = NA
#                , names = converter[,1]
                , main = names(allVals[[1]])[[j]]
                , ylim = c(0, 1)
                , notch = TRUE
                )
        stripchart(unlist(toPlot), vertical = TRUE, method = "jitter", pch = 21, col = "maroon", bg = "bisque", add = TRUE) 
    dev.off()
}
