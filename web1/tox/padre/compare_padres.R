#======================================================================================================================
# set up
#======================================================================================================================
library(scales)
library(caret)
#library(ROCR)
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

#======================================================================================================================
# read input
#======================================================================================================================

if (length(args) < 3) {
    warning("compare_sepps.R <SE_predictions> <real_SE> <outputFH>")
    quit('no', as.numeric(exitCodes['usageError']))
}
predSEs <- read.table(args[1], header=FALSE, sep="\t")
realSEs <- read.table(args[2], header=TRUE, sep="\t")

if (length(unique(predSEs[,1])) > 1 || length(unique(realSEs[,1])) > 1){
    print("This only works with a single drug at a time. Quitting.")
    quit('n')
}

outputFH <- args[3]
totalSEs <- as.numeric(args[4])

useable_realSE <- realSEs[,2:3]
useable_realSE[useable_realSE[,2] == 'inf', 2] <- '10'
useable_realSE[,2] <- asNum(useable_realSE [,2])
useable_realSE <- useable_realSE[which(useable_realSE[,2] > 0),]
useable_pred <- predSEs[, 2:3] 
useable_pred[,2] <- asNum(useable_pred[,2])
# merge on the SE name
real_pred <- merge(useable_pred, useable_realSE, by=1, all=TRUE)
# pull out all this info to make a confusion matrix and generate stats of how we did
#  fp <- length(which(is.na(real_pred[,2]) & ! is.na(real_pred[,3])))
#  fn <- length(which(is.na(real_pred[,3]) & ! is.na(real_pred[,2])))
#  tp <- nrow(real_pred) - fp - fn
#  tn <- totalSEs - nrow(real_pred)
real_pred[,2] <- asNum(real_pred[,2])
real_pred[,3] <- asNum(real_pred[,3])
real_pred[is.na(real_pred)] <- 0

#pdf(paste(outputFH, "predicted_vs_real_SEs.pdf", sep="_"))
#plot(real_pred[,c(3,2)]
#    , pch=20
#    , col=alpha('darkblue', 0.5)
#    , xlab="Side effect prevalence"
#    , ylab="Predicted side effect score"
#    , main=predSEs[1,1]
#)
#corval <- cor(real_pred[,c(3,2)], method='spearman')
#legend('topright', paste("Spearman Rho:", as.character(signif(corval[1,2], 3)), sep="\n"))
#legend('right', paste(paste("TP:",tp), paste('FP:', fp), paste('FN:', fn), paste('TN:', tn), sep="\n"))
#dev.off()

# Now we calculate the stats, which are probably more relevant
truefalse <- real_pred[,2:3] > 0
truefalse <- truefalse*1 # convert to numeric
colnames(truefalse) <- c('v1', 'v2')
filler <- data.frame(v1=rep(0, totalSEs - nrow(truefalse)), v2=rep(0, totalSEs - nrow(truefalse)))
fortest <- rbind(truefalse, filler)
fortest[,1] <- as.factor(fortest[,1])
fortest[,2] <- as.factor(fortest[,2])
res <- confusionMatrix(fortest[,2], fortest[,1], positive = '1')
vals <- cbind(t(res$overall),t(res$byClass))
toreport <- data.frame(names=colnames(vals), values = vals[1,])
write.table(toreport, file=paste(outputFH, "predicted_vs_real_SEs_stats.tsv", sep="_"), quote=F, row.names=F, sep="\t", col.names=F)
# also use rocr for a roc curve: 
#pred <- prediction(c(real_pred[,2], rep(0, totalSEs - nrow(truefalse))), fortest[,2])
#roc <-performance(pred, measure='tpr',x.measure='fpr')
#auc <- performance(pred, measure='auc')
#saveRDS(list(roc=roc, auc=auc@y.values[[1]]), file=paste(outputFH, "rocAUC.rds", sep="_"))
