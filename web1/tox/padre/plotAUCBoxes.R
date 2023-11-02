library(dplyr)
#=============================
# Get set up
#=============================
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
if (length(args) < 1) {
  quit('no', as.numeric(exitCodes['usageError']))
}
infile <- args[1]
indata <- read.table(infile, header=F, sep="\t", stringsAsFactor=F)
colnames(indata) <- c('adr', 'roc', 'pr')
indata$roc <- asNum(indata$roc)
indata$pr <- asNum(indata$pr)
adr_meds <- data.frame(indata %>% group_by(adr) %>% summarise(roc_med=median(roc, na.rm = T), pr_med=median(pr, na.rm = T)))

# for each measurement type
pdf("aucBoxplots.pdf")
    boxplot(list(indata$roc, indata$pr)
            , outpch = '.'
            , names = c('ROC', 'Precision-Recall')
            , ylim = c(0,1)
            , notch = TRUE
            , ylab = 'AUC'
            )
    hist(indata$roc, breaks=100, main='ROC-AUC histogram')
    hist(indata$pr, breaks=100, main='PR-AUC histogram')
dev.off()

pdf("groupedADR_aucBoxplots.pdf")
    boxplot(list(adr_meds$roc_med, adr_meds$pr_med)
            , outpch = '.'
            , names = c('ROC', 'Precision-Recall')
            , ylim = c(0,1)
            , notch = TRUE
            , ylab = 'AUC'
            )
    hist(adr_meds$roc_med, breaks=100, main='Grouped ADRs, ROC-AUC')
    hist(adr_meds$pr_med, breaks=100, main='Grouped ADRs, PR-AUC')
dev.off()

