#=============================
# Get set up
#=============================
#library(RColorBrewer)
library(scales)

getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) stop("can't determine script dir: please call the script with Rscript")
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}
avg_csv <- function(x){
    mean(as.numeric(unlist(strsplit(x, ","))))
}

thisScriptsPath <- getScriptPath()
source(paste0(thisScriptsPath, './../R/supportFunctionsForAllRCode.R'))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  quit('no', as.numeric(exitCodes['usageError']))
}

indata <- read.table(args[1], header=T, sep="\t", row.names=1)
plotData <- indata[, -1 * c(grep('^n$', colnames(indata)))]
x <- indata$n

pdf(args[2])
for (i in 1:ncol(plotData)){
    # if this is a comma separated list, I'll need to separate them and take the average
    if (any(grepl(",", plotData[,i]))){
        y_list <- as.character(plotData[,i])
        y <- unlist(lapply(y_list, FUN = avg_csv))
    }else{
        y <- asNum(plotData[,i])
    }
    plot(x,y, pch = 20, col = alpha('darkblue', 0.4), main = colnames(plotData)[i])
    plotData[,i] <- y
}
boxplot(plotData, names = colnames(plotData), notch = T, ylim=c(0,1))
dev.off()

