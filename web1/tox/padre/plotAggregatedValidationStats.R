#=============================
# Get set up
#=============================
library(RColorBrewer)
library(scales)

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
prefix <- strsplit(basename(infile), "\\.")[[1]][1]
indata <- read.table(infile, header=T, sep="\t", stringsAsFactor=F)

cutoffs <- unique(asNum(indata$cutoff))
cutoffs <- cutoffs[order(cutoffs, decreasing=F)]
iterations <- unique(asNum(indata$iteration))
iterations <- iterations[order(iterations, decreasing=F)]
toplot <- vector(mode='list', length=length(cutoffs))
stripData <- vector(mode='list', length=length(iterations))
for (i in iterations){
    stripData[[i]] <- vector(mode='list', length=length(cutoffs))
}
for (i in 1:nrow(indata)){
    j <- grep(paste0("^", indata[i, 'cutoff'], "$"), cutoffs)
    toplot[[j]] <- c(toplot[[j]], indata[i, 'value'])
    k <- grep(paste0("^", indata[i, 'iteration'], "$"), iterations)
    stripData[[k]][[j]] <- c(stripData[[k]][[j]], indata[i, 'value'])
}

strip_median <- stripMin <- stripMax<- stripData
for (i in 1:length(stripData)){
    for (j in 1:length(cutoffs)){
        strip_median[[i]][[j]] <- median(stripData[[i]][[j]])
        stripMin[[i]][[j]] <- min(stripData[[i]][[j]])
        stripMax[[i]][[j]] <- max(stripData[[i]][[j]])
    }
}

n <- length(iterations)
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
mycolors = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

# for each measurement type
pdf(paste0(prefix, "_asFuncOf_predictedProb.pdf"))
    par(xpd=T, mar=c(4.1, 4.1, 4.1, 8.1))
    boxplot(toplot
            , outpch = NA
            , names = cutoffs
            , ylim = range(c(0,1, unlist(toplot)))
            , notch = TRUE
            , ylab = prefix
            , xlab = "Predicted probability"
            , bty='L'
            , col = 'black'
            )
    for (i in 1:length(iterations)){
        stripchart(strip_median[[i]], vertical = TRUE, method = "jitter", jitter=0.2, pch = 20, col = mycolors[i], add = TRUE, bty='L')
        stripchart(stripMin[[i]], vertical = TRUE, method = "jitter", jitter=0.2, pch = 20, col = alpha(mycolors[i], 0.3), add = TRUE, bty='L')
        stripchart(stripMax[[i]], vertical = TRUE, method = "jitter", jitter=0.2, pch = 20, col = alpha(mycolors[i], 0.3), add = TRUE, bty='L')
    }
    legend(length(cutoffs)+1, 1.1, paste("CV-fold", iterations), fill=mycolors)
dev.off()
