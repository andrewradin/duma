#======================================================================================================================
# set up
#======================================================================================================================
library(scales)
library(rgeos)
getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl = TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) stop("can't determine script dir: please call the script with Rscript")
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}
checkArgAndReturnBool <- function(argumentStr){
    if(! grepl('true', argumentStr, ignore.case = TRUE) & ! grepl('false', argumentStr, ignore.case = TRUE)){
        warning("deaPlotter.R <drugScores.txt> <RES.txt> <matches.txt> <outputFH> <plot small, TRUE or FLASE>")
        quit('no', as.numeric(exitCodes['usageError']))
    }else if(grepl('true', argumentStr, ignore.case = TRUE)){
        return(TRUE)
    }
    return(FALSE)
}

thisScriptsPath <- getScriptPath()
source(paste0(thisScriptsPath, '/../supportFunctionsForAllRCode.R'))

#======================================================================================================================
# settings
#======================================================================================================================
underColor <- "dodgerblue3"
resColor <- 'red'
bgColor <- 'lightpink'
polygonColor <- 'green'

args <- commandArgs(trailingOnly = TRUE)

#======================================================================================================================
# functions
#======================================================================================================================
plot_res <- function(RES, bg_score, resColor, bgColor, xaxis){
    ylim <- c(-1, 1)
    plot(xaxis, RES
        , type = 'l'
        , col = resColor
        , yaxt = 'n'
        , ylab = ''
        , lwd = 2
        , ylim = ylim
        , xlab = "Drug List Index"
        , main = 'Drug Enrichment Analysis'
        )
    yax <- axis(2, col = resColor, labels = FALSE)
    text(x = -1*length(xaxis)/20, y = yax, labels = yax, pos = 2, xpd = TRUE, col = resColor)
    mtext("Running Enrichment Score", side = 2, line = 3, col = resColor, font = 2)
    abline(h = 0, lty = 2, col = 'black')
    abline(h = asNum(bg_score), col = bgColor)
    return(ylim)
}
plot_pg_area <- function(lastPeak, xaxis, RES, polygonColor){
    # a bar at 0-1, to support the situation where the peak is at index 1
    yax_res_peak_plygon_x <- c(0, xaxis[1:lastPeak], rev(xaxis[1:lastPeak]), 0)
    yax_res_peak_plygon_y <- c(0, RES[1:lastPeak], rep(1, lastPeak), 1)
    polygon(yax_res_peak_plygon_x, yax_res_peak_plygon_y, col = polygonColor, border = NA)
}
plot_drug_scores <- function(drugScores, xaxis, ylim, underColor){
    minVal <- min(c(0, min(drugScores))) # a min value of at least 0
    # the weird yaxis is just for asthetics, I want the drug score to be in the bottom half of the plot
    maxVal <- 2 * ylim[2] * (max(drugScores) - minVal)
    if (! is.finite(minVal) || ! is.finite(maxVal)){
        print(paste("Issue with ylim values:", minVal, maxVal, ylim, max(drugScores)))
    }
    plot(xaxis, drugScores
        , type = "l"
        , col = underColor
        , xaxt = "n"
        , yaxt = "n"
        , xlab = ""
        , ylab = ""
        , ylim = c(minVal,maxVal)
    )
    bottom_at <- pretty(drugScores, n = 3) # default is n=5, but that looked to cluttered
    yax <- axis(4, labels = FALSE, col = underColor, at = bottom_at)
    text(x = 1.05*length(xaxis), y = yax, labels = yax, pos = 4, xpd = TRUE, col = underColor)
    mtext("Drug scores", side = 4, line = 3, col = underColor, font = 2)
    polygon(c(xaxis, rev(xaxis))
           , c(rep(0, length(drugScores))
           , rev(drugScores))
           , col = underColor
           , border = NA
           )
}
plot_matches <- function(matches, max_drug_score){
    for (i in 1:length(matches)){
        if(matches[i] == 1){
            segments( i
                    , max_drug_score / 2 - max_drug_score / 15
                    , i
                    , max_drug_score / 2 + max_drug_score / 15
                    , col = 'black'
                    )
        }
    }
}
add_legend <- function(pg_area, matches, lastPeak, leadingEdgeCnt, used_ES, bg_score, nes, drugScores, pval){
    legend('topright', paste( paste("Norm'd polygon area:", round(pg_area, 2)) 
                            , paste("Peak at:", lastPeak)
                            , paste("Leading edge count:", leadingEdgeCnt)
                            , paste("Drugs evaluated:", sum(matches))
                            , paste("Enrichment Score:", round(used_ES, 3))
                            , paste("Background score:", round(asNum(bg_score), 3))
                            , paste("NES:", nes[1])
                            , pval
                            , paste("Total drugs analyzed:", length(drugScores))
                            , sep = "\n"
                            )
            , bty = 'n', inset = c(0,-0.15)
            )
}
add_limited_legend <- function(pg_area, matches, lastPeak, leadingEdgeCnt, used_ES, drugScores){
    legend('topright', paste( paste("Norm'd polygon area:", round(pg_area, 2)) 
                            , paste("Peak at:", lastPeak)
                            , paste("Leading edge count:", leadingEdgeCnt)
                            , paste("Drugs evaluated:", sum(matches))
                            , paste("Enrichment Score:", round(used_ES, 3))
                            , paste("Total drugs analyzed:", length(drugScores))
                            , sep = "\n"
                            )
            , bty = 'n', inset = c(0,-0.1)
            )
}

#======================================================================================================================
# read input
#======================================================================================================================

if (length(args) < 5) {
    warning("deaPlotter.R <drugScores.txt> <RES.txt> <matches.txt>  <bootstrapStats.txt> <outputFH> <ChiSq enrichment for any score, boolean> <plot png, boolean>")
    quit('no', as.numeric(exitCodes['usageError'])) 
}
if (length(args) > 5) {
    calcEnrich <- checkArgAndReturnBool(args[6])
}else{
    calcEnrich <- FALSE
}
if (length(args) > 6) {
    plotSmall <- checkArgAndReturnBool(args[7])
}else{
    plotSmall <- FALSE
}

drugScores <- read.csv(args[1], header = FALSE)[,1]
RES <- read.csv(args[2], header = FALSE)[,1]
matches <- read.csv(args[3], header = FALSE)[,1]
bootStuff <- read.csv(args[4], header = FALSE)[,1]
nes <- bootStuff[1]
pval <- bootStuff[2]
bg_score <- bootStuff[3] 
leadingEdgeCnt <- asNum(bootStuff[(length(bootStuff)-2)])
used_ES <- asNum(bootStuff[length(bootStuff)-1])
pg_area <- asNum(bootStuff[length(bootStuff)])
#======================================================================================================================
# plot
#======================================================================================================================

if (plotSmall){
    png(paste0(args[5], "_DEAPlots.png"))
}else{
    pdf(paste0(args[5], "_DEAPlots.pdf"))
}
    par(mar = rep(4.1, 4))
    par(new = FALSE)
    xaxis <- seq(1, length(RES))
    ylim <- plot_res(RES, bg_score, resColor, bgColor, xaxis)
    lastPeak <- tail(which(RES == used_ES), 1)
    plot_pg_area(lastPeak, xaxis, RES, polygonColor)
    par(new = TRUE)
    plot_drug_scores(drugScores, xaxis, ylim, underColor)
    max_drug_score <- max(abs(drugScores))
    plot_matches(matches, max_drug_score)
    abline(v = lastPeak, lty = 3, col = resColor, lwd = 2)
    add_legend(pg_area, matches, lastPeak, leadingEdgeCnt, used_ES, bg_score, nes, drugScores, pval)
    # now add a new plot: a zoom in to only those drugs with scores
    par(mar = rep(4.1, 4))
    par(new = FALSE)
    far_right <- min(length(drugScores), max(50, floor(lastPeak * 1.1))) # if the peak is within the top 50, zoom out at least 50
    xaxis <- seq(1, far_right)
    ylim <- plot_res(RES[1:far_right], bg_score, resColor, bgColor, xaxis)
    plot_pg_area(lastPeak, xaxis, RES, polygonColor)
    par(new = TRUE)
    plot_drug_scores(drugScores[1:far_right], xaxis, ylim, underColor)
    plot_matches(matches, max_drug_score)
    abline(v = lastPeak, lty = 3, col = resColor, lwd = 2)
    add_limited_legend(pg_area, matches, lastPeak, leadingEdgeCnt, used_ES, drugScores)
dev.off()

if (calcEnrich){
    # I'll add the ability to calculate enrichment, this was specifically written for Nigams work
    together <- data.frame(drugScores = drugScores > 0, matches = matches)
    tbl <- table(together)
    enrich <- chisq.test(tbl)
    write.table(data.frame(unlist(enrich)), file = paste0(args[4], "_DEA_enrichment.txt"), quote = FALSE)
}
