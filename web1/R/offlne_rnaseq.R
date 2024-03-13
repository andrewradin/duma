getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) return('None')
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}

thisScriptsPath <- getScriptPath()
if(thisScriptsPath != 'None') source(paste0(thisScriptsPath, '/sigGEO_supportFunctions.R'))

args <- commandArgs(trailingOnly = TRUE)
if ( length(args) < 5){
    stop('USEAGE: Rscript offlne_rnaseq.R <directory to read and write files from/to> <input RDS file name> <case_searchTerm> <cont_searchTerm> <disease or dataset name>')
}

source('sigGEO_supportFunctions.R')
source('sigGEO_rnaSeq.R')
source('general_supportFunctions.R')
source('supportFunctionsForAllRCode.R')
library("pheatmap")
debug <- F
minLogVal <- 0
tisID <- 'offline'

pubdir <- args[1]
ifile <- args[2]
case_searchTerm <- args[3]
cont_searchTerm <- args[4]
geoID <- args[5]
databaseTable <- paste(geoID, tisID, 'rnaseq_results', sep = '_')
outdir <- pubdir

txi <- readRDS(paste(pubdir, ifile, sep = '/'))
countsData <- txi$counts
countsColNames <- colnames(txi$counts)
columnAssignments <- list(case = countsColNames[grep(case_searchTerm, countsColNames)],
                          cont = countsColNames[grep(cont_searchTerm, countsColNames)]
                         )

x <- buildRNASeqX(txi$counts, columnAssignments$case, columnAssignments$cont)
x <- filterNonexpressedGenes(x, length(columnAssignments$cont))
xForPlotting <- thoroughLog2ing(x)
plotDiagnostics(xForPlotting, columnAssignments$cont, columnAssignments$case)
significantUniprots <- run_edgeR_tximport(x, txi$length, columnAssignments)
if (nrow(significantUniprots)>0){
    toPrintOut <- as.data.frame(matrix(nrow = nrow(significantUniprots), ncol = 7))
    toPrintOut[,1] <- NULL
    toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
    # this isn't really applicable for RNA-seq
    toPrintOut[,3] <- '1;1'
    toPrintOut[,4] <- formatC((1 - asNum(significantUniprots[,"q-value"])),
                               format = 'e',
                               digits = 3
                              )
    toPrintOut[,5] <- significantUniprots[,"direction"]
    toPrintOut[,6] <- tisID
    toPrintOut[,7] <- significantUniprots[,"Fold Change"]
    outFile <- paste0(pubdir, "/", databaseTable, ".tsv")
    write.table(toPrintOut, file = outFile, quote = FALSE,
                row.names = FALSE, col.names = FALSE, sep = "\t")
}else{
    print(sprintf("%s has no significant expression", geoID))
}

pdf(paste0(pubdir, "/", geoID, "_", tisID, "_sigPCAMDS.pdf"))
        templist <- cleanUpSigProbeDFs(significantUniprots
                                       , 'RNAseq'
                                       , xForPlotting
                                       , columnAssignments$case
                                       , columnAssignments$cont
                                       , calcScore = TRUE
                                       , geneColName = "UNIPROTKB"
                                      )
dev.off()
qcSig(columnAssignments
          , length(which(significantUniprots[,"q-value"] <= 0.05)) / nrow(significantUniprots)
          , length(which(significantUniprots[,"q-value"] == 0)) / nrow(significantUniprots)
          , list(consistDirScore = 1.0
                 , mappingScore = 1.0
                 )
          , list(ccCorScore = templist$ccCorScore, concordScore = 1.0)
         )
