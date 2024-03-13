wd <- "/home/ubuntu/2xar/twoxar-demo/web1/R"
thisScriptsPath <- wd
setwd(wd)
library(Biobase)
library(limma)
library(GEOquery)
source('supportFunctionsForAllRCode.R')
source('general_supportFunctions.R')
source('settings.R')
source('metaGEO_supportFunctions.R')
source('sigGEO_supportFunctions.R')
loadLibraries('sig')
runSVA <- F
testTheScript <- F
miRNA <- F
EntrezToUniprotMap <- '~/2xar/ws/HUMAN_9606_Protein_Entrez.tsv'
absoluteMinUniprotProportion <- 0.35
minProbePortionAgreeingOnDirection <- 0.66
permNum <- 100
gpl <- 'GPL22366' # I looked this up. It corresponds to the bead they used: Illumina HumanHT-12 V4.0 expression beadchip. Though it's an alternate, it matches the IDs we have for the probes
geoID <- fullGeoID <- 'EB_uArrays'
microarrayDir <- outdir <- pubdir <- '/mnt2/ubuntu/EB_uArrays'
get_inds <- function(toMatchWith, toMatchAgainst){
    unlist(lapply(toMatchWith, function(x) which(endsWith(toMatchAgainst, x))))
}
tisID <- 'noLesion'

setwd(microarrayDir)

idat_files <- paste(microarrayDir, list.files(path = microarrayDir, pattern = '*idat'), sep = '/')
bgx_file = paste0(microarrayDir, '/HumanHT-12_V4_0_R2_15002873_B.bgx')
x <- read.idat(idat_files, bgx_file)
x$genes$DectionPValue <- detectionPValues(x)
y <- neqc(x)
eset <- ExpressionSet(assayData = assayDataNew(exprs = y$E))

exprsEset <- exprs(eset)
filteredExprEset <- checkForMissingData(exprsEset)
filter <- rownames(exprsEset) %in% rownames(filteredExprEset)
eset <- eset[filter,]
checkTopQuantile(eset, top1PercentSignalThreshold)
needsLogging <- read.table("needsLog2.txt",
                               colClasses = "logical",
                               header = TRUE
                              )[1,1]
library(arrayQualityMetrics)
qcResults <- arrayQualityMetrics(expressionset = eset,
                                     outdir = './',
                                     force = TRUE,
                                     do.logtransform = needsLogging)
heatmapOutliers <- names(qcResults$modules$heatmap@outliers@which)
boxPlotOutliers <- names(qcResults$modules$boxplot@outliers@which)
# This results in a vector of indexes of gsms that are outliers in both measures
strictOutliers <- Reduce(intersect, list(boxPlotOutliers, heatmapOutliers))
if(length(strictOutliers) > 0){
    print('WARNING: outliers detected. Deal with that.')
    print(strictOutliers)
}

expressionData <- exprsEset
### This is where you can control which cases are used.
### noLesion_case_inds correspond to unwounded/non-lesional biopsies from EB patients
### wounded_case_inds are biopsies from open wounds from EB patients
### control_case_inds are non-wounded skin biopsies from patients without EB
conds <- read.csv(paste(microarrayDir, 'conditions_correct.csv', sep = '/'))
noLesion_case_inds <- get_inds(removeExt(conds[conds$Condition=='Normal','IDATfile']), colnames(expressionData))
wounded_case_inds <- get_inds(removeExt(conds[conds$Condition=='Wounded','IDATfile']), colnames(expressionData))
control_inds <- get_inds(removeExt(conds[conds$Condition=='Control','IDATfile']), colnames(expressionData))
case_inds <- noLesion_case_inds
columnAssignments <- list(case = colnames(expressionData)[case_inds], control = colnames(expressionData)[control_inds])

# Identify significantly differential probes
callingScoresList <- NA
x <- buildX(expressionData, columnAssignments$case, columnAssignments$cont)
setwd(wd)
tempList <- compareDEGs(runSamR(x, columnAssignments$cont, columnAssignments$case)
                                , runLimma(x, columnAssignments$cont, columnAssignments$case)
                                , runRankProd(x, columnAssignments$cont, columnAssignments$case)
                                , x
                                , columnAssignments$cont
                                , columnAssignments$case
                               )
significantProbes <- tempList$significantProbes
callingScoresList <- tempList$callingScoresList

# now get the mapping data. 
gplObj <- getGEO(gpl)
gplTable <- Table(gplObj)


#colnames(gplTable)[1] <- 'previous_ID'
#colnames(gplTable)[grep('Array_Address_Id', colnames(gplTable))] <- 'ID'

mappingInfo <- list(table =  gplTable,
                    chipType = Meta(gplObj)$title
                   )
mappingScoresList <- writeProbes(significantProbes, mappingInfo)
qcSig(columnAssignments
          , length(which(significantProbes[,"q-value"] <= 0.05)) / nrow(significantProbes)
          , length(which(significantProbes[,"q-value"] == 0.0)) / nrow(significantProbes)
          , mappingScoresList
          , callingScoresList
         )
