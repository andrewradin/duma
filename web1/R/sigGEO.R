#====================================
# Load settings and generally set-up
#====================================
# Contains a few functions used for setting up, as well as loading necessary settings
getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl = TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) return('None')
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}

thisScriptsPath <- getScriptPath()
if(thisScriptsPath != 'None') source(paste0(thisScriptsPath, '/sigGEO_supportFunctions.R'))

# read and process command line arguments, and set up the log directory
argsList<-parseCommands('sigGEO.R')
fullGeoID <- argsList$fullGeoID
geoID <- argsList$geoID
tisID <- argsList$tisID
idType <- argsList$idType
useBothColors <- argsList$useBothColors
microarrayDir <- argsList$microarrayDir
indir <- argsList$indir
outdir <- argsList$outdir
pubdir <- argsList$pubdir
testTheScript <- argsList$testTheScript
customSettings <- paste(indir, 'settings.R', sep = '/')
if( file.exists(customSettings)) source(customSettings)

#====================================
# Load the necessary data
#====================================

# RNA-seq experiments get handled differently, so first we look for them.
if(grepl("-seq", idType)){
    if(file.exists(paste0(microarrayDir, "/rnaSeqAnalysis/uniprot_expression_data.rds"))){
        print("Analyzing RNA-seq data")
        loadLibraries('seq')
        allUniprots <- analyzeAndReportRNASeqData()
    }else{
        print("WARNING: metaGEO.R has not been run yet.  Retry sigGEO.R after running metaGEO.R")
        quit('no', as.numeric(exitCodes['usageError']))
    }
}else{

    # Load the libraries we need
    loadLibraries('sig')
    # This is used later if there are issues identifying the relevant columns in the expression data
    originalColNames <- NA
    
    if(file.exists(paste0(microarrayDir, "/", geoID, "_normalized.rds"))){

        # Load the normalized data
        eset <- readRDS(paste0(microarrayDir, "/", geoID, "_normalized.rds"))
        # and get just the expression information
        expressionData <- exprs(eset)
        if(debug){print(colnames(expressionData))}
        
        #=========================================================================================
        # Clean up the column names
        #=========================================================================================
        # In the case of illumina beadChips, the column names are not the proper GSMs,
        # so let's take care of that
        if(file.exists(paste0(microarrayDir,"/illuminaGsmConverter.rds"))){
            expressionData <- cleanupIlluminaHeaders(expressionData)
            # this doesn't always work, so I want to check and see if it did
            # but I'll do that once I have the column assignments below.
            # if it doesn't I'll try to parse the metaData file
        }
        # In some cases the column names have the GSMs plus some trailing information
        # that muddles merging.
        # But some of these also need the trailing prefixes
        # So I'll save the original column names, and if it doesn't work I'll bring them back
        if(! useBothColors){
            originalColNames <- colnames(expressionData)
            if(idType!='ae'){
                toBeColNames <- just_gsms(colnames(expressionData))
            }
        }
    }else if(file.exists(paste0(microarrayDir,'/',geoID,"_asFromGeo.rds"))){
        print("Attempting to analyse processed data, using the previous sigGEO approach")
        expressionData <- readRDS(paste0(microarrayDir,'/',geoID,"_asFromGeo.rds"))
        # I ran into a few files that were saving as characters, this addresses that
        for(col in 1:ncol(expressionData)){ expressionData[,col] <- as.numeric(expressionData[,col])}
        
    }else{
        print("WARNING: metaGEO.R has not been run yet.  Retry sigGEO.R after running metaGEO.R")
        quit('no', as.numeric(exitCodes['usageError']))
    }

    # Pull down the control and case information - tissue specific
    l <- checkColAssigns(parseDataColumns(), expressionData, idType)
    columnAssignments <- l$ca
    expressionData <- l$ed

    # Identify significantly differential probes
    callingScoresList <- NA
    x <- buildX(expressionData, columnAssignments$case, columnAssignments$cont)
    if(algoToUse == 'RP'){
         print("Using only RankProd for differential expression detection")
        significantProbes <- runRankProd(x,columnAssignments$cont,columnAssignments$case)
    }else if(algoToUse == 'SAM'){
         print("Using only SAMR for differential expression detection")
        significantProbes <- runSamR(x,columnAssignments$cont,columnAssignments$case)
    }else if(algoToUse == 'Limma'){
        print("Using only Limma for differential expression detection")
        significantProbes <- runLimma(x,columnAssignments$cont,columnAssignments$case)
        pdf(paste0(pubdir, "/", geoID, "_", tisID, "_sigPCAMDS.pdf"))
        templist <- cleanUpSigProbeDFs(significantProbes,
                                       'Limma',
                                        x,
                                        columnAssignments$case,
                                        columnAssignments$cont,
                                        calcScore = TRUE
                                      )
        dev.off()
        callingScoresList <- list(ccCorScore = templist$ccCorScore, concordScore = NA)
    }else if(algoToUse == 'all'){
        print("Using SAMR, RankProd, Limma and GeoDE for differential expression detection")
        tempList <- compareDEGs(runSamR(x, columnAssignments$cont, columnAssignments$case)
                                , runLimma(x, columnAssignments$cont, columnAssignments$case)
                                , runRankProd(x, columnAssignments$cont, columnAssignments$case)
                                , x
                                , columnAssignments$cont
                                , columnAssignments$case
                               )
        significantProbes <- tempList$significantProbes
        callingScoresList <- tempList$callingScoresList
    }else if(algoToUse == 'GeoDE'){
        print("Using geoDE for differential expression detection")
        significantProbes <- runGeoDE(x, columnAssignments$cont,columnAssignments$case) 
        # the fold change isn't really a FC per se with this method,
        # and instead is something unique to the method
        # thus we don't want to filter anything else,
        # so we'll set the method to the filter value that was already used
        #minFC <- minChdirScore
        # I ended up just changing the filtering when GeoDE is used
    }else{
        print(paste("WARNING: Inappropriate value for algoToUse (in settings.R),",
                    "this will cause no genes to be reported."
                  ))
        quit('no', as.numeric(exitCodes['usageError']))
    }
    # For de-bugging
    if(debug){write.table(significantProbes,
                          file = paste0(outdir,
                                        "/",
                                        geoID,
                                        '_significantProbes.tsv'
                           ), 
                           sep = '\t',
                          quote = FALSE
                         )
              }

    # And concert the probes to proteins
    mappingInfo <- readRDS(file = paste0(microarrayDir, "/", geoID, "_forMapping.rds"))        
    mappingScoresList <- writeProbes(significantProbes, mappingInfo)
    qcSig(columnAssignments
          , length(which(significantProbes[,"q-value"] <= 0.05)) / nrow(significantProbes)
          , length(which(significantProbes[,"q-value"] == 0.0)) / nrow(significantProbes)
          , mappingScoresList
          , callingScoresList
         )
}

finishedSuccessfully()
