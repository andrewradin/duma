#====================================
# Load settings and generally set-up
#====================================
# Contains a few functions used for setting up, as well as loading necessary settings
getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) return('None')
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}

thisScriptsPath <- getScriptPath()
if (thisScriptsPath != 'None') source(paste0(thisScriptsPath, '/metaGEO_supportFunctions.R'))

# read and process command line arguments, and set up the log directory
argsList<-parseCommands('metaGEO.R')
geoID <- argsList$geoID
fullGeoID <- argsList$fullGeoID
tisID <- argsList$tisID
preferredGpl <- argsList$preferredGpl
idType <- argsList$idType
useBothColors <- argsList$useBothColors
microarrayDir <- argsList$microarrayDir
indir <- argsList$indir
customSettings <- paste(indir, 'settings.R', sep = '/')
if( file.exists(customSettings)) source(customSettings)

# starting with plat319 we are now able to combine geoIDs
# but that means we need to parse them out as for
# some functions we are leaving them as a single comma separated string
idVec <- geoID
if(grepl(",", geoID)){
    idVec <- strsplit(geoID, ",")[[1]]
}

#====================================
# Data from GEO
#====================================
if(grepl('geo', idType)){
    library(GEOquery)
    options('download.file.method.GEOquery' = 'wget')
#
# This is for microarrays where, for whatever reason, we can't get or process the raw data
#
    if(idType == "geo-orig"){
    # First we ensure that this has not already been successfully run
        if(! file.exists(paste0(microarrayDir, '/', geoID, "_asFromGeo.rds"))){
            print("Running meta using the data directly from geo, instead of the raw data and normalizing it")

            # load necessary libraries
            loadLibraries('meta')

            # Retrieve meta data including mapping information
            accessAndSaveMetaData(geoID, idType, idVec, preferredGpl, fullGeoID)
            # pull down data sets from geo and annotation files
            eset <- oldVersionToEsetAndSaveData(geoID, idVec, preferredGpl)
            
            setwd(microarrayDir)
            # we don't actually save the eset in this case; it is more about looking for outliers
            processAndSaveEset(idType, eset, geoID, fullGeoID)
        }else{
            alreadyDone()
        }
#
# This is what we prefer: microarray data where we can get the raw data and normalize it ourselves
#
    }else if(idType=='geo' || idType=='geo-mixed'){
        if(! file.exists(paste0(microarrayDir,"/",geoID,"_normalized.rds"))){
            # load necessary libraries
            loadLibraries('meta')
            
            print("Retrieving raw data from GEO")
            # Retrieve meta data including mapping information
            metaDataRelated <- accessAndSaveMetaData(geoID, idType, idVec, preferredGpl, fullGeoID)
            # now that we support grouped geoIDs, this may not be a single ID, but a vector of them
            gseIDVec <- metaDataRelated$gseIDVec
            chipType <- metaDataRelated$chipType
            gseList <- metaDataRelated$gseList
            gsmsToUseList <- metaDataRelated$gsmsToUseList
            gplName <- metaDataRelated$gplName
             
            print("Downloading gse data")
            # Pull down the raw data
            # This shouldn't need to happen, but we were getting issues
            for(gseID in gseIDVec){
                system(paste0('mkdir -p ', microarrayDir,'/', gseID))
                got <- NULL
                tries <- 0
                while(class(got) != 'data.frame' & (tries < 2)){
                got <- tryCatch(
                           getGEOSuppFiles(gseID, baseDir = microarrayDir),
                           error=function(e){Sys.sleep(5); print(paste("Retrying getGEOSuppFiles for the", tries, "times")); return(e)}
                       )
                tries <- tries + 1
               }
               if (class(got) != 'data.frame'){
                   print(paste("After 2 tries we were unable to download the",
                               "supplemental data from GEO. This occasionally happens",
                               "when GEO is performing maintenance, but also could be",
                               "because there is no supplemental data"))
                   quit('no', as.numeric(exitCodes['unableToFindDataError']))
               }else{
                   if(debug){print(paste("got is", class(got)))}
                   if(debug){print(got)}
               }
            }
            
            # Now we process the raw expression data, depending on what sort of microarray was used
            setwd(microarrayDir)
            for (gseID in gseIDVec) untarRawData(gseID)
            if(grepl("Illumina", chipType) && grepl("beadchip", chipType, ignore.case = TRUE)){
                esetAndDescList <- processAndNormalizeIlluminaBedChip(gseIDVec, idType, gplName)
                eset <- esetAndDescList$eset
                illuminaDescriptionList(gseList, idType, esetAndDescList$desc)
            }else if (grepl("Agilent", chipType)){
                eset <- processAndNormAgilentChip(gseIDVec, idType)
            }else if (areGprFiles(gzipped = TRUE)){
                unzipGpr()
                eset <- gprFilesToEset()
            }else if (areGprFiles()){
                eset <- gprFilesToEset()
            }else{
                # Anything else should be Affy (or nimblegen, but that isn't yet supported)
                esetList <- list()
                for(j in 1:length(gseIDVec)){
                    gseID <- gseIDVec[j]
                    gsmsToUse <- gsmsToUseList
                    rawData <- getrawData(gseID, gsmsToUseList[[j]], chipType)
                    # The oligo package will attempt to identify the annotation package required to read the
                    # data in. If this annotation package is not installed, oligo will try to download it from
                    # BioConductor. If the annotation is not available on BioConductor, the user should use 
                    # the pdInfoBuilder package to create an appropriate annotation. In case oligo fails
                    # to identify the annotation package's name correctly, the user can use the pkgname argument
                    # available for both read.celfiles and read.xysfiles.     
                    print("Tying to run RMA")
                    esetList[[j]] <- rma(rawData)
                    # This is how we used to do it with the affy package, but oligo is much easier!
                    # This reads in the cel files directly and then runs RMA
                    #eset <- justRMA(filenames=celFiles)
                }
                # now combine all of the esets
                eset <- do.call("combine", esetList)
            }
            
            # Next we look for outliers and save the eset
            processAndSaveEset(idType, eset, geoID, fullGeoID)
        }else{
            alreadyDone()
        }
# Even better, RNA-seq data
#
    }else if(idType=='geo-seq'){
        if(! file.exists(paste0(microarrayDir,"/rnaSeqAnalysis/uniprot_expression_data.rds"))){
            # load necessary libraries
            loadLibraries('meta')
            
            print("Retrieving meta data from GEO")
            # Retrieve meta data including mapping information
            metaDataRelated <- accessAndSaveMetaData(geoID,
                                                     idType,
                                                     idVec,
                                                     preferredGpl,
                                                     fullGeoID
                                                     )
            prepRNAseqDownload(metaDataRelated, idType, fullGeoID)
        }else{
            alreadyDone()
        }
    }else{
        print(paste("WARNING: An un-recognized data source code was provided which include 'geo'.",
                    "The only options including geo are: geo and geo-orig. Quitting now"))
        quit('no', as.numeric(exitCodes['usageError']))
    }
#====================================
# Data from arrayExpress
#====================================
}else if(grepl('ae', idType)){
    library(ArrayExpress)
    source('getAE.R')
#
# This is for data where, for whatever reason, we can't get or process the raw data
#
    if(idType=="ae-orig"){
        if(! file.exists(paste0(microarrayDir,'/',geoID,"_asFromGeo.rds"))){
            print(paste("Running meta using the data directly from arrayExpress,",
                        "instead of the raw data and normalizing that"))

            # load necessary libraries
            loadLibraries('meta')
            
            # This was written to work with: "E-MTAB-2976",
            # which didn't fit the example given in the bioConductor pdf for arrayExpress
            # That may be b/c it's agilent, and they're a pain in the ass to deal with
            # First get the data
            setwd(microarrayDir)
            esetList <- allDataList <- list()
            for(i in 1:length(idVec)){
                id <- idVec[i]
                # Retrieve the expression data first
                esetAndAllData <- getAeOrigEset(id)
                esetList[[i]] <- esetAndAllData$eset
                allDataList[[i]] <- esetAndAllData$allData
            }
            
            # retreive the chiptype info from the adf file
            adfFiles <- list()
            for(i in 1:length(allDataList)){
                adfFiles[[i]] <- allDataList[[i]]$adf
            }
            # for ae-orig we've co-opted the preferredGpl for the preferred processed file.
            chipType <- getAeChipType(adfFiles, preferredGpl)
            # combine the esets
            eset <- combineEsets(esetList)

            # now get the rest: meta data and mapping information
            print("Now parsing meta data")
            getAndSaveAEMetaData(idType,
                                 allDataList,
                                 fullGeoID,
                                 preferredGpl,
                                 eset = eset,
                                 chipType = chipType)

            # Look for outliers and then save the data
            processAndSaveEset(idType, eset, geoID, fullGeoID)
            
        }else{
            alreadyDone()
        }
#
# This is what we prefer: microarray data where we can get the raw data and normalize it ourselves
#
    }else if(idType=='ae' || idType=='ae-mixed'){
        if(! file.exists(paste0(microarrayDir,"/",geoID,"_normalized.rds"))){
            
            # load necessary libraries
            loadLibraries('meta')
            
            setwd(microarrayDir)
            esetList <- rawDataList <- allDataList <- chipTypeList <- list()
            for (i in 1:length(idVec)){
                id <- idVec[i]
                print("Retrieving data from Array Express")
                # There is a one off exception where someone saved their raw data as processed data
                # The easiest, though hacky, way to deal with this is to change the idType for that ID here
                # This is not advised, but what are you gonna do...
                trueRawFile <- NULL
                if(grepl("E-TABM-576", id)){
                    print(paste("WARNING: This dataset, E-TABM-576 was incorrectly",
                                "saved on arrayExpress, and to deal with that we are",
                                "treating the processed data as raw data.",
                                "If, for some reason in the future this is no longer correct,",
                                "this code will need to be commented out"))
                    allData <- getAE(id, type = "full")
                    trueRawFile <- allData$processedFiles
                }

                allDataList[[i]] <- getAE(id, type = "raw")

                # not sure this is necessary, but in the one-off case above, I'll plug the file in here too
                if(! is.null(trueRawFile)){
                    allDataList[[i]]$rawArchive <- trueRawFile
                }
                
                currentAllData <- allDataList[[i]]
                chipTypeList[[i]] <- getAeChipType(currentAllData$adf, preferredGpl)
                
                if(grepl("Illumina", chipTypeList[[i]]) &&
                   grepl("beadchip", chipTypeList[[i]], ignore.case = TRUE)){
                    esetAndEscList <- processAndNormalizeIlluminaBedChip(idVec, idType)
                    esetList[[i]] <- esetAndEscList$eset
                    # this isn't up and working yet  illuminaDescriptionList(gse, idType, esetAndEscList$desc)
                }else if (grepl("Agilent", chipTypeList[[i]])){
                    esetList[[i]] <- processAndNormAgilentChip(idVec, idType)
                }else if (areGprFiles(gzipped = TRUE)){
                    unzipGpr()
                    esetList[[i]] <- gprFilesToEset()
                }else if (areGprFiles()){
                    esetList[[i]] <- gprFilesToEset()
                }else if ( grepl("-NCMF-", id)){
                    esetList[[i]] <- ncmfFilesToEset()
                # these stupid datasets have a lot of "unique" data types,
                # so I'll just deal with them in their own functions
                }else if ( grepl("-SMDB-", id) ){
                    adfFile <- selectSingleAdf(currentAllData$adf, preferredGpl)
                    esetList[[i]] <- smdbFilesToEset(id, adfFile)
                }else if(grepl("E-MEXP-100", id)){
                    targets <-  readWithLimma_initial(glob2rx("E-MEXP-100-raw-data-*txt"))
                    RG <- prefixedGenePix(targets)
                    esetList[[i]] <- readWithLimma_final(RG)
                }else if(grepl("E-MEXP-1007", id)){
                    targets <-  readWithLimma_initial(glob2rx("*.txt.txt"))
                    RG <- prefixedGenePix(targets)
                    esetList[[i]] <- readWithLimma_final(RG)
                }else if(grepl("E-TABM-115", id) | grepl("E-TABM-77", id)){
                    targets <-  readWithLimma_initial(glob2rx("US22*.txt"))
                    RG <- read.maimages(targets,
                                       columns = list(R = "Feature Extraction Software:rMeanSignal"
                                                    , G = "Feature Extraction Software:gMeanSignal"
                                                    , Rb = "Feature Extraction Software:rBGMeanSignal"
                                                    , Gb = "Feature Extraction Software:gBGMeanSignal"
                                                   ),
                                       annotation = c("metaColumn",
                                                      "metaRow",
                                                      "column",
                                                      "row",
                                                      "Reporter identifier"
                                                     )
                                       )
                    esetList[[i]] <- readWithLimma_final(RG)
                }else{
                    # Anything else should be Affy (or nimblegen, but that isn't yet supported)
                    rawDataList[[i]] <- getrawData(id, gsmsToUse = NA, chipTypeList[[i]])
                }
            }
            # combine everything
            # th first thing to do is check and see if the platforms are the same
            # we'll use the adf file as a proxy for that
            initialAdfFile <- allDataList[[1]]$adf
            
            # test to ensure that all of the platforms are the same
            if(length(allDataList) > 1){
                for(i in 2:length(allDataList)){
                    if(allDataList[[i]]$adf != initialAdfFile){
                        print("WARNING: adf files do not match, suggesting different platforms. We do not yet support multiple platforms")
                        quit('no', as.numeric(exitCodes['multiplePlatformsError']))
                    }
                }
            }
            
            if(length(rawDataList) >= 1){
                print("Tying to run RMA")
                # somehow combine the rawDataList
                # combine the affyBatch objects
                rawData <- rawDataList[[1]]
                if(length(rawData) > 1){
                    for(i in 2:length(rawData)){
                        temp <- merge.AffyBatch(rawData, rawDataList[[i]])
                        rawData <- temp
                    }
                }
                eset <- rma(rawData)
            }else{
                eset <- combineEsets(esetList)
            }

            # now combine the chipTypes
            chipType <- chipTypeList[[1]]
            if(length(chipTypeList) > 2){
                for(i in 2:length(chipTypeList)){
                    if(chipTypeList[[i]] != chipType){
                        quit('no', as.numeric(exitCodes['multiplePlatformsError']))
                    }
                }
            }
            # now get the rest: meta data and mapping information
            getAndSaveAEMetaData(idType,
                                 allDataList,
                                 fullGeoID,
                                 preferredGpl,
                                 eset = eset,
                                 chipType = chipType
                                )
            # Look for outliers and then save the data
            processAndSaveEset(idType, eset, geoID, fullGeoID)
        }else{
            alreadyDone()
        }
    }else if(idType=='ae-seq'){
        if(! file.exists(paste0(microarrayDir,"/",geoID,"_normalized.rds"))){
            
            # load necessary libraries
            loadLibraries('meta')
            
            setwd(microarrayDir)
            print("Retrieving data from Array Express")
            allData = getAE(geoID, type = "raw")

            getAndSaveAEMetaData(idType, list(allData), fullGeoID, preferredGpl)
            prepRNAseqDownload(allData, idType, geoID)
        }
    }else{
        print("WARNING: An un-recognized data source code including 'ae' was provided. ae is for arrayExpress and can be ae or ae-orig. Quitting now")
        quit('no', as.numeric(exitCodes['usageError']))
    }
}else{
    print("WARNING: An un-recognized data source code was provided. We support data from GEO and ArrayExpress. Quitting now")
    quit('no', as.numeric(exitCodes['usageError']))
}

finishedSuccessfully()
