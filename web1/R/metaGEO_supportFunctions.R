#========================================================================================
# functions for metaGEO
#========================================================================================
getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) return('None')
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}


thisScriptsPath <- getScriptPath()
if(thisScriptsPath != 'None') source(paste0(thisScriptsPath, '/general_supportFunctions.R'))

#========================================================================================
# For use with GEO
#========================================================================================

# This will download and save metaData (including mapping data) for either geo or geo-orig
robustGetGEO <- function(id, destdir = tempdir(), GSEMatrix = TRUE, maxTries = 3, sleepTime = 5){
    got <- NULL
    tries <- 0
    while(! class(got) %in% c('GSE', 'GDS', 'GPL', 'list') & (tries < maxTries)){
        if (tries > 0){
            Sys.sleep(sleepTime)
            print(paste('The previous attempt at getGEO failed, retrying on attempt number:', tries))
        }
        got <- tryCatch(
                   getGEO(id, destdir = destdir, GSEMatrix = GSEMatrix),
                   error=function(e){return(e)}
               )
        tries <- tries + 1
    }
    if (! class(got) %in% c('GSE', 'GDS', 'GPL', 'list')){
        print(paste("After",
                     maxTries,
                    "tries we were unable to download the data from GEO.",
                    "This occasionally happens when GEO is performing maintenance.",
                    "Try again in a few hours."))
        print(paste('got class ended up being:', class(got)))
        print('And got itself:')
        print(got)
        quit('no', as.numeric(exitCodes['unableToFindDataError']))
    }else{
        if(debug){print(paste('got ended up being:', class(got)))}
    }
    return(got)
}

accessAndSaveMetaData <- function (geoID, idType, idVec, preferredGpl, fullGeoID){
    # because we're now dealing with potentially multiple geoIDs, we need to store
    # all the important information, and combine them before returning

    if(idType=='geo-orig'){
        metaDataList <- vector("list", length(idVec))
        for(i in 1:length(idVec)){
            id <- idVec[i]
            if (grepl("GDS", id)) {
                gds <- robustGetGEO(id, destdir = microarrayDir)
                metaDataList[[i]] <- Columns(gds)
            
            } else if (grepl("GSE", id)) {
                gse <- robustGetGEO(id, GSEMatrix = TRUE, destdir = microarrayDir)
                # And now some characteristics of each:
                phenotypicData <- pData(phenoData(gse[[1]]))
                gsms <- getGsms(gse, 1)
                metaDataList[[i]] <- parseMetaData(gsms, phenotypicData)
                
            } else {
                print ("Unknown GEO data type. We only accept GDS and GSE. Exiting")
                quit('no', as.numeric(exitCodes['unknownGeoDataTypeError']))
            }
        }
        # combine the metaDataList, and report them
        metaData <- do.call("rbind", metaDataList)
        saveMetaData(metaData, idType, fullGeoID)

    }else if(idType=='geo' || idType=='geo-mixed'){
        # because we're now dealing with potentially multiple geoIDs, we need to store
        # all the important information, and combine them before returning
        gplList <- gseIDList <- gseList <- gsmsToUseList <- metaDataList <- vector("list", length(idVec))
        
        for(i in 1:length(idVec)){
            id <- idVec[i]
            if (grepl("GDS", id)) { # if GDS, need to get the GSE
                gds <- robustGetGEO(id, destdir = microarrayDir)
                metaDataList[[i]] <- Columns(gds)
                gplList[[i]] <- checkGpls(Meta(gds)$platform, preferredGpl, id)
                gseIDList[[i]] <- Meta(gds)$reference_series
                # This also serves to double check that there are not multiple GSEs
                # The issue is, that even if there are multiple GSEs, it may not be an issue
                # That's what I take care of below with the gsmsToUse
                print(paste0("For ", id, " the original GSE id is: ", gseIDList[[i]]))
                
                # In going back to the original gse some of the files have way more GSMs than we want
                # To get around that I'll make a list of the original GSMs we wanted and eliminate anything else
                sampleIds <- Meta(gds)$sample_id
                # But of course this is a nonsensical order
                sampleIds <- paste(sampleIds, collapse=",")
                gsmsToUseList[[i]] <- unique(strsplit(sampleIds,",")[[1]])
                
                print(paste0("Getting the original GSE: ", gseIDList[[i]]))
                gseList[[i]] <- robustGetGEO(gseIDList[[i]], GSEMatrix = FALSE, destdir = microarrayDir)
                
            }else if (grepl("GSE", id)){
                gseIDList[[i]] <- id
                # And this will be needed for mapping probes to proteins later
                gseList[[i]] <- robustGetGEO(gseIDList[[i]], GSEMatrix = FALSE, destdir = microarrayDir)
                
                platforms <- lapply(GSMList(gseList[[i]]),function(x) {Meta(x)$platform})
                gplList[[i]] <- checkGpls(platforms, preferredGpl, id)
                gseForMeta <- robustGetGEO(id, GSEMatrix = TRUE, destdir = microarrayDir)
                differentNames <- names(gseForMeta)
                gplToUse <- gplList[[i]]
# we were running into issues when the preferred GPL, necessary b/c there were multiple platforms,
# was being replaced by a better GPL. The better GPL is necessary for mapping succesfully, but
# the preferredGPL is needed for pulling out only one type of chip
                if (! is.na(preferredGpl)){
                    gplName <- preferredGpl
                }else{
                    gplName <- gplToUse@header$geo_accession
                }
                if( any( grepl( gplName, differentNames, ignore.case = TRUE) ) ){
                    indForOurGpl <- grep(gplName, differentNames, ignore.case = TRUE)
                    print(paste("Total entries in GSE object", length(names(gseForMeta))))
                    print(paste("indForOurGpl", indForOurGpl))
                    gsms <- getGsms(gseForMeta, indForOurGpl)
                    gsmsToUseList[[i]] <- gsms
                    phenotypicData <- pData(phenoData(gseForMeta[[indForOurGpl]]))
                }else{
                    phenotypicData <- pData(phenoData(gseForMeta[[1]]))
                    gsmsToUseList[[i]] <- NA # this is simply a place holder
                }
                # And now some characteristics of each:
                gsms <- phenotypicData[,2]
                metaDataList[[i]] <- parseMetaData(gsms, phenotypicData)
            }else { # We can only work with those 2 types
                print ("Unknown GEO data type. We only accept GDS and GSE. Exiting.")
                quit('no', as.numeric(exitCodes['unknownGeoDataTypeError']))
            }
        }
        # now we combine all of the data from the different geoIDs
        # I also included some checks to make sure we weren't mixing platforms here
        gpl <- unique(unlist(gplList))
        if(length(gpl) != 1){
            print(paste('Multiple gpls found.',
                         'This may be because 2 data types are being mixed',
                         '(e.g. gene expression and genome variation). Quitting.'
                       ))
            quit('no',  as.numeric(exitCodes['multiplePlatformsError']))
        }
        gpl <- gpl[[1]]
        
        # Save the mapping information for ease of loading in sigGEO.R
        
        mappingOutputFile <- paste0(microarrayDir, "/", geoID, "_forMapping.rds")
        chipType <- saveMappingInfo('geo', gpl, mappingOutputFile, gse=geoID)
        
        metaData <- do.call("rbind", metaDataList)
        saveMetaData(metaData, idType, fullGeoID)

        gseIDVec = unique(unlist(gseIDList))
        return(list(gseIDVec = gseIDVec,
                    chipType = chipType,
                    gseList = gseList,
                    gsmsToUseList = gsmsToUseList,
                    gplName = gpl@header$geo_accession
                  ))
        
    }else if(idType=='geo-seq'){
        if(length(idVec) > 1){
            print("WARNING: Sorry, multiple geoID support not available for RNA-seq data, yet.")
            quit('no', as.numeric(exitCodes['usageError']))
        }
        id <- geoID
        if (grepl("GDS", id)) { # if GDS, need to get the GSE
            # Just for simplicity I'm going back to the original GSE
            gds <- robustGetGEO(id, destdir = microarrayDir)
            metaData <- Columns(gds)
            gpl <- robustGetGEO(Meta(gds)$platform, destdir = microarrayDir) 
            gseID <- Meta(gds)$reference_series
            # This also serves to double check that there are not multiple GSEs
            # The issue is that even if there are multiple GSEs, it may not be a probelm
            # That's what I take care of below with the gsmsToUse
            print(paste0("For ", id, " the original GSE id is: ", gseID))
            
            # In going back to the original gse some of the files have way more GSMs than we want
            # To get around that I'll make a list of the original GSMs we wanted
            # and eliminate anything else
            sampleIds <- Meta(gds)$sample_id
            # But of course this is a nonsensical order
            sampleIds <- paste(sampleIds, collapse=",")
            gsmsToUse <- unique(strsplit(sampleIds,",")[[1]])
            
        }else if (grepl("GSE", id)){
            gseID <- id
            gsmsToUse <- NA # this is simply a place holder
        }else { # We can only work with those 2 types
            print("Unknown GEO data type. We only accept GDS and GSE. Exiting")
            quit('no', as.numeric(exitCodes['unknownGeoDataTypeError']))
        }
        gse <- robustGetGEO(gseID, GSEMatrix = FALSE, destdir = microarrayDir)
        platforms <- lapply(GSMList(gse),function(x) {Meta(x)$platform})
        
        if(any(grepl("GPL9442", platforms))){
            print("WARNING: We do not support RNA-seq from Solid as of yet. Sorry.")
            quit('no', as.numeric(exitCodes['usageError']))
        }
        # And now some characteristics of each:
        gsms <- names(gse@gsms)
        pdata_list <- list()
        max_nrow <- 0
        for (gsm in gsms){
            max_nrow <- max(c(max_nrow, length(unlist(get(gsm, gse@gsms)@header))))
        }
        phenotypicData <- data.frame(matrix(ncol = length(gsms), nrow = max_nrow))
        colnames(phenotypicData) <- gsms
        for (gsm in gsms){
            print(gsm)
            x <- unlist(get(gsm, gse@gsms)@header)
            while(length(x) < nrow(phenotypicData)){
                x <- c(x, 'NA')
            }
            phenotypicData[,gsm] <- x
        }
        metaData <- parseMetaData(gsms, t(phenotypicData))
        saveMetaData(metaData, idType, fullGeoID)
        return(metaData)
    }
}

getGsms <- function(gseForMeta, indForOurGpl){
    our_GSE <- gseForMeta[[indForOurGpl]]
    temp <- rownames(pData(our_GSE))
    if (! all(grepl('GSM', temp, ignore.case = TRUE))){
        print('Row names were expected to be GSMs, but were not. Trying to recover')
        if (is.null(our_GSE$geo_accession)){
            print('Unable to find alternate GSM source')
        }else{
            temp <- our_GSE$geo_accession
        }
    }
    return(temp)
}

# we've run into a few GPLs that just aren't any good, so this is one function to deal with all of the one-off situations where we've taken the time to find 
# a different GPL that is an alternative and works better. That doesn't mean that there isn't an even more optimal GPL out there, just that these that we
# assign are better than what is given by the GEO data
checkGpls <- function(platforms, preferredGpl, id = NA){
    plat_opts <- get_uniq_platforms(platforms)
    if (length(plat_opts) != 1) {
         if( ! is.na(id) & grepl("GDS", id) ){
             print(paste("We do not support analyzing multiple platforms for GDS data.",
                         "We can for GSE data however, so go back to GEO",
                         "and find the original GSE ID and try that"
                       ))
             quit('no', as.numeric(exitCodes['multiplePlatformsError']))
         }else if( ! is.na(preferredGpl)){
             print(paste(preferredGpl,
                         "was selected as the prefered GPL. Moving forward with that."
                        ))
             plat_opts <- preferredGpl
         }else{
             print(paste("We do not support analyzing multiple platforms at once.",
                         "You must choose one.",
                         "For this sample your options are:",
                         paste(plat_opts, collapse = ", ")
                        ))
             print("If you re-run meta now, you should have the option to choose one of these.")
             # file name ends in 'err.txt' so it gets copied back to the
             # platform machine; a filter excludes most .txt files because
             # some quite large temporary ones may get unpacked during meta
             write.table(paste(plat_opts,
                               collapse = "\n"),
                         file = paste0(microarrayDir, "/multipleGPLsToChooseFrom.err.txt"),
                         quote = FALSE,
                         col.names = FALSE,
                         row.names = FALSE
                        )
             quit('no', as.numeric(exitCodes['multiplePlatformsError']))
    	}
    }
    gplToUse <- checkForAlternativeGpls(plat_opts, id)
    print(paste("Using", as.character(gplToUse), "as mapping file."))
    return(robustGetGEO(gplToUse, destdir = microarrayDir))
}

get_uniq_platforms <- function(v){
    x <- unique(v)
    if(is.list(x)){
        if (length(x) == 1) {
            x <- unique(unlist(x))
        }else if(length(x) == 2){
            if(length(x[[2]]) == 0) x <- unique(x[[1]])
        }else{
            print('WARNING: unexpectedly long list of platforms:')
            print(x)
        }
    }
    return(x)
}

checkForAlternativeGpls <- function(platforms, id){
    # This GPL is an alternative,
    # and it doesn't seem to be working, go back to the original
    replaced <- FALSE
    if(as.character(unique(platforms)) == "GPL17047" ||
       as.character(unique(platforms)) == "GPL16522" ||
       as.character(unique(platforms)) == "GPL15401" ||
       as.character(unique(platforms)) == "GPL16987"
       ){
         gplToUse  <- "GPL6244"
         replaced <- TRUE
    # GPL15974 and GPL5175 are all older alternatives to GPL18761 
    # and they were not working all that well 
    # I'm trying this as an alternative, hopefully it will work better
    }else if( (as.character(unique(platforms)) == "GPL15974" ||
               as.character(unique(platforms)) == "GPL5175" ||
               as.character(unique(platforms)) == "GPL20982" || # this lists 5175 as an alternative, so two hops leads to 18761
               as.character(unique(platforms)) == "GPL17629" ||
               as.character(unique(platforms)) == "GPL11028" || # this lists 5188 as an alternative, so two hops leads to 18761
               as.character(unique(platforms)) == "GPL20188" || # this should list 5188 as an alternative, so two hops leads to 18761
               as.character(unique(platforms)) == "GPL8238" ||
               as.character(unique(platforms)) == "GPL5188") &&
               (id != 'GSE22498' &&
                id != 'GSE22498:GPL5188' &&
                id != 'GSE50421'
               )
             ){
         gplToUse <- "GPL18761"
         replaced <- TRUE
    # Elsewhere I've adjusted the columns of "GPL4133" for this GSE.
    # There may be other GSEs for which this is useful
    }else if( (as.character(unique(platforms)) == "GPL4133" &&
               (id != 'GSE29801' && id != 'GSE32413')
             )){
         gplToUse <- "GPL6480"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL10526" ||
              as.character(unique(platforms)) == "GPL9486" ||
              as.character(unique(platforms)) == "GPL5760" ||
              as.character(unique(platforms)) == "GPL11670" ||
              as.character(unique(platforms)) == "GPL14877" ||
              as.character(unique(platforms)) == "GPL10371" ||
              as.character(unique(platforms)) == "GPL7869" ||
              as.character(unique(platforms)) == "GPL9324" ||
              as.character(unique(platforms)) == "GPL9101" ||
              as.character(unique(platforms)) == "GPL6791" ||
              as.character(unique(platforms)) == "GPL11433" ||
              as.character(unique(platforms)) == "GPL16311" ||
              as.character(unique(platforms)) == "GPL19109" ||
              as.character(unique(platforms)) == "GPL16268" ||
              as.character(unique(platforms)) == "GPL22945" ||
              as.character(unique(platforms)) == "GPL17996"
             ){
         gplToUse <- "GPL570"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL21970" ||
              as.character(unique(platforms)) == "GPL25483"
             ){
         gplToUse <- "GPL16686"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL14663" ||
              as.character(unique(platforms)) == "GPL24120" ||
              as.character(unique(platforms)) == "GPL19184"
             ){
         gplToUse <- "GPL96"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL19180"
             ){
         gplToUse <- "GPL97"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL13447" ||
              as.character(unique(platforms)) == "GPL23080"
             ){
         gplToUse <- "GPL571"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL10123"
             ){
         gplToUse <- "GPL10150"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL19983" ||
              as.character(unique(platforms)) == "GPL18964"
             ){
         gplToUse <- "GPL17692"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL7341"
             ){
         gplToUse <- "GPL6968"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL18402"
             ){
         gplToUse <- "GPL23860"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL20650"
             ){
         gplToUse <- "GPL13667"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL16770" &&
              id != 'GSE68224'
             ){
         gplToUse <- "GPL15159"
         replaced <- TRUE
    }else if( as.character(unique(platforms)) == "GPL10850"
             ){
         gplToUse <- "GPL13264"
         replaced <- TRUE
#
#
# By elminating this I know I'm breaking one dataset, but I'm fixing 2 other datasets that we're currently using, 
# so I'll have to come back here when I find out which one I broked, and add it's GSE below
#
#
#    }else if( as.character(unique(platforms)) == "GPL1708" && geoID == 'GSE____'){
#         print(paste0("The GPL for this dataset is ", as.character(unique(platforms)), ".  We have had better luck replacing that with GPL6848. Doing so now."))
#         gplToUse <- "GPL6848"
    }else{
         gplToUse <- as.character(unique(platforms))
    }
    if(replaced){
         print(paste("The GPL for this dataset is",
                     as.character(unique(platforms)),
                     ".  We have had better luck replacing that with ",
                     gplToUse,
                     ". Doing so now."
                   ))
    }
    return(gplToUse)
} 

# This will download the actual expression data for geo-orig
oldVersionToEsetAndSaveData <- function(geoID, idVec, preferredGpl){
    gplList <- esetList <- dataList <- vector("list", length(idVec))
    for(i in 1:length(idVec)){
        id <- idVec[i]
        if (grepl("GDS", id)) { # if GDS, simply pull down the data
            print("Parsing GDS")
            gds <- robustGetGEO(id, destdir = microarrayDir)
            gplList[[i]] <- checkGpls(Meta(gds)$platform, preferredGpl, id)
            dataList[[i]] <- Table(gds)
            rownames(dataList[[i]]) <- dataList[[i]][,"ID_REF"]
            gdsToBeEset <- robustGetGEO(id, destdir = microarrayDir)
            esetList[[i]] <- GDS2eSet(gdsToBeEset)
        } else if (grepl("GSE", id)) { # GSEs take a little extra processing
            print("Parsing GSE")
            gse <- robustGetGEO(id, GSEMatrix = FALSE, destdir = microarrayDir)
            platforms <- lapply(GSMList(gse),function(x) {Meta(x)$platform})
            gplList[[i]] <- checkGpls(platforms, preferredGpl, id)
            gseDataFile <- paste0(microarrayDir,"/", id, "_gseData.rds")   
            if( file.exists(gseDataFile)){
                dataList[[i]] <- readRDS(gseDataFile)
            }else{
                d_tbl <- Table(GSMList(gse)[[1]])
                data <- d_tbl["ID_REF"]
                samples <- length(GSMList(gse))
                for (j in 1:samples) {
                    data <- merge(data, Table(GSMList(gse)[[j]])[c("ID_REF","VALUE")], by = 1, all = TRUE)
                }
                nonEmptyRows <- apply(data, 1, function(row) all(row !=0 ))
                data <- data[nonEmptyRows,]
                nonEmptyRows <- apply(data, 1, function(row) all(! is.na(row) ))
                data <- data[nonEmptyRows,]
                colnames(data)[2:(samples+1)] <- names(GSMList(gse))
                rownames(data) <- data$ID_REF
                dataList[[i]] <- data
            }
            esetList[[i]] <- new("ExpressionSet", exprs=as.matrix(dataList[[i]][,2:ncol(dataList[[i]])]))
        } else { # We can only work with those 2 types
            print ("Unknown GEO data type. We only accept GDS and GSE. Exiting")
            quit('no', as.numeric(exitCodes['unknownGeoDataTypeError']))
        }
    
    # now we have to combine the data from the various sources
    # I also included some checks to make sure we weren't mixing platforms here
        gpl <- unique(unlist(gplList))
        if(length(gpl) != 1){
            print(paste('Multiple gpls found in oldVersion.',
                        'This may be because 2 data types are being mixed',
                        '(e.g. gene expression and genome variation).',
                        'Quitting.'
                       ))
            quit('no', as.numeric(exitCodes['multiplePlatformsError']))
        }
    }
    mappingOutputFile <- paste0(microarrayDir, "/", geoID, "_forMapping.rds")
    chipType <- saveMappingInfo('geo', gplList[[1]], mappingOutputFile, gse=geoID)

    # I'll use the same process as we did for a single GSE:
    print("Saving pre-processed data")
    dataFile = paste0(microarrayDir,"/", geoID,"_asFromGeo.rds")
    if(! file.exists(dataFile)){
        allData <- dataList[[1]]
        if(length(idVec) > 1){
            # this merge isn't working, likely because at least one of these is empty
            for( j in 2:length(dataList)){
                allData <- merge(allData, dataList[[j]], by = 1, all = TRUE)
            }
                    
            nonEmptyRows<-apply(allData, 1, function(row) all(row !=0 ))
            allData <- allData[nonEmptyRows,]
            nonEmptyRows <- apply(allData, 1, function(row) all(! is.na(row) ))
            allData <- allData[nonEmptyRows,]
            rownames(data) <- data$ID_REF
            # my hope is that the column names will still be there as each one was processed on its own
            #colnames(allData)[2:(samples+1)] <- names(GSMList(gse))
        }
        saveRDS(allData, file = dataFile)
    }

    eset <- combineEsets(esetList)
}

# There is also a 2nd step with Illumina chips because the headers are often times incorrect
illuminaDescriptionList <- function(gseList, idType, descriptionList){
 
    # The eset currently has column names that aren't GSMs, this is an issue for analysis later
    # This will connect the GSM name and the current column names, and save it as a file for conversion in sigGEO.R
    # I did that rather than changing them here, because changing the column names is much easier once you only have the
    # expression data, which happens in sigGEO

    # in some cases I will have already been able to generate this list
    if(is.na(descriptionList)){
        if(idType=='geo'){
            for (gse in gseList) {
                temp <- lapply(GSMList(gse),function(x){ Meta(x)$description})
#                descriptionsList[[length(descriptionsList)]] <- sapply(descriptionsList, '[[', 1) # this gets the first sub element of every entry in a list
### The above approach was used before sprint189fixes, but it's not clear to me that this should have ever worked.
### I've left it here just in case this messess up previously working datasets, but the approach below looks to be more sensible
### Note I also started using 'temp' whereas before it was descriptionsList
                descriptionsList <- sapply(temp, '[[', 1) # this gets the first sub element of every entry in a list
            }
            saveRDS(unlist(descriptionsList), file = 'illuminaGsmConverter.rds')
            # values are the descriptions, and the names are the gsms
        }else{
            print('WARNING: Illumina beads from ArrayExpress not yet supported.')
            quit('no', as.numeric(exitCodes['illuminaBeadChipError']))
        }
    }else{
        saveRDS(descriptionList, file = 'illuminaGsmConverter.rds')
    }
}

#========================================================================================
# For use with ArrayExpress
#========================================================================================
# This pulls down all of the processed data
getAeOrigEset <- function(geoID){
    print("Retrieving data from Array Express")
    allData = getAE(geoID, type = "full")
    # there is at least one that has multiple files within here
    # Specifically they have mutliple cy5/cy3 files.
    # Not sure how to deal with this currently
    # I'll create a ticket, but for the time being I'll check for these files,
    # and put out a better warning if they're found
    if( ! is.null( allData$processedFiles ) ){
        if( any( grepl( "cy5.txt",  allData$processedFiles, fixed=T ) ) ){
            print("The processed data files from AE seem to be cy3/5 files, which we do not yet support. Sorry you can't use this data yet.")
            quit('no', as.numeric(exitCodes['usageError']))
        }
        if(length(unique(allData$processedFiles)) > 1){
            print("We found multiple processed data files, and don't know which one to choose from.")
            quit('no', as.numeric(exitCodes['multiplePlatformsError']))
        }
        
        print(paste("Trying to open", allData$processedFiles))
        procData <- read.csv( allData$processedFiles, sep = "\t", header = TRUE)
        probeColumn <- grep("probe", colnames(procData), ignore.case = TRUE)
        if(length(probeColumn) == 0){
            probeColumn <- grep(glob2rx("*REF"), colnames(procData), ignore.case = TRUE)
        }
        if(length(probeColumn) == 0){
            print("Unable to find the probe column for the AE pre-processed data.")
            quit('no', as.numeric(exitCodes['unableToFindDataError']))
        }
        toRemove <- c(probeColumn, grep("symbol", colnames(procData), ignore.case = TRUE))
        rownames(procData) <- procData[,probeColumn]
        procData <- procData[,-1*toRemove]
        # I ran into some data with an additional header. Rather than really dealing with that I'll just make them NAs
        matrixToUse <- as.matrix(procData)
        matrixToUse[1, grep("Software Unknown:log2 ratio", matrixToUse[1,])] <- NA
        return(list(eset = new("ExpressionSet", exprs = matrixToUse), allData = allData))
    }else{
        print("Could not find ArrayExpress processed data file. Ensure the ID is correct.")
        quit('no', as.numeric(exitCodes['unableToFindDataError']))
    }
}

# to deal with multiple adf files we don one thing, but call it several times
selectSingleAdf <- function(adfFile, preferredGpl){
    if(length(unique(unlist(adfFile))) > length(unique(adfFile))){
        adfFile <- unlist(adfFile)
    }
    print(paste("preferredGpl is", preferredGpl))
    if(length(unlist(adfFile)) > 1 ){
        if( ! is.na(preferredGpl) ){
            print(paste(preferredGpl, "was selected as the preferred ADF file. Moving forward with that."))
            adfFile <- preferredGpl
        }else{
            plat_opts <- get_uniq_platforms(adfFile)
            print(paste('We do not support analyzing multiple platforms at once.',
                        'You must choose one. For this sample your options are:',
                         paste(plat_opts, collapse = ", ")
                       ))
            write.table(paste(plat_opts,
                              collapse = "\n"),
                        file = paste0(microarrayDir,
                                      "/multipleGPLsToChooseFrom.err.txt"
                                     ),
                        quote = FALSE,
                        col.names = FALSE,
                        row.names = FALSE
                       )
            quit('no', as.numeric(exitCodes['multiplePlatformsError']))
        }
    }else{
        adfFile <- unlist(adfFile)
    }
    print(paste("Using", adfFile))
    return(adfFile)
}

# retrieve the chipType information
getAeChipType <- function(adfFile, preferredGpl){
    adfFile <- selectSingleAdf(adfFile, preferredGpl)
    print(paste("adfFile is", adfFile))
    if(debug){print(paste("adfFile class is", class(adfFile)))}
    wholeAdfFile <- read.csv(adfFile, header = FALSE, sep = "\t")
    print(paste("returning", as.character(wholeAdfFile[1,2]), "as the chiptype"))
    return(as.character(wholeAdfFile[1,2]))
}

# when downloading with AE, this actually gets done. but it intermixes it with other files, and I can't tell what is the data file I need, so I do it myself, just to be sure
unpackAERawData <- function(gseID, toMoveDataTo){
    if(grepl("E-TABM-576", id)){
        system(paste0("cp Inma_extractions_Sample_Probe_Profile.txt ", toMoveDataTo, "/")) 
        return(list('Inma_extractions_Sample_Probe_Profile.txt'))
    }else{
        currentWD <- getwd()
        system(paste0("rm -r -f ", toMoveDataTo, "/*"))
        system(paste0("cp ", gseID,".raw*.zip ", toMoveDataTo))
        setwd(toMoveDataTo)
        system("for file in *.zip; do unzip $file; done")
        system("rm *.zip")
        unzipGpr()
        unpacked_files <- list.files()
        setwd(currentWD)
        return(unpacked_files)
    }
}

# access, format and save the metaData including mapping information (only for microarrays) for raw arrayExpress data
getAndSaveAEMetaData <- function(idType, allDataList, fullGeoID, preferredGpl, chipType = NULL, eset = NULL){
    gsmsList <- metaDataList <- vector("list", length(idVec))
    for(i in 1:length(allDataList)){
    # The sdrf file has all of the info we need:
        phenotypicData <- read.csv(allDataList[[i]]$sdrf, sep = "\t", header = TRUE)
        # get the GSMs  out for meta data:
        if(idType =='ae-seq'){
            gsms <- phenotypicData[,2]
        }else{
            gsms <- rownames(eset@protocolData@data)
        }
        # Get rid of any underscores, the GSM should always be first and there are descriptions after that separated by _
        gsmsList[[i]] <- sapply(gsms, '[[', 1) # this gets the first sub element of every entry in a list
        metaDataList[[i]] <- parseMetaData(gsmsList[[i]], phenotypicData)
    }
    # combine the metaData
    metaData <- do.call("rbind", metaDataList)
    saveMetaData(metaData, idType, fullGeoID)
    if(idType!='ae-seq'){
       getMappingDataForAe(allDataList, chipType, preferredGpl)
    }
}

# Parses through some of the meta data that comes from AE to get the information needed to turn probes into gene names
getMappingDataForAe <- function(allDataList, chipType, preferredGpl){
    adfFiles <- list()
    for(i in 1:length(allDataList)){
        adfFiles[[i]] <- allDataList[[i]]$adf
    }
    print(paste("Initial adfs", unlist(adfFiles)))
    adfFile <- selectSingleAdf(unlist(adfFiles), preferredGpl)
    
    # The adf file is the equivalent of a GPL file for geo, and only needed for microarrays
    # Ignoring some of the header information b/c I only care about the mapping information
    print(paste("ADF file we're tying is", adfFile))
    gplEquiv <- check_adf(adfFile)
    # For whatever reason when I tried to do this is place it didn't work, so step it out
    columnNamesOfGpl <- colnames(gplEquiv) 
    
    # The subsequent code calls the probe column 'ID', so we fix that here
    columnNamesOfGpl[1] <- "ID"
    
    # The convert names have this annoying prefix: Composite.Element.Database.Entry. and end in .
    # So we remove it
    for (i in 1:length(columnNamesOfGpl)){
        if(grepl('Reporter.Database.Entry.', columnNamesOfGpl[i])){
            temp <- strsplit(columnNamesOfGpl[i],'Reporter.Database.Entry.', fixed=T)[[1]]
            columnNamesOfGpl[i] <- temp[length(temp)]
            columnNamesOfGpl[i] <- gsub(".", "",columnNamesOfGpl[i], fixed=TRUE)
        }
        if(grepl('Composite.Element.Database.Entry.', columnNamesOfGpl[i])){
            temp <- strsplit(columnNamesOfGpl[i],'Composite.Element.Database.Entry.', fixed=T)[[1]]
            columnNamesOfGpl[i] <- temp[length(temp)]
            columnNamesOfGpl[i] <- gsub(".", "",columnNamesOfGpl[i], fixed=TRUE)
        }
    }
    colnames(gplEquiv) <- columnNamesOfGpl
    
    # Let's save this for ease of loading in sigGEO.R
    saveMappingInfo('ae', list(table = gplEquiv, chipType = chipType), paste0(geoID,"_forMapping.rds"))
}

check_adf <- function(adfFile){
    # some (1 ID'd so far) files have EOLs that are causing issues
    if (endsWith(adfFile, 'A-AGIL-11.adf.txt')){
        print("The ADF files provided, has been seen to have non-Unix EOL characters. Fixing that...")
        command <- paste0('cat ', adfFile, " | tr '\r' '\n' | tr -s '\n' > tmp")
        system(command, intern = FALSE)
        command <- paste0('mv tmp ',adfFile)
        system(command, intern = FALSE)
    }
    command <- paste0('grep -n "\\[main\\]" ', adfFile, ' | cut -f1 -d:')
    headerNum <- as.integer(system(command, intern = TRUE))
    command2 <- paste0('wc -l ', adfFile, ' | cut -f1 -d" "')
    totalNum <- as.integer(system(command2, intern = TRUE))
    if(totalNum == headerNum){
        print("ADF file is not in the expected format. Searching for a file name in the ADF file that is actually helpful")
        possible_file <- system(paste('grep "Comment\\[AdditionalFile:CSV" ', adfFile, ' | cut -f2'), intern = TRUE)
        possible_file <- gsub('\\[main\\]', '', possible_file)
        adf_prefix <- gsub("\\.adf\\.txt$", "", adfFile, perl = TRUE)
        download_file <- paste0(adf_prefix, '.additional.1.zip')
        system(paste0('wget http://www.ebi.ac.uk/arrayexpress/files/', adf_prefix, '/', download_file))
        system(paste0('unzip ', download_file))
        gplEquiv <- read.csv(possible_file,
                             sep = ',',
                             header = TRUE,
                             comment.char = "#"
                             )
    }else{
        gplEquiv <- read.csv(adfFile,
                             sep = '\t',
                             header = TRUE,
                             skip = headerNum
                            )
    }
    return(gplEquiv)
}

#========================================================================================
# For use with any data
#========================================================================================
# exactly what it says, this function uses the metadata to get the addresses for the RNA-seq reads, and then kicks off the bash script that processes them.
prepRNAseqDownload <- function(metaData, idType, fullGeoID){
# Set up the directories needed
    outDir <- paste0(microarrayDir, "/rnaSeqAnalysis")
    system(paste('mkdir -p', outDir))
    rawFastqDir <- paste0(outDir, "/rawFastq")
    system(paste('mkdir -p', rawFastqDir))

# The parsing and downloading is dependent on where the data is coming from
    if(idType=="geo-seq"){
        # Go through the meta data to find the srx id for each GSM.
        # Then create a conversion file to be used by sigGEO 
        # combine column 2 and 3, split on " " and or ";"
        # grep for sra, and pull out ftp address
        gsmToSrxConversion <- metaData[,1:2] # This is an easy way to get the same size dataframe
        adressesToDownload <- list()
        to_rm_from_meta_data <- list()
        srrs_to_combine <- list()
        # It's probably also worthwhile to see if we can get the sequencing information;
        # particularly the kit made so we can get the adaptor sequences
        for(i in 1:nrow(metaData)){
            gsm <- metaData[i,1]
            individualFields <- unlist(strsplit(metaData[i,2], "\t"))
            ftpIndex <- grep("ftp://ftp-trace.ncbi.nlm.nih.gov/sra/sra-instant/reads/",
                             individualFields,
                             fixed = TRUE
                            )
            if (length(ftpIndex) == 1){
                srx <- getSRXfromUrl(trimWhiteSpaces(individualFields[ftpIndex]))
            }else{
                httpsSearchIndex <- grep("https://www.ncbi.nlm.nih.gov/sra?term=SRX",
                             individualFields,
                             fixed = TRUE
                            )
                if (length(httpsSearchIndex) == 1){
                    srx <- getSRXfromUrl(trimWhiteSpaces(individualFields[httpsSearchIndex]),
                                         sep = '='
                                        )
                }else{
                    print(paste("WARNING: Unable to find SRA download address for:",
                             gsm,
                             "Verify that this is RNA-seq and that these addresses are correct,",
                             "but we're leaving this sample out for now.")
                         )
                    print(httpsSearchIndex)
                    if (debug) print(individualFields)
                    to_rm_from_meta_data[[length(to_rm_from_meta_data) + 1]] <- gsm
                    next
                }
            }

            # just use the first one as the prefered
            preferredAcc <- srx # temp$accs[1]
            gsmToSrxConversion[i,2] <- preferredAcc
            adressesToDownload[[length(adressesToDownload) + 1]] <- srx
        }
        adressesToDownload <- unlist(adressesToDownload)
        # first remove the missing data from the existing meta file
        if (length(to_rm_from_meta_data) > 0){
            to_rm_from_meta_data <- unlist(to_rm_from_meta_data)
            metaDataFile <- get_meta_data_file_name(idType, fullGeoID)
            old_metaData <- read.table(metaDataFile,
                                       sep = "\t",
                                       header = FALSE,
                                       stringsAsFactors = FALSE
                                      )
            rows_to_rm <- list()
            for (i in 1:nrow(old_metaData)){
                if (length(rows_to_rm) == length(to_rm_from_meta_data)){
                    break
                }
                if (grepl(old_metaData[i,1], to_rm_from_meta_data)){
                    rows_to_rm[[length(rows_to_rm) + 1]] <- i * -1
               }
            }
            metaData <- old_metaData[rows_to_rm,]
            saveMetaData(metaData, idType, fullGeoID)
        }
        # write out the converter
        write.csv(gsmToSrxConversion,
                  file = paste0(microarrayDir,
                                "/",
                                fullGeoID,
                                "_gsmToSrxConversion.csv"
                               ),
                  quote = FALSE,
                  row.names = FALSE,
                  col.names = FALSE
                  )
        # write out the addresses to get
        toDownloadFile <- paste0(microarrayDir,
                                 "/",
                                 fullGeoID,
                                 "_SRAsToDownload.csv"
                                )
        write.csv(adressesToDownload,
                  file = toDownloadFile,
                  quote = FALSE,
                  row.names = FALSE,
                  col.names = FALSE
                 )
# This also aroise from the issues dealt with in plat2765 (see above)
#        # write the file used to combine certain samples
#        toCombineFile <- paste0(microarrayDir,
#                                 "/",
#                                 fullGeoID,
#                                 "_SRRsToCombine.tsv"
#                                )
#        # I already formatted the lines as I wanted,
#        # now just one line per entry
#        write.csv(unlist(srrs_to_combine),
#                  file = toCombineFile,
#                  quote = FALSE,
#                  row.names = FALSE,
#                  col.names = FALSE,
#                  sep = "\n"
#                 )
    }else if(idType=='ae-seq'){
            # the address for each sample is in the sdrf file,
            # so we will load that and get the correct column,
            # and simply write it to the download file
            sdrfData <- read.csv(metaData$sdrf, sep = "\t", header = TRUE)
            if(any(grepl("FASTQ_URI", colnames(sdrfData)))){
                urlColInd <- grep("FASTQ_URI", colnames(sdrfData))
                urlFile <- sdrfData
            }else{
                print("WARNING: Unable to find FASTQ_URI column from sdrf file, trying IDF file")
                idfData <- read.csv(metaData$idf, sep = "\t", header = TRUE)
                if(any(grepl("FASTQ_URI", colnames(idfData)))){
                    urlColInd <- grep("FASTQ_URI", colnames(idfData))
                    urlFile <- idfData
                }else{
                    print("WARNING: Unable to find FASTQ_URI column from idf file, that means we can't find the address to download the data from.")                    
                    quit('no', as.numeric(exitCodes['unableToFindDataError']))
                }
            }
            # write out the addresses to get
            toDownloadFile <- paste0(microarrayDir, "/", fullGeoID,"_FASTQsToDownload.csv")
            write.table(urlFile[,urlColInd], file = toDownloadFile, quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\n")
            
            # now get the conversion info
            if(any(grepl("ENA_RUN",colnames(sdrfData)))){
                runColInd <- grep("ENA_RUN",colnames(sdrfData)) # this is what the fastqs will be named
            # this might need to be "Scan Name"
            }else if(any(grepl("ENA_EXPERIMENT",colnames(sdrfData)))){
                runColInd <- grep("ENA_EXPERIMENT",colnames(sdrfData)) # this is what the fastqs will be named
            }else{
                print("WARNING: Unable to find ENA_RUN column from sdrf file, and therefore cannot analyze this data.")
                quit('no', as.numeric(exitCodes['unableToFindDataError']))
            }
            if(any(grepl("ENA_SAMPLE",colnames(sdrfData)))){
                sampleColInd <- grep("ENA_SAMPLE",colnames(sdrfData)) # this is what the metaData will be using
            }else{
                print("WARNING: Unable to find ENA_SAMPLE column from sdrf file, and therefore cannot analyze this data.")
                quit('no', as.numeric(exitCodes['unableToFindDataError']))
            }
            
            write.csv(cbind(as.character(sdrfData[,sampleColInd]), as.character(sdrfData[,runColInd]))
                       , file = paste0(microarrayDir, "/", fullGeoID, "_gsmToSrxConversion.csv")
                       , quote = FALSE, row.names = FALSE, col.names = FALSE)
    }else{
        print("WARNING: Unexpected idType given for RNA-seq pipeline. Only support geo-seq and ae-seq.")
        quit('no', as.numeric(exitCodes['usageError']))
    }
   # Now we plug into the shared pipeline
   # The first thing I want to take care of is to take care of the column name that R still puts on this file even though I tell it not to
   # and to put all of the strings on a single line
   system(paste("tail -n+2", toDownloadFile, '| tr "\n" " " >', paste0(microarrayDir, "/tempToDownload.tmp")))
   system(paste("mv", paste0(microarrayDir, "/tempToDownload.tmp"), toDownloadFile))
}


### If this ever fails us, we could also try a GSM to SRR conversion. See here:
### https://www.biostars.org/p/244150/
getSRXfromUrl <- function(url, sep = '/'){
    addressParts <- unlist(strsplit(url, sep)[[1]])
    return(addressParts[length(addressParts)])
}

# This will download the actual expression data for geo
getrawData <- function(gseID, gsmsToUse, chipType){
    if(grepl("GSE", gseID)){ # this only happens if it's data from GEO
        #if (file.exists(paste0(gseID,"/",gseID,"_RAW.tar"))){
        #    untar(paste0(gseID,"/",gseID,"_RAW.tar"))
        #}else 
        if(file.exists(paste0(gseID, "/", gseID, "_Raw_data.txt"))){
            print('Trying to load raw data from a single file')
            rawdata <- read.table(paste0(gseID, "/",
                                         gseID,"_Raw_data.txt"
                                        ),
                                  header = TRUE,
                                  row.names = 1
                                 )
            if(debug){print(head(rawdata))}
            print(paste("We have found the raw data in a simple text file,",
                        "but do not yet support this format.",
                        "We're working on it!"
                        )
                 )
            return(NULL)
        }
    }
    print(paste0("Raw data for ", gseID, " downloaded and unpacked"))
    celFiles <- retreiveAffyFiles(gsmsToUse, chipType)
    if(grepl("HuEx-1_0-st", chipType)){
       # it seems that this particular chipType normally tries to use a pd package
       # that doesn't exist: pd.huex.1.0.st.v1,
       # so this is a patch to make it work with v2,
       # which does have some differences in the control probes,
       # but none in the exons examined;
       # so hopefully it will work.
        affyRaw <- read.celfiles(celFiles, pkgname = 'pd.huex.1.0.st.v2')
    }else if(grepl(gseID, "E-TABM-158")) {
        print(paste("E-TABM-158 is trying to access a nonexistant pd. It's chiptype is",
                    chipType
                  ))
        # it seems that this particular chipType
        # normally tries to use a pd package that doesn't exist
        affyRaw <- read.celfiles(celFiles, pkgname = 'pd.ht.hg.u133a')
    }else if(grepl(gseID, "GSE68956")) {
        print(paste(gseID,
                    "is trying to access a nonexistant pd. It's chiptype is",
                    chipType
                   ))
        # this dataset supposedly only has one chip type, but read.celfiles was finding
        # that not to be true. I chose to force it in this case
        affyRaw <- read.celfiles(celFiles, pkgname = 'pd.hu6800', checkType = FALSE)
    }else{
        affyRaw <- read.celfiles(celFiles)
    }
    return(affyRaw)
}

untarRawData <- function(gseID){
    f <- paste0(gseID,"/",gseID,"_RAW.tar")
    if (file.exists(f)) untar(f)
}

areGprFiles <- function(gzipped = FALSE){
    if (gzipped){
        pattern = "*.gpr.gz"
    }else{
        pattern = "*.gpr"
    }
    if(debug) print(pattern)
    if(debug) print(list.files())
    if(debug) print(getwd())
    if (length(list.files(pattern = pattern, ignore.case = TRUE)) > 0) return(TRUE)
    return(FALSE)
}


# In essence all this does is read in the cel files from the current directory,
# but it also cleans up names, and only reads in the samples of interest
retreiveAffyFiles <- function(gsmsToUse,chipType){
    celFiles <- list.files(pattern = "*.CEL", ignore.case = TRUE)
    if(length(celFiles) == 0){
        # first thing to check is that the cel files aren't gzipped
        # I may need to consider case sensitivity, but 
        if(length(list.files(pattern = "*.cel.gz", ignore.case = TRUE)) > 0 ){
            if(length(list.files(pattern = "*.cel.gz", ignore.case = FALSE)) > 0 ){
                system("gunzip *.cel.gz")
            }else if(length(list.files(pattern = "*.CEL.gz", ignore.case = FALSE)) > 0 ){
                system("gunzip *.CEL.gz")
            }
            celFiles <- list.files(pattern = "*.CEL", ignore.case = TRUE)
        }else if(areGprFiles()){
            eset <- gprFilesToEset()
        }else if(areGprFiles(gzipped = TRUE)){
            unzipGpr()
            eset <- gprFilesToEset()
        }else{
            # better yet would be to have it search for xys files (from nimbleGen), which oligo can also handle.
            # we haven't had that be an issue as of yet though
            print("Unable to find .cel files. This likely means this is type of array we do not yet support.")
            print(paste("The chip type is:",chipType))
            if(length(list.files(pattern = "*.xys", ignore.case = TRUE)) > 0){
                print("We did see some xys files. This may be a NimbleGen array...time to add that ability.")
            }else{
                print("We didn't see any file types we recognized. These are all of the files in the directory: ")
                print(list.files())
            }
            quit('no', as.numeric(exitCodes['unknownGeoDataTypeError']))
        }
    }
    
    if( ! is.na(gsmsToUse) ){
        print('Subsetting GSMs using the gsmsToUse list')
        print(head(gsmsToUse))
    # I included this because there were some geoIDs superSeries that included GSMs from other studies
    # (e.g. miRNA reads in addition to mRNA)
        celIndsToUse <- vector(mode = 'numeric', length = length(gsmsToUse))
        for (i in 1:length(gsmsToUse)){
            celIndsToUse[i] <- grep(gsmsToUse[i], celFiles)
        }
        celFiles <- celFiles[celIndsToUse]
    }
    return(celFiles)
}


# We read several types of files with Limma, but they share some functions,
# we split them in 2 parts, inital and final
readWithLimma_initial <- function(patternToMatch){
    allFiles <- list.files(pattern = patternToMatch, ignore.case = TRUE)
    targetFile <- paste0( microarrayDir, "/targets.txt")
    write.table( c("FileName", unlist(allFiles)),
                file = targetFile,
                quote = FALSE,
                row.names = FALSE,
                col.names = FALSE
               )
    library(limma)
    library(convert)
    targets <- readTargets(targetFile)
    return(targets)
}

# Set up a filter so that any spot with a flag of .99 or less gets zero weight.
limmaFlagFunction <- function(x) as.numeric(x$Flags > -99)

readWithLimma_final <- function(RG_in){
    RG <- backgroundCorrect(RG_in, method = "normexp", offset=50)
    # this wasn't working for some arrays, so we'll try another normalization in those cases
    MA <- NULL
    # Print-tip loess normalization:
    try(MA <- normalizeWithinArrays(RG))
    if(is.null(MA)){
        print("Default, Print-tip loess normalization, normalization wasn't able to be performed. Attempting global Loess.")
        MA <- normalizeWithinArrays(RG, method = "loess")
    }
    eset <- as(MA, "ExpressionSet")
    return(eset)
}

# this seems to work for all AE files with the NCMF prefix
ncmfFilesToEset <- function(){
    targets <- readWithLimma_initial("BioDataCube_For_MeasuredData_[0-9]+.txt")
    RG <- read.maimages(targets,
                        columns = list(R = "ImaGene:Signal Mean_Cy5",
                                       G = "ImaGene:Signal Mean_Cy3",
                                       Rb = "ImaGene:Background Mean_Cy5",
                                       Gb = "ImaGene:Background Mean_Cy3"
                                      ),
                        annotation = c("metaColumn",
                                       "metaRow",
                                       "column",
                                       "row",
                                       "Reporter identifier"
                                      )
                        )
    return(readWithLimma_final(RG))
}

# all of these data seem to be somewhat unique, so I'll setup a function just to deal with them
smdbFilesToEset <- function(id, adfFile){
    if(adfFile == "A-SMDB-101.adf.txt"){
        targets <- readWithLimma_initial(paste0(id, "-raw-data-[0-9]+.txt"))
        # I'm far from certain that these are correct, but this is the best I could piece together.
        # I found a patent: http://www.google.com/patents/US7186510
        # that suggested that Ch1 is cy3 and ch2 is cy5.
        # After that, the Limma user guide suggested that for SMDB data I is foreground and B is background,
        # and that median should be used for B
        if(debug){print(targets) }
        RG <- read.maimages(targets,
                            columns = list(R = "*CH2I_MEAN",
                                           G = "*CH1I_MEAN",
                                           Rb = "*CH2B_MEDIAN",
                                           Gb = "*CH1B_MEDIAN"
                                          ),
                            annotation = c("metaColumn",
                                           "metaRow",
                                           "column",
                                           "row",
                                           "Reporter identifier"
                                          )
                           )
        return(readWithLimma_final(RG))
    }else if(adfFile == "A-SMDB-567.adf.txt" | adfFile == "A-SMDB-837.adf.txt"){
        targets <-  readWithLimma_initial(glob2rx("*GENEPIX*.txt"))
        RG <- prefixedGenePix(targets)
        return(readWithLimma_final(RG))
    }else{
        print(paste("We aren't able to support this data set just yet. For our reference, the adf is:", adfFile))
        quit('no', as.numeric(exitCodes['unableToFindDataError']))
    }
}


# We use Limma to load and process GPR files
gprFilesToEset <- function(){
    targets <- readWithLimma_initial("*.gpr")
    # this wasn't working for everything
    # I ran into a weird situation where some of the headers for these files have a prefix they shouldn't
    # Read in the data.
    RG <- NULL
    try(RG <- read.maimages(targets,
                            source = "genepix",
                            wt.fun = limmaFlagFunction
                           )
       )
    if(is.null(RG)){
        print("Something went wrong with read.maimages in gprFilesToEset. Trying to fix")
        # I ran into this enough times to make its own function
        RG <- prefixedGenePix(targets)
    }
    return(readWithLimma_final(RG))
}

# I ran into this enough times to make its own function
prefixedGenePix <- function(targets){
    RG <- read.maimages(targets,
                        columns=list(R="GenePix:F635 Mean", G="GenePix:F532 Mean", Rb="GenePix:B635 Mean",Gb="GenePix:B532 Mean"),
                        annotation=c("metaColumn", "metaRow", "column", "row", "Reporter identifier")
                        )
    return(RG)
}

# Illumina chips are normalized their own way
processAndNormalizeIlluminaBedChip <- function(gseIDVec, idType, gplName){
    print("The chip is an Illumina beadChip, processing as such")
    if(length(gseIDVec) > 1){
        print("WARNING: We do not currently support combining of multiple Illumina geoIDs")
        quit('no', as.numeric(exitCodes['usageError']))
    }
    gseID <- gseIDVec[1]
    library(lumi)
    # in some cases I'm able to generate a key to connect GSMs to the IDs in the illumina raw data
    # the default is that I cannot though
    descriptionList = NA
    # further down I may write over this
    # The names of the raw data files vary, but the consistency seems to be gseID*.txt.gz
    # This is a little hacky, but it seems like it is working
    if(idType=='geo'){
        possibleFiles <- list.files(path = paste0('./',gseID), pattern = ".txt.gz", ignore.case = TRUE)
        rawDataFile <- possibleFiles[grep(gseID, possibleFiles)]
        
        if(length(rawDataFile)==0){
            untar(paste0(gseID,"/",gseID,"_RAW.tar"))
            possibleFiles <- list.files(path = paste0('./',gseID),pattern = ".txt.gz", ignore.case = TRUE)
            rawDataFile <- possibleFiles[grep(gseID, possibleFiles)]
        }
        
        if(length(rawDataFile) > 1){
            gplMatches <- grep(gplName, rawDataFile)
            if( length( gplMatches ) == 1){
                rawDataFile <- rawDataFile[gplMatches]
            }else{
                print(paste("There seems to be multiple raw data files for",
                             gseID,
                             "that match the current pattern, and of those",
                             length(gplMatches),
                             "include the preferred GPL.",
                             "Because we don't know which file has the data",
                             "we cannot process this data."
                            )
                     )
                quit('no', as.numeric(exitCodes['illuminaBeadChipError']))
            }
        }
        
        rawDataFileToUse <- paste(gseID,rawDataFile, sep = "/")
        headerFile <- paste(gseID,"header.txt", sep = "/")

    }else if(idType=='ae'){
        rawDataDir <-"illuminaRawData"
        system(paste0("mkdir -p ", rawDataDir))
        unpacked_files <- unpackAERawData(gseID, rawDataDir)
        possibleFiles <- list.files(path = paste0('./',rawDataDir),
                                    pattern = ".txt",
                                    ignore.case = TRUE
                                   )
        if(length(possibleFiles) == 1){
            rawDataFileToUseNotZipped <- paste0('./',rawDataDir,"/", possibleFiles[1])
            removeBadIlluminaHeader(possibleFiles[1], rawDataFileToUseNotZipped)
            system(paste("gzip", rawDataFileToUseNotZipped))
            rawDataFileToUse <- paste0(rawDataFileToUseNotZipped, ".gz")
        }else{
            rawDataFileToUseNotZipped <- "illuminaRawData.txt"
            rawDataFileToUse <- paste0(rawDataFileToUseNotZipped, ".gz")
            removeBadIlluminaHeader(possibleFiles[1], rawDataFileToUseNotZipped)
            # Now add in all of the other files via join
            # I want to ensure that we have the same number of lines before and after
            linesInOriginalFile <- getLineNumberFromFile(rawDataFileToUseNotZipped) 
            for(i in 2:length(possibleFiles)){
                possibleFile <- possibleFiles[i]
                temp_file <- "temp_illuminaRawData.txt"
                removeBadIlluminaHeader(possibleFile, temp_file)
                system(paste("join",
                              rawDataFileToUseNotZipped,
                              temp_file,
                              "| sed \"s/ /\t/g\" > temp.txt"
                              )
                      )
                linesInNewlyMergedData <- getLineNumberFromFile("temp.txt")
                if( linesInNewlyMergedData != linesInOriginalFile ){
                    print(paste("WARNING: In attempting to join multiple raw Illumina data files,",
                                "some rows (probes) were lost.",
                                "Proceeding anyway, but this could bias results."
                               )
                         )
                }
                # If we decide not to proceed anyway,
                # this line would need to be commented out,
                # and what to do with the extra data needs to be decided.
                system(paste0("mv temp.txt ", rawDataFileToUseNotZipped))
            }
            system(paste("gzip ", rawDataFileToUseNotZipped))
        }
        headerFile <- "header.txt"
    }

    # I ran into some issues with the raw Illumina data not being saved properly,
    # or at least with the correct headers, so I need to verify things
    system(paste0('zcat ', rawDataFileToUse, ' | head -n1 > ', headerFile))
    rawDataColumnNames <- as.vector(as.matrix(read.csv(headerFile,
                                                       sep = "\t",
                                                       header = FALSE
                                                      )[1,]
                                              )
                                    )
    
    detectionColumns <- findIlluminaDetectionCols(rawDataColumnNames)
    
    # another one I've been seeing is that the first line actually has the GSMs
    # and the 2nd line is the expected header, so I'll try that
    if(length(detectionColumns) == 0){
        print("Trying to see if the 2nd line might actually be the needed header")
        
        # pull out the 2nd line and treat that as the header file as above
        system(paste0('zcat -f ',
                      rawDataFileToUse,
                      ' | tail -n +2 | head -n 1 > ',
                      headerFile,
                      "2ndTry"
                     )
              )
        rawDataColumnNamesSecondTry <- as.vector(as.matrix(read.csv(paste0(headerFile,
                                                                          "2ndTry"
                                                                    ),
                                                                    sep = "\t",
                                                                    header = FALSE
                                                                    )[1,]
                                                 ))
        # There was atleast one weird dataset that had a prefix to remove before the avg_signal, etc
        if(any(grepl("BeadStudio:", rawDataColumnNamesSecondTry))){
            print("Removing 'BeadStudio:' prefix from Illumina headers")
            # change it in R, but also edit the header file
            rawDataColumnNamesSecondTry  <- gsub("BeadStudio:",
                                                 "",
                                                 rawDataColumnNamesSecondTry,
                                                 ignore.case = TRUE
                                                )
            # and to add a unique identifier for each column
            rawDataColumnNamesSecondTry <- paste(seq(1,
                                                     length(rawDataColumnNamesSecondTry)
                                                    ),
                                                 rawDataColumnNamesSecondTry,
                                                 sep = "_"
                                                ) 
            # The issue is that write.table is considering each entry to be it's own line;
            # so I switched the eol to a tab...
            write.table(rawDataColumnNamesSecondTry,
                        file = paste0(headerFile, "2ndTry"),
                        quote = FALSE,
                        row.names = FALSE,
                        col.names = FALSE,
                        eol = "\t"
                       )
        }
        
        detectionColumns <- findIlluminaDetectionCols(rawDataColumnNamesSecondTry)
        
        # if we've had success now we'll have detection columns
        # and we'll want to move things around so that the original header line
        # is removed and the 2nd line becomes the real header
        # I've also been running into the situation where there are no detection columns,
        # and just signal columns
        # this seems to be especially true for AE data
        # I'll account for that here
        # started using avg_signal b/c there were some columsn with max_signal or min_signal
        if(length(detectionColumns) > 0 |
           any(grepl("avg_Signal",
                     rawDataColumnNamesSecondTry,
                     ignore.case = TRUE 
               ))
           ){
            print("It looks like the 2nd line was the needed header line. Moving files to address this")
            system(paste0('mv ',  headerFile, " ", headerFile, "_orig"))
            system(paste0('mv ',  headerFile, "2ndTry ", headerFile))
            # In theory the header I want should just be the 2nd line (n+2), 
            # but there is the one dumb case where I edited the header, 
            # so I'll add that header to this file just in case
            system(paste('zcat', rawDataFileToUse, '| tail -n+3 > tempFile'))
            system(paste('cat', headerFile, 'tempFile | gzip -c >',  rawDataFileToUse))
            system('rm tempFile')
            
            # rawDataColumnNames contains the GSMs that we need
            gsmsForIllumina <- rawDataColumnNames
            rawDataColumnNames <- rawDataColumnNamesSecondTry
            
            # I want to connect the GSMs to their sample names
            if( grepl('geo', idType, ignore.case = TRUE)){
                indsToUse <- grep("GSM", gsmsForIllumina, ignore.case = TRUE)
            }else{
            # I don't actually know what's best here,
            # but for the time being I'll use all those that are headed by avg_signal
                indsToUse <- grep( "Avg_Signal", rawDataColumnNames, ignore.case = TRUE )
            # If that doesn't work... I'll just include all of them?
                if(length(indsToUse) <= 2 ){ # 2 b/c that's the minimum number of samples
                    print(paste("Not sure what to use as the description list",
                                "because there are less than 2 columns headed by avg_signal.",
                                "Using all columns:"
                               ))
                    print(paste(gsmsForIllumina, collapse = " "))
                    indsToUse <- seq(1, length(gsmsForIllumina))
                }
            }
            
            descriptionList <- rawDataColumnNames[indsToUse]
            descriptionList <- gsub("_AVG_SIGNAL", "", descriptionList, ignore.case = TRUE)
            # I've seen it both ways
            descriptionList <- gsub(".AVG_SIGNAL", "", descriptionList, ignore.case = TRUE)
            # in some cases this results in no distinct names for these columns,
            # which means we can't select them.
            # that's not good, but it's easy enough to fix
            if ( length( unique(descriptionList) ) != length(descriptionList) ){
                print(paste("Non-unique names found for Illumina header description list.",
                            "Replacing with unique identifiers"
                           ))
                descriptionList <- seq(1, length(descriptionList))
            }
            names(descriptionList) <- gsmsForIllumina[indsToUse]
        }
    }

    idRefColumn <- getIDRefColumn(rawDataColumnNames)
    avgSignalColumn <- getSignalColumn(rawDataColumnNames, gseID, rawDataFileToUse)
    # for debugging, I want to see what this is picking up for avg signal and detection
    if(debug){
        print("For debugging Illumina beadChip data. Raw data headers:")
        print(rawDataColumnNames)
    }
    print(paste("Avg Signal column number:", length(avgSignalColumn)))
    print(paste("Detection column number:", length(detectionColumns)))
    # not all of the illumina data we're getting is saved as expected for lumiR
    # some of them are being saved more as an eset,
    # so in that case we handle things differently
    # these options, will, I hope,
    # deal with the situation where the data was pretty much saved as an eset
    # hopefully this actually works
    if( length( avgSignalColumn ) > 0 & length( detectionColumns ) == 0 ){
        toReport <- paste("It looks like the Illumina data was saved",
                          "with only the expression data.",
                          "It will not work with LumiR now, so instead we'll try RMA"
                         )
        if(debug){print(toReport)}
        rawData <- read.table(gzfile(rawDataFileToUse),
                              header = TRUE,
                              sep = "\t",
                              strip.white = TRUE
                              )
        library(affy)
        return(rma(rawData))
    }else if( length( avgSignalColumn ) == 0 & length( detectionColumns ) == 0 ){
        toReport <- "It looks like the Illumina data was not saved as expected. Try toggling to fallback"
        if(debug){print(toReport)}
        quit('no', as.numeric(exitCodes['illuminaBeadChipError']))
    }else{
        # there should be one column of each for each sample
        if(length(avgSignalColumn) != length(detectionColumns)){
            # check to see if the number of non ID_REF 
            # or Detection columns == the number of detection columns.
            # If so, simply append with _AVG_SIGNAL
            firstNum <- length(rawDataColumnNames) - length(detectionColumns) - length(idRefColumn)
            if(firstNum == length(detectionColumns)){
                ind <- -1*c(idRefColumn, detectionColumns)
                appendSignalCols(rawDataColumnNames, ind, rawDataFileToUse)
            }else{
                print(paste("WARNING: In processing Illumina BeadChip raw data,",
                            "there are an unequal number of 'Detection' and",
                            "'AVG_SIGNAL' columns as well as 'Detection' and other columns.",
                            "We're currently unable to fix this.",
                            "This is not always an issue however."
                           ))
            }
        }
        dos2unixGzippedFile(rawDataFileToUse)
        print(paste("calling lumiR with", rawDataFileToUse))
        if(debug) print(system(paste("zcat -f", rawDataFileToUse, "| head"),
                               intern = TRUE
                        ))
        rawData <- lumiR(rawDataFileToUse)
    }
    print("Summary QC for raw data: ")
    summary(rawData,'QC')
    print("Normalizing using lumiExpresso")
    esetToReturn <- lumiExpresso(rawData)
    return(list(eset = esetToReturn, desc = descriptionList))

}

# change the header on the rawDataFileToUse
appendSignalCols <- function(rawDataColumnNames, ind, rawDataFileToUse){
    print(paste('IND', ind))
    rawDataColumnNames[ind] <- paste0(rawDataColumnNames[ind], "_AVG_SIGNAL")
    # The issue is that write.table is considering each entry to be it's own line;
    # so I switched the eol to a tab...
    tempHeaderFile <- 'headerToUse.txt'
    write.table(rawDataColumnNames,
                file = tempHeaderFile,
                quote = FALSE,
                row.names = FALSE,
                col.names = FALSE,
                eol = "\t"
               )
    #...but that means I need to add a newline to the header file, or lose the first 
    system(paste("echo '' >>", tempHeaderFile))
    tempFile <- "tempRawDataFile.txt"
    system(paste("zcat", 
                 rawDataFileToUse,
                "| tail -n+2 >",
                tempFile
                ))
    system(paste("cat", tempHeaderFile, tempFile, "| gzip >", rawDataFileToUse))
    system(paste("rm", tempFile, tempHeaderFile))
}

# just in case I'll strip out the 'wrong' EOLs
dos2unixGzippedFile <- function(file){
    system(paste('zcat',
                 file,
                 "| dos2unix | gzip > temp"
                 ))
   system(paste("mv temp", file))
}

getSignalColumn <- function(rawDataColumnNames, gseID, rawDataFileToUse){
    avgSignalColumn <- grep("AVG_SIGNAL", rawDataColumnNames, ignore.case = TRUE)
    if( length( avgSignalColumn ) == 0 ){
        avgSignalColumn <- grep("SIGNAL", rawDataColumnNames, ignore.case = TRUE)
    }
    # I've come across at least one instance where they labeled the signal as sample
# the one I cam across that this helped failed b/c there was not way to connect the 
# names in the files to the GSMs
#    if( length( avgSignalColumn ) == 0 ){
#        avgSignalColumn <- grep("SAMPLE", rawDataColumnNames, ignore.case = TRUE)
#        appendSignalCols(rawDataColumnNames, avgSignalColumn, rawDataFileToUse)
#    }
    if( length( avgSignalColumn ) == 0 ){
        avgSignalColumn <- illuminaExceptionDatasets(rawDataColumnNames, gseID)
    }
    return(avgSignalColumn)
}

# there are some datasets that are just one offs and don't match any rules
# I'll consolidate those here
illuminaExceptionDatasets <- function(rawDataColumnNames, gseID){
    avgSignalColumn <- c()
    useAllCols <- c('GSE42572')
    toIgnore <- getIDRefColumn(rawDataColumnNames)
    if (any(grepl(gseID, useAllCols))){
        print(paste('Found a dataset that is known the have issues',
                     gseID,
                     '. We know the right thing to do here is use all',
                     ' columns as signal columns.'
                     ))
        avgSignalColumn <- rawDataColumnNames[-1*toIgnore]
    }
    return(avgSignalColumn)
}

getIDRefColumn <- function(rawDataColumnNames){
    # this should be the first column every time,
    # but just in case we'll grep,
    # but if we don't find it we'll just assume it's the first column
    if( any( grepl( "ID_REF", rawDataColumnNames) ) ) { 
        idRefColumn <- grep("ID_REF", rawDataColumnNames)
    }else if( any( grepl( "TargetID", rawDataColumnNames) ) ){
        idRefColumn <- grep("TargetID", rawDataColumnNames)
    }else if(any( grepl( "REF", rawDataColumnNames) ) &
             length(grep("REF", rawDataColumnNames)) == 1
             ){
        idRefColumn <- grep("REF", rawDataColumnNames)
    }else{
        print(paste("Unable to find definitive ID_REF column for Illumina data.",
                    "Defaulting to the first column"
                   ))
        idRefColumn <- 1
    }
    return(idRefColumn)
}


removeBadIlluminaHeader <- function(infile, outfile){
# I hadn't seen it in Illumina data from GEO,
# but in the examples from AE that I've been able to work with,
# there was a header to the data.
# This is to deal with that.
# I also had to deal with the possibilities of spaces in the files names,
# hence the \'d out quotes
    numberOfLinesToIgnoreAtTop <- system(paste0("grep -nr \"Detection\" \"",
                                                infile,
                                                "\" | cut -d: -f1"
                                                ),
                                         intern = TRUE
                                        )
    system(paste0("tail -n+",
                  numberOfLinesToIgnoreAtTop,
                  " \"",
                  infile,
                  "\" > ",
                  outfile
                  )
           )
}

# a small support function for the illumina files
findIlluminaDetectionCols <- function(rawDataColumnNames){
    detectionColumns <- grep("Detection", rawDataColumnNames)
    # because people can't spell, if we don't find any detection,
    # we also search for Dection; it fixed at least one datasets
    if(length(detectionColumns)==0){
        print("Trying a misspelling of detection to see if that is able to find columns")
        detectionColumns <- grep("Dection", rawDataColumnNames)
    }
    return(detectionColumns)
}


file_pattern_in_wd <- function(pattern){
    matches <- Sys.glob(pattern)
    return(length(matches) > 0)
}

# Agilent also has to be processed with their package
processAndNormAgilentChip <- function(gseIDVec, idType){
    print("Microarray appears to be from Agilent, processing as such")
 
    # I have pretty much just followed this:
    # http://master.bioconductor.org/packages/release/bioc/vignettes/agilp/inst/doc/agilp_manual.pdf
    library(agilp)
    library(Biobase)
    
    # set up necessary files
    agilentDir <- "agilentProcessingDir/"
    system(paste0('mkdir -p ', agilentDir))
    inputdir <- paste0(agilentDir, 'initialData/')
    rawDataDir <- paste0(agilentDir, 'rawData/')
    normalizedDataDir <- paste0(agilentDir, 'normdData/')
    system(paste0('mkdir -p ', rawDataDir))
    system(paste0('mkdir -p ', normalizedDataDir))
    system(paste0('mkdir -p ', inputdir))
    
    # now unpack and itialize everything for all of the samples
    esetlist <- list()
    for(gseID in gseIDVec){
        # unpack the supplemental files    
        if(idType == 'geo' || idType == 'geo-mixed'){
            untar(paste0(gseID, "/", gseID, "_RAW.tar"))
            full_inputdir_path <- paste0(getwd(),'/', inputdir)
            if (file_pattern_in_wd('GSM*.gz')){
                system(paste('mv', 'GSM*.gz', inputdir))
                system(paste0('gunzip ', inputdir, 'GSM*.gz'))
            }else if (file_pattern_in_wd('GSE*.gz')){
                system(paste('mv', 'GSE*.gz', inputdir))
                system(paste0('gunzip ', inputdir, 'GSE*.gz'))
            }else{
                print(paste("WARNING: We did not find the expected files in the agilent download.",
                            "These are file files in", inputdir))
                print(list.files())
            }
            unpackedfiles <- ' '
        }else if(idType=='ae' || idType=='ae-mixed'){
            unpackedfiles <- unpackAERawData(gseID, inputdir)
        }
        if (grepl("gpr$", unpackedfiles, perl = TRUE)){
            print("Found gpr files in the unpacked files, so processing as such")
            esetlist[[length(esetlist) + 1 ]] <- gprFilesToEset()
        }else{
            # run AAProcess, step one in agilent analysis, which basically just loads the data
            AAProcess(input = inputdir, output = rawDataDir)
        }
    }
    if (length(esetlist) > 1){
    # combine the esets and return them
        return(combineEsets(esetlist))
    }else if(length(esetlist) == 1){
        return(esetlist[[1]])
    }
    # now we start on the normalization
    # before loess, we need a baseline
    outputbase <- paste0(agilentDir, 'outputbase.txt')
    Baseline(NORM = "LOG",
              allfiles = "TRUE",
              r = 2, A = 2, B = 3,
              input = rawDataDir,
              baseout = outputbase
             )
    # Now actually run the normalization
    AALoess(input = rawDataDir,
            output = normalizedDataDir,
            baseline = outputbase,
            LOG = "TRUE"
           )
    
    # Read in the data and convert it to an eset
    if (useBothColors){
        patternToUse <- glob2rx("n*")
    }else{
        patternToUse <- glob2rx("ng_*")
    }
    normalizedDataFile <- list.files(path = normalizedDataDir, pattern = patternToUse)
    normalizedData <- read.csv(paste0(normalizedDataDir, normalizedDataFile[1]), sep = "\t", header = TRUE)
    for (i in 2:length(normalizedDataFile)){
        normalizedData <- merge(normalizedData, read.csv(paste0(normalizedDataDir, normalizedDataFile[i])
                                                         , sep = "\t", header = TRUE)
                                , by = 1)
    }
    #clean up the column names
    newColumnNames <- strsplit(colnames(normalizedData),"_", fixed = TRUE)
    gsms <- sapply(newColumnNames[2:length(newColumnNames)], '[[', 2) # this gets the first sub element of every entry in a list
    if (useBothColors){
        prefix <- sapply(newColumnNames[2:length(newColumnNames)], '[[', 1)
        for (i in 1:length(gsms)){
            gsms[i] <- paste(prefix[i], gsms[i], sep = "_")
        }
    }
    colnames(normalizedData) <- c("ID_REF", gsms)
    # now convert to eset
    assayData <- as.matrix(normalizedData[,2:ncol(normalizedData)])
    rownames(assayData) <- normalizedData$ID_REF
    return(ExpressionSet(assayData = assayData))
}

unzipGpr <- function(){
    if (any(grepl("gpr.gz$", list.files(), perl = TRUE))){
        if (debug) print('gunzipping GPR files.')
        system("gunzip *.gpr.gz")
    }
}

# Check the eset for outliers, and save it to an RDS
processAndSaveEset <- function(idType, eset, geoID, fullGeoID){
### checkForMissingData expects a matrix and currently can remove the relevant rows
### then return the the cleaned matrix.
### But how to do that for an eset?
### I'm trying to take the rows from this new one to filter the whole eset...not sure it'll work
    exprsEset <- exprs(eset)
    filteredExprEset <- checkForMissingData(exprsEset)
    filter <- rownames(exprsEset) %in% rownames(filteredExprEset)
    if(debug){
        print(paste('before filtering missing values the eset had the following dimensions:',
                    paste(dim(eset), collapse = " ")
             ))
    }
    eset <- eset[filter,]
    if(debug){
        print(paste('AFTER filtering missing values the eset had the following dimensions:',
                    paste(dim(eset), collapse = " ")
             ))
    }
    # check if the data needs to be log2
    checkTopQuantile(eset, top1PercentSignalThreshold)
    needsLogging <- read.table("needsLog2.txt",
                               colClasses = "logical",
                               header = TRUE
                              )[1,1]
    # Now detect and remove outliers
### We previously had a try block here for historical reasons, 
### but recently it was just allowing AQM to fail and not tell us.
### as of sprint119fixes, the try() has been removed
    runArrayQC(eset, needsLogging, idType, fullGeoID)

    if(grepl("orig", idType)){
        # Now detect and flag outliers
        print("Saving pre-normalized data")
        saveRDS(exprs(eset), file = paste0(geoID,"_asFromGeo.rds"))
    }else{
        print("Saving data we normalized")
        saveRDS(eset, file = paste0(geoID, "_normalized.rds"))
    }

    if( idType != 'ae-orig'){
        # remove the supplementalData raw data, including gpl information, as everything we need is currently saved as RDS files
        cleanUpFiles(idType)
    }
}

# a little function that combines esets if there are multiple
combineEsets <- function(esetList){
    eset <- esetList[[1]]
    if(length(idVec) > 1){
        for(e in 2:length(esetList)){
            temp <- combine(eset, esetList[[e]])
            eset <- temp
        }
    }
    return(eset)
}


# Go through the meta data and get it into a useable format
parseMetaData <- function(gsms, phenoData){
    # first get rid of white spaces in 
        
    for (j in 1:ncol(phenoData)){
        phenoData[,j] <- as.character(phenoData[,j])
    }
    for (j in 1:ncol(phenoData)){
        for (i in 1:nrow(phenoData)){
            phenoData[i,j] <- gsub("[[:space:]]", " ", phenoData[i,j])
        }
    }
    # we used to do this in a different way such that parsing actually happened, not this is mostly historical
    phenoString <- apply(phenoData, 1, paste, collapse="\t")
    with_gsms <- cbind(as.character(gsms),as.character(phenoString))
    if (useBothColors){
        toReturn <- data.frame(gsms = c(paste0('nr_', as.character(gsms)),
                                        paste0('ng_', as.character(gsms))
                                        ),
                               str = rep(as.character(phenoString), 2),
                               colors = c(rep('Red', length(phenoString)),
                                          rep('Green', length(phenoString))
                                         )
                               )
        if(debug){
            print(head(toReturn))
            print(tail(toReturn))
        }
        return(toReturn)
    }else{
        return(with_gsms)
    }
}


# Safely save the metaData to a file
saveMetaData <- function(metaData, idType, geoID){
    # we assume nothing is an outlier,
    # then if we find some later we update to True
    metaData <- cbind(metaData,rep('Outlier: False', nrow(metaData)))
    print(paste0("There are ",
                 nrow(metaData),
                 " entries of meta data to enter into the DB"
                 ))
    # Add the meta data to the database
    metaDataFile <- get_meta_data_file_name(idType, geoID)
    if(file.exists(metaDataFile)){
        system(paste("rm", metaDataFile))
    }
    saveRDS(metaData, file = paste0(metaDataFile,'.rds'))
}

get_meta_data_file_name <- function(idType, geoID){
    if(grepl('geo', idType)){
        fn <- paste0(microarrayDir,'/', geoID, "_metadata.tsv")
    }else if(grepl('ae', idType)){
        fn <- paste0(geoID, "_metadata.tsv")
    }else{
        print("get_meta_data_file_name was not provided the proper arguments. Quitting.")
        quit('no', as.numeric(exitCodes['usageError']))
    }
    return(fn)
}

# write the mapping data to an RDS for use in sigGEO
saveMappingInfo <- function (type, gpl, outputFile, gse=''){
    if(type=="geo"){
        gplTable <- Table(gpl)
# a one off within a one off where I happen to know the column called ID is our best bet for mapping.
# changing it's name to something the mapper will later pick up on
        if (Meta(gpl)$geo_accession == 'GPL4133' && gse == 'GSE29801'){
            print(paste("For",
                     gse,
                    "we have found that the first column is best for mapping. Manipulating to ensure it is used for mapping"
                   ))
            colnames(gplTable)[1] <- 'ProbeName'
        # This is a one off situation where the unigene is called something really vague that we don't want to add to waht we match on b/c of how vague it is
        } else if (Meta(gpl)$geo_accession == 'GPL538'){
            colnames(gplTable)[grep('cluster_id', colnames(gplTable))] <- 'unigene'
        } else if (Meta(gpl)$geo_accession == 'GPL4133' ||
            Meta(gpl)$geo_accession == 'GPL21810'
           ){
            print(paste("For",
                         Meta(gpl)$geo_accession,
                        "we have found that SPOT_ID is not useful, so we're removing that header column"
                       ))
            colnames(gplTable)[grep('SPOT_ID', colnames(gplTable))] <- 'formerlySpotID'
        }else if (Meta(gpl)$geo_accession == 'GPL17425'){
            print("For GPL17425 we have found that SEQ_ID is actually GB_ACC. Renaming now.")
            colnames(gplTable)[grep('SEQ_ID', colnames(gplTable))] <- 'GB_ACC'
        }else if (Meta(gpl)$geo_accession == 'GPL891' ||
                  (Meta(gpl)$geo_accession == 'GPL4134' && gse == 'GSE36496')
                  ){
            print(paste("For", gse, "when using",
                         Meta(gpl)$geo_accession,
                        "we have found that both SPOT_ID and name are not useful, so we're renaming those header columns"
                       ))
            colnames(gplTable)[grep('NAME', colnames(gplTable))] <- 'ignore'
            colnames(gplTable)[grep('SPOT_ID', colnames(gplTable))] <- 'ignore2'
        }
        saveRDS(list(table = gplTable,
                     chipType = Meta(gpl)$title
                     ),
                 file = outputFile)
        print("Mapping data saved")
        return(Meta(gpl)$title)
    }else if (type == 'ae'){
        if(! is.null(gpl$chipType) ){
            if(grepl('agilent', gpl$chipType, ignore.case = TRUE)){
                if(any(grepl("Composite.Element.Name", colnames(gpl$table)))){
                    print("Attempting to change the column names for the mapping info of an agilent chip")
                    colToSwitch <- grep("Composite.Element.Name", colnames(gpl$table))
                    colnames(gpl$table)[colToSwitch] = "ENTREZ_GENE_ID"
                }
            }
        }
        saveRDS(gpl, file = outputFile)
        print("Mapping data saved")
    }else{
        print("WARNING:Something went wrong with saveing the mapping data, and it looks like this is not data from GEO or ArrayExpress.")
        quit('no', as.numeric(exitCodes['usageError']))
    }
}


# this gets called if the data necessary for sigGEO is already present
alreadyDone <- function(){
    print("metaGEO.R was already successfully run. Proceed to sigGEO.R")
    quit('no', as.numeric(exitCodes['alreadyDone']))
}

# To see if the data is already log2 transformed we check what the 99th quantile is
checkTopQuantile <- function (eset, top1PercentSignalThreshold){
    toReturn <- NULL
    if(debug){
        print(head(eset))
        print(head(exprs(eset)))
    }
    top1PercentSignal <- NULL
    try(top1PercentSignal <- quantile(as.numeric(exprs(eset)), 0.99, na.rm = TRUE))
    if(is.null(top1PercentSignal)){
        print("Something went wrong with checking the quantile")
        print("This is what the exprs eset looks like:")
        print(head(as.numeric(exprs(eset))))
        quit('no', as.numeric(exitCodes['unableToFindDataError']))
    }
    if(top1PercentSignal < top1PercentSignalThreshold){
        print(paste0("Assuming data is already log2, because the 99th percentile of signal is ",
                     top1PercentSignal,
                     " which is BELOW the threshold of ",
                     top1PercentSignalThreshold))
        toReturn <- FALSE
    }else{
        print(paste0("Assuming data is NOT log2, because the 99th percentile of signal is ",
                     top1PercentSignal,
                     "while ABOVE the threshold of ",
                     top1PercentSignalThreshold))
        toReturn <- TRUE
    }
    write.table(toReturn,
                file = "needsLog2.txt",
                quote = FALSE,
                row.names = FALSE
               )
}

# run the arrayQualityMetics package and process the data to identify any outliers identified
runArrayQC <- function(eset, needsLogging, idType, geoID){
    print("Checking for outliers")
    library(arrayQualityMetrics)
    qcResults <- arrayQualityMetrics(expressionset = eset,
                                     outdir = './',
                                     force = TRUE,
                                     do.logtransform = needsLogging)
    heatmapOutliers <- names(qcResults$modules$heatmap@outliers@which)
    boxPlotOutliers <- names(qcResults$modules$boxplot@outliers@which)
    # This results in a vector of indexes of gsms that are outliers in both measures
    strictOutliers <- Reduce(intersect, list(boxPlotOutliers, heatmapOutliers))
    metaDataFile <- get_meta_data_file_name(idType, geoID)
    writeMetaDataWithOutliers(strictOutliers, metaDataFile, idType)
}
writeMetaDataWithOutliers <- function(strictOutliers, metaDataFile, idType){
    # we're now going to add this data to the previous metaData
    metaData <- readRDS(paste0(metaDataFile,'.rds'))
    if(length(strictOutliers) > 0 && is.na(any(match(strictOutliers, metaData[,1])))){
        if(grepl('geo', idType)){
            strictOutliers <- just_gsms(strictOutliers)
        }else{
            print(paste("WARNING: Unable to match outlier names to column names for",
                        "non-GEO data type. Quitting."
                        )
                 )
            quit('no', as.numeric(exitCodes['unableToFindDataError']))
        }
    }
    metaData[metaData[,1] %in% strictOutliers, ncol(metaData)] <- 'Outlier: True'
    write.table(metaData,
                file = metaDataFile,
                append = TRUE,
                sep = "\t",
                quote = FALSE,
                col.names = FALSE,
                row.names = FALSE
                )
    print(paste0("Outlier GSMs detected: ", paste(strictOutliers, collapse = ", ")))
}

# Remove the large amount of raw data downloaded
cleanUpFiles <- function(idType){
    unlink('*.tsv.rds')
    try(unlink('*.soft*'))
    try(unlink('GPL*'))
    try(unlink('GSM*.gz'))
    try(unlink('*.CEL'))
    if(idType=='geo' || idType=='geo-mixed'){
        try(unlink(gseID))
    }
    if(file.exists( "agilentProcessingDir" )){
        try(unlink("agilentProcessingDir"))
    }else if( file.exists( "illuminaRawData" ) ){
        try(unlink("illuminaRawData"))
    }
}
