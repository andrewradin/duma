# lots of miscellaneous settings here
getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) return('None')
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}


thisScriptsPath <- getScriptPath()
if(thisScriptsPath != 'None'){
    source(paste0(thisScriptsPath, '/supportFunctionsForAllRCode.R'))
    source(paste0(thisScriptsPath, '/settings.R'))
}

#========================================================================================
# General support functions
#========================================================================================#

# A single command to load the necessary libraries
loadLibraries <- function(runType){
    library(RMySQL)
    if(runType=='meta'){
        library(Biobase)
        library(oligo)
    }else if(runType=='sig' || runType=='seq'){
        library(dplyr)
        library(pheatmap)
        library(RColorBrewer)
        if(runType=='sig'){
            library(org.Hs.eg.db)
            library(limma)
        }
    }
}

# This takes care of reading in arguments, and assigning them.
parseCommands <- function(command){
    args <- commandArgs(trailingOnly = TRUE)
	required_args <- c('idType','DSID{:mappingID}')
	if (command == 'sigGEO.R') {
		required_args <- c(required_args,'tisID','indir','outdir','pubdir')
	}else{
		required_args <- c(required_args,'0','indir')
	}
	min_length <- length(required_args)
	max_length <- min_length + 1
    if (length(args) < min_length || length(args) > max_length) {
		print(paste(c("usage:",command,required_args,"{testing}"),collapse=' '))
        quit('no', as.numeric(exitCodes['usageError']))
    }
    idType <- args[1]
    if(idType != "ae" && idType != "geo" && idType != "geo-orig" && idType != "ae-orig" && idType != "geo-seq" && idType != "ae-seq" && idType != 'ae-mixed' && idType != 'geo-mixed'){
        print("WARNING: ID type must be ae for arrayExpress or geo for GEOquery, no other options are currently supported")
        quit('no', as.numeric(exitCodes['usageError']))
    }
    if (idType == 'ae-mixed' || idType == 'geo-mixed'){
        print("WARNING: Attempting to pull apart a 2 color array. This isn't a very common setting, be sure you meant to do this")
        useBothColors <- TRUE
    }else{
        useBothColors <- FALSE
    }    
    fullGeoID <- args[2] # For ease I'm keeping this as geoID even though it may be arrayExpress IDs as well
    tisID <- as.integer(args[3])
    testTheScript <- FALSE
    
    if ( length(args) == min_length+1) {
        if (args[min_length+1] == "testing"){
            testTheScript <- TRUE
            print("Test under way")
        }else{
            print("WARNING: Optional argument must be 'testing'. Quitting.")
            quit('no', as.numeric(exitCodes['usageError']))
        }
    }
    dirList <- setUpGEOPaths(fullGeoID)
	if (command == 'sigGEO.R') {
		dirList <- c( list( indir=args[4], outdir=args[5], pubdir=args[6] ),dirList)
	}else{
		dirList <- c( list( indir=args[4] ),dirList)
	}
    
    # To support the use of specific GPLs, we now support GEOIDs:GPLIDs
    # The directory name(created aboev) includes the GPL, but we need the actual GEO ID
    # Not all GEO IDs will have this though, so we need to test for it
    if(grepl(":", fullGeoID)){
        idParts <- strsplit(fullGeoID, ":")[[1]]
        preferredGpl <- idParts[2]
        geoID <- idParts[1]
    }else{
        preferredGpl <- NA
        geoID <- fullGeoID
    }
    
    return( c( list( testTheScript = testTheScript,
                     idType = idType,
                     fullGeoID = fullGeoID,
                     geoID = geoID,
                     tisID = tisID,
                     preferredGpl = preferredGpl,
                     useBothColors = useBothColors
                   ),
               dirList
            ) )
}

setUpGEOPaths<- function(geoID){
    microarrayDir<-paste(staticdir, geoID, sep = "/")
    if(! file.exists(microarrayDir)){
        dir.create(microarrayDir)
    }
    
    needsLoggingFile = paste0(microarrayDir, "/needsLog2.txt")
    outlierFile <- paste(microarrayDir, "outlierGsms.csv", sep = "/")
    return( list( microarrayDir = microarrayDir,
                  needsLoggingFile = needsLoggingFile,
                  outlierFile = outlierFile
              ) )
}
just_gsms <- function(x){
    if (! any(grepl('GSM', x))) return(x)

    to_strip <- unlist(lapply(x, function(y) gsub("^GSM[0-9]+", "", y, perl = TRUE)))

    if (all(to_strip == "")) return(x)

    to_ret <- x
    for (i in 1:length(x)){
        if (grepl('GSM', x[i]) && to_strip[i]!=''){
            to_ret[i] <- gsub(to_strip[i], "", x[i], fixed = TRUE)
        }
    }

    return(to_ret)
}

checkForMissingData <- function(x){
    if (ignoreMissing){
        print('WARNING: You selected to ignoring missing probes. This means you should be very careful about the number of total proteins')
        x <- na.omit(x)
    }
    if (any(is.na(x))){
        print('WARNING: missing data. This is very unusual and suggests bad data. We would suggest not using this dataset. Quitting.')
        quit('no', as.numeric(exitCodes['unexpectedDataFormat']))
    }
    return(x)
}
