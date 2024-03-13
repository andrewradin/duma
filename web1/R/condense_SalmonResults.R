library(tximport)

# from where this script is always called, this is the case and makes subsequent calls easier
microarrayDir <- '..' 
source('/home/ubuntu/2xar/twoxar-demo/web1/R/metaGEO_supportFunctions.R')
# rather than passing in arguments we'll just deduce them
temp <- strsplit(getwd(), '/')[[1]]
geoID <- temp[length(temp) - 1]
if(startsWith(geoID, 'G')){
    idType <- 'geo-seq'
}else{
    idType <- 'ae-seq'
}

print(paste("Running condenseSalmon from", getwd()))

# This assumes our working directory is just above the publish/<geoID> directory...
customSettings <- paste('..', 'settings.R', sep = "/")
# We need customSettings to know where the uniprot mapping file is.
source(customSettings)

print(paste("Using uniprot conversion from", EnsemblTRSToUniprotMap))

tx2gene <- read.table(EnsemblTRSToUniprotMap, header = FALSE)


dirs <- list.dirs(recursive = FALSE)
dirs <- dirs[ !grepl("rawFastq", dirs) ]
files <- paste(dirs, 'quant.sf', sep = "/")
names(files) <- gsub("^./", "", dirs, perl = TRUE)
incompleteSamples <- names(files)[!file.exists(files)]
completeFiles <- files[file.exists(files)]

### The approach below misses any samples that did not even reach the step where the dirs are created
### To address that we'll read in the initial download file
### XXX Everywhere except here get_meta_data_file_name should not append the microarrayDir
### XXX to the metaDataFile for AE samples, but in this case it needs to.
### XXX Therefore to keep the hackyness to one place, we'll pass in a dummy idType
metaDataFile <- get_meta_data_file_name('geo-seq', geoID)
metaDataRds <- paste0(metaDataFile,'.rds')

if(idType == 'geo-seq'){
    incompleteSamples <- gsub("_$", "", incompleteSamples, perl = TRUE)
    completeSamples <- gsub("_$", "", names(completeFiles), perl = TRUE)
    sampleToSrrConversion <- read.csv(paste0('../', geoID, '_sampleToSrrConversion.csv'))
    outliers <- list()
    for(srr in incompleteSamples){
        outliers[[length(outliers) + 1]] <- sampleToSrrConversion[grep(srr, sampleToSrrConversion[,2]),1]
    }
    metaData <- readRDS(metaDataRds)
    # now we need to add in the samples that never even got to the directory stage
    for (gsm in metaData[,1]){
        srr <- as.character(sampleToSrrConversion[grep(gsm, sampleToSrrConversion[,1]),2])
        if (!is.na(srr) && !any(grepl(srr, completeSamples))){
            outliers[[length(outliers) + 1]] <- gsm
        }
    }
    outliers <- unique(as.character(unlist(outliers)))
}else if(idType == 'ae-seq'){
    outliers <- incompleteSamples
}


if(file.exists(metaDataRds)) {
    writeMetaDataWithOutliers(outliers, metaDataFile, idType)
}
txi.salmon <- tximport(completeFiles,
                       type = "salmon",
                       tx2gene = tx2gene,
                       ignoreTxVersion = TRUE,
                       dropInfReps = TRUE
                      )
saveRDS(txi.salmon, 'uniprot_expression_data.rds')
