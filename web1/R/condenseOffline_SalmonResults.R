library(tximport)

# from where this script is always called, this is the case and makes subsequent calls easier
microarrayDir <- '..' 
source('/home/ubuntu/2xar/twoxar-demo/web1/R/metaGEO_supportFunctions.R')
# rather than passing in arguments we'll just deduce them
temp <- strsplit(getwd(), '/')[[1]]
geoID <- 'RDEB'
idType <- 'offline'
#XXX this should be sourced from the settings written for this job
tx2gene <- read.table('/home/ubuntu/2xar/ws/HUMAN_9606_Protein_EnsemblTRS.tsv',
                       header = FALSE
                      )
dirs <- list.dirs(recursive = FALSE)
dirs <- dirs[ !grepl("rawFastq", dirs) ]
files <- paste(dirs, 'quant.sf', sep = "/")
names(files) <- gsub("^./", "", dirs, perl = TRUE)
incompleteSamples <- names(files)[!file.exists(files)]
completeFiles <- files[file.exists(files)]
incompleteSamples <- gsub("_$", "", incompleteSamples, perl = TRUE)
completeSamples <- gsub("_$", "", names(completeFiles), perl = TRUE)
outliers <- incompleteSamples
#writeMetaDataWithOutliers(outliers, idType, geoID)
txi.salmon <- tximport(completeFiles,
                       type = "salmon",
                       tx2gene = tx2gene,
                       ignoreTxVersion = TRUE,
                       dropInfReps = TRUE
                      )
saveRDS(txi.salmon, paste(geoID, idType, 'uniprot_expression_data.rds', sep = '_'))
