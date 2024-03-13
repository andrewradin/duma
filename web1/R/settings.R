#===========================================
# General output settings
#==========================================

debug <- FALSE

# directories used throughout
basedir <- '/tmp'
staticdir <- callPathHelper("publish")
storageDir <- callPathHelper("storage")
softDir <- '/tmp'
testDir <- '/home/ubuntu/2xar/twoxar-demo/web1/R/forSigGeoTesting'
scriptDir <- callPathHelper("Rscripts")

# mysql settings
mysql_user <- 'root'
mysql_host <- 'localhost'
metaTable <- 'web1.browse_sample'
database <- 'web1'
databaseTable <- 'browse_significantprotein'
geoTable <- 'web1.browse_significantprotein'

# just in case we ever decide to try to parallelize this
nproc <- 1

# Should we ignore missing data - default to not processing missing data
ignoreMissing <- FALSE

#===========================================
# Which algorithm to use to detect significant probes
#===========================================
# I don't actually think these do anything any more
CASE <- 1
CONT <- 2

# A setting for the log2 detection
top1PercentSignalThreshold <- 50

#===========================================
# Settings for differential probes callers
#==========================================
# minimum log2 intesnsity value for the probe to be considered
# this is not currently being used as it didn't seem to help, and if anything may have been making things worse
# it also may just need some fine tuning
minLogVal <- 0
minChdirScore <- 0

# maximum q-value to report
qmax <- 100.0

# Min fold change (to be log2'd)
minFC <- 1
#cv_thresh <- 0.02

# we set delta value to 0 and use q-value to filter , this is only for SAM
delta <- 0.0
# set random seed for reproducible results
seed <- 123

# Proportion of probes we need to successfully convert, to consider this a successful run
# This perhaps should be changes now that we are using the whole chip....we'll see
minProbeToUniprotProportion <- 1
maxMappingPortion <- 0
bestMappingYet <- NA
