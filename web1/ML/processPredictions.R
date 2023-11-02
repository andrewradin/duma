#=============================
# Get set up
#=============================
getScriptPath <- function(){
    cmd.args <- commandArgs()
    m <- regexpr("(?<=^--file=).+", cmd.args, perl=TRUE)
    script.dir <- dirname(regmatches(cmd.args, m))
    if(length(script.dir) == 0) stop("can't determine script dir: please call the script with Rscript")
    if(length(script.dir) > 1) stop("can't determine script dir: more than one '--file' argument detected")
    return(script.dir)
}


thisScriptsPath <- getScriptPath()
source(paste0(thisScriptsPath, '/../R/supportFunctionsForAllRCode.R'))

library(ROCR)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  warning("processPredictions.R <directory with predictions ending in .tsv> <mapping file> <rowNumsOftestTreatmentsDirectory> <directory containing the support functions for this script> <number of outer iterations>")
  quit('no', as.numeric(exitCodes['usageError']))
}
dataDir <- args[1]
mapper <- args[2]
rowNumsOftestTreatmentsDirectory <- args[3]
scriptDir <- args[4]
source(paste0(scriptDir,"/processPredictions_supportFunctions.R"))

setwd(dataDir)

# read in the mapper we have
mapperTable <- read.csv(mapper, header=F)

# get the directories in predictions, each one should be its own set
predictionSubDirs <- dir()[file.info(dir())$isdir]
# problems arise here if we use this for length
# If you run fewer iterations after having run more, the old subdirectory is picked up
# we just pass as a variable the number of outer iterations run for this classifier run
outerIterations <- as.numeric(args[5])

# Make some lists to hold the different types of results we are about to make. Each entry in each list will be one of the iterations
allItersPredictions <- roc <- auc <- precRecall <- list()
everything <- list(allItersPredictions = allItersPredictions, roc = roc, auc = auc,precRecall = precRecall)
forRankingsPlots <- list()

#===============================================================================================
# Run through each iteration and pull out the relevant information as well as quality metrics
#===============================================================================================

for (i in 1:outerIterations){
    iter <- predictionSubDirs[i]
    rowNumsOftestTreatments <- read.csv(paste0(rowNumsOftestTreatmentsDirectory,"/",iter,"/rowNumberOfTestTreatmentsInDF.txt"), header=F)
    setwd(iter)
    
    #=============================
    # Import all of the probabilities
    #=============================
    # Get a list of all of the files
    filenames <- list.files(pattern="*.tsv", full.names=F)
    # Set up a df to store all of probabilities (columns) that each drug (row) is related to the disease
    file <- read.csv(filenames[1], header=F, sep="\t")
    allModels <- data.frame(drug=file[,1], knownRelation=file[,2], dbID=mapperTable[,1])
    # Now go through all of the files and pull out the one value we care about
    for (j in 1:length(filenames)){
        allModels[,(ncol(allModels)+1)] <- as.numeric(read.csv(filenames[j], header=F, sep="\t")[,3])  
    }
    
    #=============================
    # Process the df
    #=============================
    # We'll take the mean of all of the models
    allModels$mean <- rowMeans(allModels[,c(-1,-2,-3)])
    allModels$testTreament <- FALSE
    allModels$testTreament[unlist(rowNumsOftestTreatments)] <- TRUE
    # also add these test treatments as known treatments
    allModels$knownRelation[unlist(rowNumsOftestTreatments)] <- "1:True"
      
    # Order them by the likelihood of being related to the disease
    orderedByProb <- allModels[rev(order(allModels$mean)),c("dbID", "knownRelation", "testTreament", "mean")]
    everything$allItersPredictions[[i]] <- orderedByProb
    #=============================
    # Evaluate the meta-model/predictions and show results
    #=============================
    everything <- evalModel(allModels$knownRelation, allModels$mean, everything)
    # For overlaying the known treatments, so that we can see verify we did a good job (the known treatments should be at the top)
    allTreatX<-which(orderedByProb$knownRelation=="1:True")
    allTreatY<-orderedByProb$mean[allTreatX]
    testTreatX<-which(orderedByProb$testTreament==TRUE)
    testTreatY<-orderedByProb$mean[testTreatX]
    # Move back to the main dir, to ensure things are saved in the same place
    setwd(dataDir)
    forRankingsPlots[[i]] <- list(orderedByProb=orderedByProb$mean, allTreatX=allTreatX, allTreatY=allTreatY, testTreatX=testTreatX, testTreatY=testTreatY)
 }

# Now combine the iterations into one cohesive model
combinedModel <- everything$allItersPredictions[[1]]
if( length( everything$allItersPredictions ) > 1 ){
    for (currentIteration in 2:length(everything$allItersPredictions)){
        combinedModel <- merge(combinedModel, everything$allItersPredictions[[currentIteration]][,c("dbID", "mean")], by="dbID")
    }
}

# get the indices of the predictions from each iterative model
meansInds <- grep("mean", colnames(combinedModel))
if(length(meansInds) != outerIterations){
    print(paste0("WARNING: ", outerIterations, " means expected, but ", length(meansInds), " found. Quitting."))
    quit('no', as.numeric(exitCodes['usageError']))
}
if(outerIterations > 1 ){
    # And finally average across them to a final prediction
    combinedModel$finalMean <- rowMeans(combinedModel[,meansInds])
}else{
    combinedModel$finalMean <- combinedModel[,meansInds]
}

#=============================
# Evaluate the final meta-model/predictions, and print out results/plots
#=============================
everything <- evalModel(combinedModel$knownRelation, combinedModel$finalMean, everything)
# and the final predictions
orderedByProb <- combinedModel[rev(order(combinedModel$finalMean)),]
# For overlaying the known treatments, so that we can see verify we did a good job (the known treatments should be at the top)
allTreatX<-which(orderedByProb$knownRelation=="1:True")
allTreatY<-orderedByProb$finalMean[allTreatX]

forRankingsPlots[[(length(forRankingsPlots)+1)]] <- list(orderedByProb=orderedByProb$finalMean, allTreatX=allTreatX, allTreatY=allTreatY, testTreatX=NA, testTreatY=NA)

plotRankings(forRankingsPlots)

#===============================================================
# Plot some QC metrics for each of the iterations
#===============================================================
# Each iteration gets it's own random color:
colors <- rgb(runif(outerIterations),runif(outerIterations),runif(outerIterations))
# the last, final one I want to be black
colors <- c(colors, 'black') 
lwds <- c(rep(1, outerIterations), 3) # and I want it to be bigger

plotAllOfTheMetrics(everything, colors, lwds)

#=============================
# Write out reports
#=============================
unknownRelation <- subset(combinedModel, combinedModel$knownRelation=="2:False")

# Now rank these drugs by probability and report all of them by their index and probability
ranked.i <- rev(order(unknownRelation$finalMean))

write.table(unknownRelation[ranked.i,c("dbID","finalMean")], file='rankedListOfPutativeRelations.csv', sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)

# Also report the known drugs in the same manner, for comparison
knownRelated <- subset(combinedModel, combinedModel$knownRelation=="1:True")
rankedKnown.i <- rev(order(knownRelated$finalMean))
write.table(knownRelated[rankedKnown.i,c("dbID","finalMean")], file='rankedListOfKnownRelations.csv', sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)

# And finally write out that information together for inclusion in the database
write.table(combinedModel[,c("dbID","finalMean")], file='allProbabilityOfAssoications.csv', sep=",", row.names=FALSE, col.names=FALSE, quote=FALSE)
