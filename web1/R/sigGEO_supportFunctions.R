#========================================================================================
# functions for sigGEO
#========================================================================================#
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
    source(paste0(thisScriptsPath, '/general_supportFunctions.R'))
    source(paste0(thisScriptsPath, '/sigGEO_rnaSeq.R'))
}

# This takes care of getting the assignments of cases and controls
parseDataColumns <- function(){
    caseFile <- paste0(indir, "/cases.tsv")
    controlFile <- paste0(indir, "/controls.tsv")
    if(! file.exists(caseFile) | ! file.exists(controlFile)){
        print(paste("Unable to find case or control column assignment files.",
                    "Make sure the cases and controls were assigned. Quitting"
                   ))
		print(paste('expected caseFile:',caseFile))
		print(paste('expected controlFile:',controlFile))
        quit('no', as.numeric(exitCodes['usageError']))
    }
    case.cols <- read.csv(caseFile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)[,1]
    control.cols <- read.csv(controlFile, header = FALSE, sep = "\t", stringsAsFactors = FALSE)[,1]
    if (length(control.cols) < 2 || length(case.cols) < 2){
        print("There were fewer than 2 cases or controls.",
              "We need at least that many to proceed.",
              "Quitting."
             )
        quit('no', as.numeric(exitCodes['unexpectedDataFormat']))
    }
    return(list(case = case.cols, cont = control.cols))
}

checkColAssigns <- function(columnAssignments, expressionData, idType){
    if(debug) print(columnAssignments)
    #=========================================================================================
    # Verify column names
    #=========================================================================================
    # I'm going to go through and ensure that all of the assigned samples are present
    # if they're not I'll try to find a match
    allMatched <- checkColumns(columnAssignments, colnames( expressionData))
    if(! allMatched ){
        colnames(expressionData) <- findColumnAssignments(originalColNames, columnAssignments)
        # now check again
        allMatched <- checkColumns(columnAssignments, colnames( expressionData), TRUE)
        if(! allMatched & idType != 'ae'){
            colnames(expressionData) <- just_gsms(colnames(expressionData))
        }
        allMatched <- checkColumns(columnAssignments, colnames( expressionData), TRUE)
    }
    # now check again
    if(! allMatched){
        print(paste("Unable to match sample names and relevant columns in the expression data",
                    "for all samples."
                   ))
        print(paste("This means you will be unable to use this data.",
                    "It is recommended that you try toggling to fallback."
                   ))
        print(paste0("For debugging purposes, the expression data column names are: "
                     , paste(colnames( expressionData), collapse = ", ")
                     , " and the samples names are: "
                     , paste(unlist(columnAssignments), collapse = ", ")
                     )
             )
        quit('no', as.numeric(exitCodes['expressionHeaderError']))
    }
    return(list(ca = columnAssignments, ed = expressionData))
}

# just a glorified for loop to check the column headers
checkColumns <- function(columnAssignments, colHeader, printMessages = FALSE){
    for( colAssignName in unlist(columnAssignments) ){
        if(printMessages){ print(paste("checking", colAssignName)) }
        if( ! any( grepl( colAssignName, colHeader, fixed = TRUE ) ) ){
            if(printMessages){ print(paste("Unable to find a match for", colAssignName)) }
            return(FALSE)
        }
    }
    return(TRUE)
}


# this is called if we've checked and not all of the column assignments are present
findColumnAssignments <- function(originalColNames, columnAssignments){
    # I took all quit statements out of here, and instead handle them in sigGEO.R
    #
    # The most likely reason this is true is b/c we're working with Illumina data
    # and they have poorly name columns
    # I'm going to bring back the original column names and try to match those
    newColNames <- originalColNames
    # now go through each one and try to find a match in the metaData
    print(paste("We were unable to find all of the expression data columns.",
               "Trying to fix that.  Reading in meta data."
              ))
    metaData <- read.table(paste0(microarrayDir, '/', fullGeoID, "_metadata.tsv"),
                           sep = "\t",
                           header = FALSE,
                           stringsAsFactors = FALSE,
                           quote=''
                           )
    # to speed things up, I'm going to go through and remove the columns
    # in metaData that are not unique
    # I know I don't want to search the first one b/c it's the one with the GSM (or equiv)
    colsToIgnore <- c(1)
    for (columnInd in 2:ncol(metaData) ){
        if(length(unique(metaData[,columnInd])) != nrow(metaData)){
            colsToIgnore <- c(colsToIgnore, columnInd)
        }
    }
    # now search
    for( colAssignName in unlist(columnAssignments) ){
        if( ! any( grepl( colAssignName, originalColNames, fixed = TRUE ) ) ){
            print(paste("Did not find column with ", colAssignName))
            # find the relevant line for this sample
            metaDataInd <- grep(colAssignName, metaData[,1], fixed = TRUE)
            # make sure we only found one line
            if( ! length( metaDataInd ) == 1 ){
                print(paste0("In trying to find the correct expression data column for ",
                             colAssignName,
                             ", unable to find 1 unique match (row) in the meta data."
                             ))
                print(paste0("Instead found ",
                             length( metaDataInd ),
                             ". This means the column assignemnt is not unique. ",
                             "Recommend trying to toggle to fallback."
                             ))
            }
            # I already removed the non-unique lines
            metaDataLine <- metaData[metaDataInd, -colsToIgnore]
            # now check and see if we can find anything relevant in this line to assign the column
            for( field in metaDataLine ){
                didWeFindAMatch <- FALSE
                # the column names have had the spaces sub'd out for some reason
                toGrepWith <- gsub(" ", ".", field)
                # this has worked in the past, so I 'll try it
                toGrepWith <- gsub("(", ".", toGrepWith, fixed = TRUE)
                toGrepWith <- gsub(")", "", toGrepWith, fixed = TRUE)
                toGrepWith <- gsub("-", ".", toGrepWith, fixed = TRUE)
                possibleMatch <- grep(toGrepWith, originalColNames)
                if( length( possibleMatch ) == 1 ){
                    print(paste0("Replacing ",
                                 originalColNames[possibleMatch],
                                 " with ",
                                 colAssignName
                                ))
                    newColNames[possibleMatch] <- colAssignName
                    didWeFindAMatch <- TRUE
                    break
# I'm originally writing this for just GSE118553,
# but if we end up getting here we might as well try anything.
# In this instance the column header was actually buried in paranatheses
# So we're stripping all that away.
                }else{
                    possibleMatch <- grep(unlist(strsplit(unlist(strsplit(field, split="(", fixed=T))[2], split=")", fixed=T))[1], originalColNames)
                    if( length( possibleMatch ) == 1 ){
                        print(paste0("After stripping parantheses, replacing ",
                                 originalColNames[possibleMatch],
                                 " with ",
                                 colAssignName
                                ))
                        newColNames[possibleMatch] <- colAssignName
                        didWeFindAMatch <- TRUE
                        break
                    }
                }
            }
            if(! didWeFindAMatch){
                print(paste("We were unable to find meta data to help find the correct",
                             "column in the expression data for", 
                             colAssignName
                            ))
                print(paste("We can't proceed with this data because of that.",
                            "Recommend trying to toggle to fallback."
                           ))
            }
        }
    }
    return(newColNames)
}

cleanupIlluminaHeaders <- function(expressionData){
    # This probably stems from the fixes I made to column names in metaGEO,
    # but in some cases there still column names that include the AVG_SIGNAL,
    # even though the descriptions below do not include that.
    #  It's a pretty easy fix for that.
    colnames(expressionData) <- gsub("_AVG_SIGNAL",
                                     "",
                                     colnames(expressionData),
                                     ignore.case = TRUE
                                     )
    # this is messy, but just in case there wasn't an _ connecting, but .
    colnames(expressionData) <- gsub(".AVG_SIGNAL",
                                     "",
                                     colnames(expressionData),
                                     ignore.case = TRUE
                                    )
    descriptions <- readRDS(paste0(microarrayDir, "/illuminaGsmConverter.rds"))
    for (i in 1:length(descriptions)){
        if (debug){
            print(paste('Changing',
                        descriptions[i],
                        'for',
                        names(descriptions[i])
                      ))
        }
        colnames(expressionData)[which(colnames(expressionData) == descriptions[i])] <- names(descriptions[i])
    }
    return(expressionData)
}

cleanUpSigProbeDFs <- function(orig, nameToAdd, x, case_cols, cont_cols,
                              calcScore = FALSE, min_q = 0.05, geneColName = "Gene ID"){
    select <- orig[which(orig[,"q-value"] <= min_q),c(geneColName,
                                                      "Fold Change",
                                                      "q-value",
                                                      "direction"
                                                     )]
    if (sum(orig[,"q-value"] <= min_q) > 1){
        names_to_use <- paste(nameToAdd, colnames(select))
        colnames(select) <- names_to_use
        select[,1] <- as.character(select[,1])
        for(i in 2:4){select[,i] <- asNum(select[,i])}
        if(debug){print(head(select))}
    # use this opportunity to generate some plots as well
        contLen <- length(cont_cols)
        caseLen <- length(case_cols)
        colForPlots <- c(rep("black", contLen)
                       ,rep('red', caseLen)
                       )
        groups <- c(rep("control", contLen)
                   , rep("case", caseLen)
                   )
        toPlot <- cbind(rownames(x), x)
        toPlot <- merge(toPlot, select[,1], by = 1)[,-1]
        for(i in 1:ncol(toPlot)){
            toPlot[,i] <- asNum(toPlot[,i])
            colnames(toPlot)[i] <- paste(groups[i], colnames(toPlot)[i], sep = "_")
        }
        if (nrow(toPlot) > 1 ){
            plotMDS(toPlot
                    , col = colForPlots
                    , labels = c(cont_cols, case_cols)
                    , main = paste(nameToAdd, 'MDS with sig probes')
                    , xlab = "Leading dimension"
                    , ylab = "Secondary dimension"
                   )
            pca <- prcomp(t(toPlot), center = TRUE, scale. = TRUE)
            plot(pca$x[, 1]
                 , pca$x[, 2]
                 , main = paste(nameToAdd, 'PCA with sig probes')
                 , xlab = "Principal Component 1"
                 , ylab = "Principal Component 2"
                 , type = 'n'
                )
            text(pca$x[, 1]
                 , pca$x[, 2]
                 , c(cont_cols, case_cols)
                 , col = colForPlots
                )
        }
        ccCorScore <- calcCcCorScore(toPlot, contLen, caseLen)
    }else{
        ccCorScore <- list(control = 0, case = 0)
    }
    if (calcScore){
        return(list(select = select, ccCorScore = ccCorScore))
    }
    return(select)
}

calcCcCorScore <- function(x, nContCols, nCaseCols){
    cors <- makeCorMat(x)
    if(nrow(x) > 1){
        if (nrow(x) > 45000){
            print(paste("More than 45000 rows given to pheatmap.",
                        "Using variance to select the most informative",
                        "45000 rows."
                       )
                  )
            var <- rowSums((x - rowMeans(x))^2)/(dim(x)[2] - 1)
            # rather than sorting we'll just take the top 45000, using quantile
            thresh <- quantile(var, 1-44990/nrow(x)) # 44,900 just to give us some rounding room
            x <- x[var>thresh,]
        }
         pheatmap(x)
    }
# report average correlation b/t controls and avg correlation b/t cases
    contScore <- list()
    for (i in 1:nContCols){
        for (j in 2:nContCols){
            if (j > i){
                contScore[[length(contScore) + 1]] <- cors[i,j]
            }
        }
    }
    caseScore <- list()
    for (i in (nContCols+1):(nContCols + nCaseCols)){
        for (j in (nContCols+2):(nContCols + nCaseCols)){
            if (j > i){
                caseScore[[length(caseScore) + 1]] <- cors[i,j]
            }
        }
    }
    return(list(contol = mean(unlist(contScore)), case = mean(unlist(caseScore))))
}

compareDEGs <- function(significantProbesSam, significantProbesLimma, significantProbesRP, x, control.cols, case.cols){
    concord <- combineDEGs(significantProbesSam, significantProbesLimma, significantProbesRP)
    pdf(paste0(pubdir, "/", geoID, "_", tisID, "_sigPCAMDS.pdf"))
      templist <- cleanUpSigProbeDFs(concord, 'Concord', x, case.cols, control.cols, calcScore = TRUE)
      conc <- templist$select
      sam <- cleanUpSigProbeDFs(significantProbesSam, 'SAM', x, case.cols, control.cols)
      lim <- cleanUpSigProbeDFs(significantProbesLimma, "Limma", x, case.cols, control.cols)
      rp <- cleanUpSigProbeDFs(significantProbesRP, "RankProd", x, case.cols, control.cols)
    dev.off()
    combined <- merge(sam, lim, all = TRUE, by = 1)
    combined <- merge(combined, rp, all = TRUE, by = 1)
    combined <- merge(combined, conc, all = TRUE, by = 1)
    intersect_cnt <- intersect_por <- intersect_cor <- matrix(0.0, nrow = 4, ncol = 4)
    totals <- vector(mode = 'numeric', length = ncol(intersect_cnt))
    names(totals) <- colnames(intersect_cnt) <- colnames(intersect_por) <- colnames(intersect_cor) <- rownames(intersect_cnt) <- rownames(intersect_por) <- rownames(intersect_cor) <- c('SAMR', 'Limma', 'RankProd', 'Concord')
    q_inds <- c(3, 6, 9, 12)
    dir_inds <- c(4, 7, 10, 11)
    for (i in 1:ncol(intersect_cnt)){
        inI <- combined[which(! is.na(combined[,q_inds[i]])), ]
        totals[i] <- nrow(inI)
        for (j in 1:ncol(intersect_cnt)){
            intrsct  <- inI[which(! is.na(inI[,q_inds[j]])),]
            intersect_cnt[i,j] <- nrow(intrsct)
            intersect_por[i,j] <- intersect_cnt[i,j] / totals[i]
            intersect_cor[i,j] <- cor( -1 * log10(asNum(intrsct[,q_inds[i]])) * asNum(intrsct[,dir_inds[i]])
                                , -1 * log10(asNum(intrsct[,q_inds[j]])) * asNum(intrsct[,dir_inds[j]])
                                , method = 'spearman'
                                )
        }
    }
    intersect_por[is.na(intersect_por)] <- 0
    intersect_cor[is.na(intersect_cor)] <- 0
    if(debug){
        print(intersect_cnt)
        print(intersect_por)
        print(intersect_cor)
        print(totals)
    }
    pdf(paste0(pubdir, "/", geoID, "_", tisID, "_concordance.pdf"))
      breakPoints <- seq(0, nrow(combined), length.out = 200)
      pheatmap(intersect_cnt
               , display_numbers = TRUE
               , main = "# of intersecting probes"
               , number_format =  "%.0f"
               , breaks = breakPoints
               , color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdYlBu")))(length(breakPoints))
               )
      breakPoints <- seq(0, 1, length.out = 200)
      pheatmap(intersect_por
               , display_numbers = TRUE
               , main = "Portion of intersecting probes"
               , breaks = breakPoints
               , color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdYlBu")))(length(breakPoints))
               )
      breakPoints <- seq(-1, 1, length.out = 200)
      pheatmap(intersect_cor
               , display_numbers = TRUE
                , main = "Spearman rho of intersecting probes"
               , breaks = breakPoints
               , color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdYlBu")))(length(breakPoints))
               )
      plot_venn(combined, q_inds, totals, intersect_cnt)
      plot_portion_sig(conc, sam, lim, rp, nrow(x))
    dev.off()
    return(list(significantProbes = concord, callingScoresList = list(ccCorScore = templist$ccCorScore, concordScore = calcConcordScore(intersect_por))))
}

calcConcordScore <- function(intersect_por){
    # average portion overlap for all 3 algos
    return(mean(c(intersect_por[1,2:3], intersect_por[2,c(1,3)], intersect_por[3, 1:2])))
}

plot_portion_sig <- function(avg_q, sam, lim, rp, total_num){
    barplot(c(nrow(avg_q) / total_num
              , nrow(sam) / total_num
              , nrow(lim) / total_num
              , nrow(rp) / total_num
             ),
            names.arg = c('Concord', 'SamR', 'Limma', 'RankProd')
            , ylab = "Portion of all probes called significant"
            , main = 'Significant probe portions by program'
           )
}

plot_venn <- function(combined, q_inds, totals, intersect_cnt){
    library(VennDiagram)
    n123 <- length(which(! is.na(combined[,q_inds[1]]) & ! is.na(combined[,q_inds[2]]) & ! is.na(combined[,q_inds[3]])))
    n124 <- length(which(! is.na(combined[,q_inds[1]]) & ! is.na(combined[,q_inds[2]]) & ! is.na(combined[,q_inds[4]])))
    n134 <- length(which(! is.na(combined[,q_inds[1]]) & ! is.na(combined[,q_inds[3]]) & ! is.na(combined[,q_inds[4]])))
    n234 <- length(which(! is.na(combined[,q_inds[2]]) & ! is.na(combined[,q_inds[3]]) & ! is.na(combined[,q_inds[4]])))
    n1234 <- length(which(! is.na(combined[,q_inds[1]]) & ! is.na(combined[,q_inds[2]]) & ! is.na(combined[,q_inds[3]]) & ! is.na(combined[,q_inds[4]])))
    grid.newpage()
    draw.quad.venn(totals[1], totals[2], totals[3], totals[4],
                   intersect_cnt[1,2], intersect_cnt[1,3], intersect_cnt[1,4],
                   intersect_cnt[2,3], intersect_cnt[2,4],
                   intersect_cnt[3,4],
                   n123, n124, n134,
                   n234,
                   n1234,
                   names(totals),
                   fill = c("skyblue", "pink1", "mediumorchid", 'green')
                  )
}

combineDEGs <- function(significantProbesSam, significantProbesLim, significantProbesRP){
    sam <- significantProbesSam[,c("Gene ID","Fold Change", "q-value","direction")]
    lim <- significantProbesLim[,c("Gene ID","Fold Change", "q-value","direction")]
    rp <- significantProbesRP[,c("Gene ID","Fold Change", "q-value","direction")]
    combined <- merge(sam, lim, by = 1, all = T)
    combined <- merge(combined, rp, by = 1, all = T)
    dir_inds <- c(4,7,10)
    fc_inds <- c(2,5,8)
    q_inds <- c(3,6,9)
    for(i in c(dir_inds, fc_inds)){combined[is.na(combined[,i]),i] <- 0}
    for(i in q_inds){combined[is.na(combined[,i]),i] <- 1} # replace the NAs with 1s for q
    if(debug){print(head(combined))}
    final <- data.frame("Gene ID" = combined[,"Gene ID"],
                        "Fold Change" = rep(0, nrow(combined)),
                        "q-value" = rep(1, nrow(combined)),
                        "direction" = rep(0, nrow(combined))
                       )
    for (i in 1:nrow(combined)){
        dirs <- sum(combined[i,dir_inds])
        if(abs(dirs) < 2){ # this is equivalent to all of them agreeing or 2 agreeing and one not having a direction
            if(debug){
                print("Direction was inconsistent. Setting FC and direction to 0, and q to 1")
                print(combined[i,])
            }
        }else{
            if (debug && length(which(combined[i,fc_inds] == 0)) == 0 && (round(combined[i,2],2) != round(combined[i,5], 2) || round(combined[i,2], 2) != round(combined[i,8], 2))){
                fcs <- paste(combined[i,fc_inds], collapase = ", ")
                print(paste("FC wasn't consistent for", combined[i,1], ". Values were:", fcs))
            }
            final[i, 2] <- median(asNum(combined[i,fc_inds]))
            final[i, 3] <- median(asNum(combined[i,q_inds]))
            final[i, 4] <- sign(dirs)
        }
    }
    colnames(final) <- c("Gene ID","Fold Change", "q-value","direction")
    return(final)
}

runGeoDE <- function(x, control.cols, case.cols){
    source("geodeCode.R")
    print("############################################################################")
    print("Running our copy of GeoDE")
    print("############################################################################")
    #=========================================================================================
    # Prep for and run GeoDE
    #=========================================================================================
    xForGeoDe <- as.data.frame(cbind(row.names(x), x))
    rowsWithNonFiniteValues <- list()
    for(row in 1:nrow(x)){
        if(any( ! is.finite(x[row,]) )){
             rowsWithNonFiniteValues[[length(rowsWithNonFiniteValues)+1]] <- sum( which(! is.finite(x[row,]) ))
        }
    }

    if(debug){
        print(paste("There are", length(rowsWithNonFiniteValues), "probes with at least one non-finite value, constituting", sum(unlist(rowsWithNonFiniteValues)), "cells in the matrix"))
        print(paste("There are a total of ", nrow(xForGeoDe), "rows in the matrix and ", nrow(xForGeoDe)*(ncol(xForGeoDe)-1), "cells."))
    }
    for(c in 2:ncol(xForGeoDe)){xForGeoDe[,c] <- asNum(xForGeoDe[,c])}
    xForGeoDe[,1] <- as.character(xForGeoDe[,1])
    colnames(xForGeoDe) <- c("probenames", colnames(x)) 
    # create map of case and control
    y <- as.factor(c(rep(1, length(control.cols)), rep(2, length(case.cols))))
    pdf(paste0(pubdir, "/", geoID, "_", tisID, "_geodePlot.pdf"))
         chdir_analysis <- chdirAnalysis(xForGeoDe, y, CalculateSig = TRUE) # I'm trying not to set parameters if I don't have to
    dev.off()
    if(debug){print(paste0("Number of all probes ", nrow(xForGeoDe)))}
    if (F){
        chrdirs <- chdir_analysis$results[[1]] # if we ever try to parameterize the gammas, we can't just take the first one (right now we only use one gamma, so we just take the one)
        pvals <- chdir_analysis$pvals[[1]]
        if (length(chrdirs) != length(pvals)){
            print("WARNING: Something went wrong with GeoDE, differing numbers of p-values and Characteristic directions. Quitting.")
            quit('no', as.numeric(exitCodes['unexpectedDataFormat']))
        }
        probes.allFC <- cbind(names(chrdirs), abs(chrdirs), pvals,  rep(1, length(chrdirs)))
        probes.allFC[which(chrdirs < 0), 4] <- -1
        for(i in 2:4){probes.allFC[,i] <- asNum(probes.allFC[,i])   }
    }else{
        sigifs <- chdir_analysis$results[[1]]
        if(debug){
            print(paste0("Number of signif probes with chrdir greater than 0.01 ", length(sigifs[which(abs(sigifs) > 0.01)])))
            print(paste0("Number of signif probes with chrdir greater than 0.001 ", length(sigifs[which(abs(sigifs) > 0.001)])))
            print(paste0("Number of signif probes with chrdir greater than 0.0001 ", length(sigifs[which(abs(sigifs) > 0.0001)])))
            print(paste0("Number of all probes ", nrow(xForGeoDe)))
            print(summary(abs(sigifs)))
        }
        # I would need to format the results as necessary for further use.
        # Likely set the q value scale to the maximal chrdir
        # Direction is from chrDir value
        # maybe swap out FC for the chrdir value.??? still need to work on that
        # If I did that I would need to make sure the log2FC didn't filter everything
    #    probes.signif <- cbind(names(sigifs), abs(sigifs), rep(0.01, length(sigifs)),  rep(1, length(sigifs)))
    #    probes.signif <- cbind(names(sigifs), abs(sigifs), (max_sig - abs(sigifs))/10,  rep(1, length(sigifs)))
        probes.signif <- cbind(names(sigifs), abs(sigifs), abs(sigifs),  rep(1, length(sigifs)))
        probes.signif[which(sigifs < 0), 4] <- -1
        # now make the nonSignif
        nonSignifProbes <- cbind(xForGeoDe[,1], rep(0, nrow(xForGeoDe)), rep(1, nrow(xForGeoDe)),  rep(0, nrow(xForGeoDe)))
        tempDF <- merge(probes.signif, nonSignifProbes, by = 1, all.y = TRUE) # this ends up keeping all of the probes from above, but puts NAs in 
    
        probes.allFC <- tempDF[,c(1:4)]
    
        for(i in 2:4){probes.allFC[,i] <- asNum(probes.allFC[,i])}
        
        # I have to take care of 2 things, the NAs that are present as a result of the merge
        # and the chdir values that are too small to be consistent
        # I've gotten anywhere from 98% of the chip to ~15% of the chip ebing significant at any cutoff
        # but if you require a minimum chdir value, then things are consistent
        probes.allFC[which(is.na(probes.allFC[,2])),2] <- probes.allFC[which(probes.allFC[,2] < minChdirScore),2] <- 0
        probes.allFC[which(is.na(probes.allFC[,3])),3] <- probes.allFC[which(probes.allFC[,2] < minChdirScore),3] <- 1
        probes.allFC[which(is.na(probes.allFC[,4])),4] <- probes.allFC[which(probes.allFC[,2] < minChdirScore),4] <- 0
        colnames(probes.allFC) <- c("Gene ID","Fold Change", "q-value","direction")
    
        # we want to have a real FC, and rather than doing anything fancy, I'm just going to run limma and pull it's FC. Kind of over kill, but it runs quickly
        limmaRes <- runLimma(x, control.cols, case.cols)
        relevantLimma <- limmaRes[,c(1,2)] # gene ID, FC
        probes.allFC <- add_FC_to_geoDE(relevantLimma, probes.allFC)
    }
    colnames(probes.allFC) <- c("Gene ID","Fold Change", "q-value","direction")
    print(paste0("Number of probes with p less than 0.05: ", length(which(probes.allFC[,"q-value"] <= 0.05))))
    print(paste0("And there are ", nrow(probes.allFC), " total probes. That is ", length(which(probes.allFC[,"q-value"] <= 0.05)) / nrow(probes.allFC)))
    return(probes.allFC)

}

thoroughLog2ing <- function(x){
    if(debug){print("Before Log2'ing summaries")}
    if(debug){for(colInd in 1:ncol(x)){ print(summary(x[,colInd]))}}
    # I ran into some microarrays with negative  0 values, but that still needed to be logged, meaning they have really large values
    # This isn't ideal, but I'm going to set all of those negative values to 0 as any negative signal basically means that we couldn't
    # detect it at any intensity, which 0 says the same thing, and won't crash the program
    # this is a little ad hoc at this point, but I want to add a small value to avoid taking the log2 of 0, or a negative number
    avoidZero <- 2^(minLogVal/10)
    if(min(x) <= 0){
        print(paste("Found values less than 0 in a matrix that also needs to be log2 transformed.",
              "Shifting everything so all values are positive, and performing log2 transformation.",
              "There are",
              sum(colSums(x <= 0)),
             "cells with zero or negative values.",
             "That includes",
              length(which(rowSums(x <= 0) > 0)),
             "rows, and",
              length(which(colSums(x <= 0) > 0)),
             "columns that have at least one non-positive number."
             ))
        avoidZero <- avoidZero + abs(min(x))
    }else{
        avoidZero <- 0
    }
    x <- log2(x + avoidZero)
    print("log2 of expression data taken. Ensuring that went smoothly")
    return(checkForNonNumericAndImputeIfNecessary(x))
}

add_FC_to_geoDE <- function(other_program_results, probes.allFC){
    colnames(probes.allFC)[1] <- "Gene ID" # only for RNA-seq, just to make this easy. The names are changed back later anyway
    colnames(other_program_results) <- c("Gene ID", "other_program_results FC")
    combined <- merge(probes.allFC, other_program_results, by = 'Gene ID')
#
# I didn't end up checking direction agreement because genes with insignificant changes have a direction of 0.
# That means if the programs get different sig probes (which they will), the directions won't all match
#
    if( nrow(combined) != nrow(probes.allFC)){
        print(paste("WARNING: Before merging probes.allFC had", nrow(probes.allFC), "rows. Now it has", nrow(combined), ". other_program_results had", nrow(other_program_results)))
        quit('no', as.numeric(exitCodes['unexpectedDataFormat']))
    }
    # these are renamed anyways
    selected <- data.frame(first = combined$"Gene ID", second = combined$"other_program_results FC", third = combined$"q-value", fourth = combined$"direction")
    return(selected)
}

# we were running into an issue b/c the GeoDE scores were far less than 1-q
# e.g. The characterisitc direction normally ranges from 0-0.08, while 1-q is around 0.95
# this hopefully deals with that.
# it doesn't get the chrdir all the way up to 0.95 though, so I'll also try to bring the RNA-seq values down a little
scale_chrDir <- function(x, scaling_factor = 0.1, cap = NA){ 
# 0.1 was selected for the scaling_factor b/c generally a little higher than the highest chrDir we see,
# and it makes this transformation basically linear until 0.2 or so
# The cap is the maximal value that the scaled RNA-seq values can get (input is : 1-q = 1),
# that way neither approach can outweigh the other
# specifically the cap is 1/(1+exp(-1))
    if(is.na(cap))cap <- 1/(1+exp(-1))
    y <- 1/(1+exp(-abs(x)*(1/scaling_factor)))
    y[which(y > cap)] <- cap
    y <- y + 1 - 1/(1+exp(-1))
    # now we need to return the negatives
    return(y*sign(x))
}

# Prep for and run rankProd
runRankProd <- function(x, control.cols, case.cols){
    source(paste0(thisScriptsPath, '/rankprod.R'))
    print("############################################################################")
    print("Running RankProd")
    print("############################################################################")

    #=========================================================================================
    # Prep for and run rankProd
    #=========================================================================================
    # How many permutations to do for rankProd
    if(testTheScript)permNum <- 1

#    x <- filter_by_cv(x)
    start <- proc.time()
    png(paste0(pubdir,"/", geoID, "_", tisID, '_rankProdPlot.png'))
        RP.out <- RP(x, control.cols, case.cols, num.perm = permNum, plot = TRUE, rand = seed)
    dev.off()
    print(paste("rankprod took:", proc.time() - start))
    siggenes.table <- topGene(RP.out, cutoff = qmax/100, method = "pfp", logbase = 2, gene.names = row.names(x))
    # pull out significantly expressed genes
    probes.up <- as.data.frame(siggenes.table$Table1)
    probes.up$geneName <- row.names(probes.up)
    probes.dn <- as.data.frame(siggenes.table$Table2)
    probes.dn$geneName <- row.names(probes.dn)

    # Just to make sure there's something there
    if(debug){
        print("Up info")
        print(head(probes.up,1))
        print(tail(probes.up,1))
        print("Down info")
        print(head(probes.dn,1))
        print(tail(probes.dn,1))
    }
    # Make the final list
    probes.allFC <- rbind(probes.up, probes.dn)[,c(1,6,2:5)]
    colnames(probes.allFC) <- c("Row", "GeneID", "RPRsum","FoldChange","q", "p")
    probes.allFC[,"FoldChange"] <- log2(probes.allFC[,"FoldChange"])
    # rankProd is dumb and reports all the probes for both directions, so we have to include a last step which picks the direction with the best q
    toRtrn <- data.frame(probes.allFC %>% group_by(GeneID) %>% filter(q == min(q)) %>% arrange(Row, GeneID, RPRsum, FoldChange, q, p))
    if(debug){print(head(toRtrn))}
    colnames(toRtrn) <- c("Row", "Gene ID", "RP/Rsum","Fold Change","q-value", "p-value")
    toRtrn$direction <- sign(toRtrn[,"Fold Change"]) * -1 # rank prod seems to have directions that are opposite of limma and samR. I'm sure it's b/c I switched something, but this is easy
    toRtrn[,"Fold Change"] <- abs(toRtrn[,"Fold Change"])
    return(toRtrn)
}

#filter_by_cv <- function(x){
#    cvs <- rowCv(x)
#    print(paste("Filtering", sum(cvs < cv_thresh), 'probes (of', nrow(x), 'total probes) for being below the CV threshold of', cv_thresh))
#    return(x[cvs >= cv_thresh,])
#}
# Prep for and run samR
runSamR <- function(x, control.cols, case.cols){
    library(samr)
    print("############################################################################")
    print("Running SamR")
    print("############################################################################")

    #=========================================================================================
    # Prep for and run SAMR
    #=========================================================================================
    # create map of case and control for SAM
    y <- c(rep(1, length(control.cols)), rep(2, length(case.cols)))
    # create data structure for SAM
#    x <- filter_by_cv(x)
    d <- list(x = x, y = y, geneid = as.character(rownames(x)), genenames = as.character(rownames(x)), logged2 = TRUE)

    # How many permutations to do for samR
    if(testTheScript) permNum <- 1

    # run SAM
    samr.obj <- samr(d, resp.type = "Two class unpaired", assay.type = "array", nperms = permNum, random.seed = seed)
    delta.table <- samr.compute.delta.table(samr.obj)
    siggenes.table <- samr.compute.siggenes.table(samr.obj, delta, d, delta.table)

    # pull out significantly expressed genes
    probes.up <- as.data.frame(siggenes.table$genes.up)
    probes.up$direction <- rep(1,nrow(probes.up))
    probes.dn <- as.data.frame(siggenes.table$genes.lo)
    probes.dn$direction <- rep(-1,nrow(probes.dn))

    # Just to make sure there's something there
    if(debug){
        print("Up info")
        print(head(probes.up,5))
        print(tail(probes.up,5))
        print("Down info")
        print(head(probes.dn,5))
        print(tail(probes.dn,5))
    }
    # we're now taking all probes and filtering after condensing to uniprot IDs
    probes.allFC <- rbind(probes.up, probes.dn)

    # Make the final list
    colnames(probes.allFC) <- c("Row", "Gene ID", "Gene Name","Score(d)","Numerator(r)","Denominator(s+s0)", "Fold Change","q-value","direction")
    probes.allFC[,"q-value"] <- asNum(probes.allFC[,"q-value"])/100 # enough of this weird percentage q-value
    probes.allFC[,"Fold Change"] <- abs(log2(asNum(probes.allFC[,"Fold Change"])))
    # plot the qqplot
    png(paste0(pubdir,"/",geoID,"_",tisID,"_qqplot.png"))
        samr.plot(samr.obj, 1)
    dev.off()
    
    #return(probes.FCFiltered)
    return(probes.allFC)
}

runLimma <- function(x, control.cols, case.cols){
    print("############################################################################")
    print("Running Limma")
    print("############################################################################")

    design <- makeDesignMat(x, control.cols, case.cols)
    fit <- lmFit(x, design)
    cont.matrix <- makeContrasts(signature = case - control, levels = design)
    fit2 <- contrasts.fit(fit, cont.matrix)
    ebFit <- eBayes(fit2)
    results <- topTable(ebFit, coef = 1, adjust.method = "none", p.value = (qmax/100), lfc = log2(minFC), number = nrow(x))

    if(nrow(results) > 0){
        direction <- sign(results$logFC)
        probes.allFC <- cbind(rownames(results), results, direction)
        colnames(probes.allFC) <- c("Gene ID", "Fold Change", "AveExpr", "t", "P.Value", "q-value", "B", "direction")
        probes.allFC[,"Fold Change"] <- abs(probes.allFC[,"Fold Change"])
    }else{
        probes.allFC <- results
    }
    print(paste0("Number of probes with p less than 0.05: ", length(which(probes.allFC[,"q-value"] <= 0.05))))
    print(paste0("And there are "
                 , nrow(probes.allFC)
                 , " total probes. That is "
                 , length(which(probes.allFC[,"q-value"] <= 0.05)) / nrow(probes.allFC)
                 )
          )
    return(probes.allFC)
}

# set up the matricies needed for SVA or Limma
makeDesignMat <- function(x, control.cols, case.cols){
    # create map of case and control for Limma
    design <- model.matrix(~ 0 + factor(c(rep(1, length(control.cols)), rep(2, length(case.cols)))))
    colnames(design) = c("control", "case")
    return(design)
}

# Take care of building the matrix of expression values (rows) by samples (cols)
buildX <- function(expData, case.cols, control.cols){
    needsLogging <- read.table(paste0(microarrayDir, "/needsLog2.txt"), colClasses = "logical", header = TRUE)[1,1]
    # Build an array with all of the data
    if (! check_cols(case.cols,control.cols,colnames(expData))){
        colnames(expData) <- just_gsms(colnames(expData))
        check_cols(case.cols, control.cols, colnames(expData))
    }
    x <- cbind(expData[,control.cols], expData[,case.cols])
    for (j in 1:ncol(x)){ x[,j] <- asNum(x[,j])}
    x <- as.matrix(x)
    x <- checkForMissingData(x)
    # check if the data needs to be log2
    if(needsLogging) x <- thoroughLog2ing(x)
    # Data visualization
    if(runSVA) x <- cleanMatViaSVA(x, control.cols, case.cols)
    rownames(x) <- rownames(expData)
    colnames(x) <- c(control.cols, case.cols)
    if (debug) print(summary(x))
    plotDiagnostics(x, control.cols, case.cols)
    return(x)
}

cleanMatViaSVA <- function(x, control.cols, case.cols){
    library(sva)    
    design <- makeDesignMat(x, control.cols, case.cols)
    null <- model.matrix(~1
                         ,data = factor(c(rep(1, length(control.cols))
                                          ,rep(2, length(case.cols))
                                          )
                                       )
                         ) # exactly what was used to build the design matrix
    # this will identify the surragate variables
    svaObj <- sva(x, design, null)
    # combine the surragate variable with the already known covariates
    x <- cleanMat(x, design, svaObj$sv)
    plotDiagnostics(x, control.cols, case.cols)
    return(x)
}

# downloaded and modified from here https://www.biostars.org/p/121489/
cleanMat <- function(y, mod, svs) {
    X = cbind(mod, svs)
    Hat = solve(t(X) %*% X) %*% t(X)
    beta = (Hat %*% t(y))
    rm(Hat)
    gc()
    P = ncol(mod)
    return(y - t(as.matrix(X[,-c(1:P)]) %*% beta[-c(1:P),]))
}

# ensure that there are no NAs or Infs in the matrix. If there are impute them.
checkForNonNumericAndImputeIfNecessary <- function(x){
    if(any(! is.numeric(x) | is.na(x))){
        print("Found non-numeric values in matrix. Imputing those now.")
        x[which(!is.numeric(x))] <- NA
        library(impute)
        return(impute.knn(x))
    }else{
        return(x)
   }
}

# a simple way to see if the microarrays (samples) are relatively similar
plotDiagnostics <- function(toPlot, cont_cols, case_cols, suffix = '', rmvd = c()){
    library(limma)
    if (length(rmvd) > 0){
        print(paste('WARNING:',
                    length(rmvd),
                    'samples were removed during processing due to poor quality,',
                    'and will not be plotted on the QC plots.',
                    'Here are the removed samples:',
                    paste(rmvd, collapse = ', ')
                   ))
        toPlot <- toPlot[,!colnames(toPlot) %in% rmvd]
        cont_cols <- cont_cols[!cont_cols %in% rmvd]
        case_cols <- case_cols[!case_cols %in% rmvd]
    }
    colForPlots <- c(rep("black",length(cont_cols))
                     ,rep('red', length(case_cols))
                    )
    groups <- c(rep("control",length(cont_cols))
                , rep("case", length(case_cols))
               )
    # Print out a plot to visualize how consistent the expression values are at the population level.
    # An easy fix for NAs
    toPlot[is.na(toPlot)] <- 0
    variances <- unlist(apply(toPlot, 1, var))
    if(sum(variances == 0) > 0){
        print(paste("In plotDiagnostics, found",
                     sum(variances == 0),
                     "probes/genes with 0 variance.",
                     "Removing those for plotting purposes only."
                    )
              )
        toPlot <- toPlot[which(variances > 0),]
    }
    if(debug) print(summary(toPlot))
    png(paste0(pubdir, "/", geoID, "_", tisID, suffix, "_ECDFs.png"))
        first<-ecdf(toPlot[,1])
        plot(first
             , xlim = range(toPlot)
             , col = colForPlots[1]
             , ylab = 'Fraction of observations less or equal to X'
             , xlab = 'Signal intensity'
             , main = 'ECDF for each microarray'
             )
        for (i in 2:ncol(toPlot)){
            lines(ecdf(toPlot[,i]), col = colForPlots[i])
        }
    dev.off()
    library(scales)
    png(paste0(pubdir, "/", geoID, "_", tisID, suffix, "_cv.png"))
        cv_data <- rowCv(toPlot)
        plot(1:length(cv_data),cv_data[order(cv_data)],
             , xlab = 'Probes ranked by CV'
             , ylab = 'Coefficient of variance'
             , main = 'CV for each probe'
             , pch = 20
             , col = alpha('darkblue', 0.3)
             , log = 'y'
            )
    dev.off()
    pca <- prcomp(t(toPlot), center = TRUE, scale. = TRUE)
    pdf(paste0(pubdir, "/", geoID, "_", tisID, suffix, "_rawMDSPCA.pdf"))
        plotMDS(toPlot
                , col = colForPlots
                , labels = c(cont_cols, case_cols)
                , main = 'MDS with all probes'
                , xlab = "Leading dimension"
                , ylab = "Secondary dimension"
               )
        plot(pca$x[, 1]
             , pca$x[, 2]
             , main = "PCA with all probes"
             , xlab = "Principal Component 1"
             , ylab = "Principal Component 2"
             , type = 'n'
             )
        text(pca$x[, 1]
             , pca$x[, 2]
             , c(cont_cols, case_cols)
             , col = colForPlots
             )
        plot(pca
             , type = 'l'
             , main = 'Variance for PCs of raw probes'
             , pch = 20
             )
    dev.off()
    colnames(toPlot) <- paste(groups, colnames(toPlot), sep = "_")
    plotCorHeatmap(toPlot, suffix)
}

rowCv <- function(df){
    library(matrixStats)
    return(rowSds(df)/rowMeans(df))
}

plotCorHeatmap <- function(x, suffix = ''){
    corrs <- makeCorMat(x)
    width <- height <- 480 # default pixel size of R plots
    if(nrow(corrs) > 50) width <- height <- 960
    png(paste0(pubdir, '/', geoID, '_', tisID, suffix, '_sampleCorrelationHeatmap.png'), width = width, height = height)
      pheatmap(corrs)
    dev.off()
}

makeCorMat <- function(x){
    # Now also create a plot showing all of the pairwise correlations of the samples. Hope to see the cases and controls cluster together
    corrs <- matrix(nrow = ncol(x), ncol = ncol(x))
    colnames(corrs) <- colnames(x)
    rownames(corrs) <- colnames(x)
    for(i in 1:ncol(x)){
      for(j in 1:ncol(x)){
        corrs[i,j] = cor(x[,i], y = x[,j])
      }
    }
    return(corrs)
}

# This take care of converting probes to proteins, condensing to single uniProts,
# filtering for p and fc, and then writing the genes to the database
writeProbes <- function(probes.FCFiltered, mappingInfo){
    print("############################################################################")
    print("Converting probes to proteins")
    print("############################################################################")
    #=================================================================================
    # Convert probes to proteins
    #=================================================================================
    # set up the conversion functions
    source("sigGEO_mapper.R")
    
    if (is.null(probes.FCFiltered) || nrow(probes.FCFiltered) == 0){
        print(paste("WARNING: Something went horribly wrong we're not",
                    "filtering probes,",
                    "yet there there still no probes to report."
                  ))
    }else{
        print(paste("Prior to mapping, there were:",
                     nrow(probes.FCFiltered),
                     "probes."
                   ))
        if( miRNA ){
            mirmap <- read.table(mirMappingFile,
                                 header = TRUE,
                                 sep = "\t",
                                 stringsAsFactors = FALSE
                                 )
            mappingInfo$table <- data.frame(lapply(mappingInfo$table, as.character),
                                            stringsAsFactors = FALSE
                                           )
            forMerge <- splitMultMirs(mappingInfo$table, findMirCol(mappingInfo$table))
### also need to be able to handle the asterisks
# https://www.researchgate.net/post/Whats_difference_between_hsa-let-7a-5p_hsa-let-7a-3p_What_do_5p_3p_refer_to?#view=51854976d11b8bcd5c000033
# can I convert from/to 5p and 3p?
            yIdx <- find_ideal_idx(probes.FCFiltered[,1], forMerge)
            if(debug){
                print('Input for mapped_probes')
                print(head(probes.FCFiltered))
                print(yIdx)
                print(head(forMerge))
            }
            mapped_probes <- unique(
                             merge(probes.FCFiltered,
                                   forMerge,
                                   by.x = 1,
                                   by.y = yIdx
                                  ))
            mir_colname <- colnames(mapped_probes)[findMirCol(mapped_probes)]
            temp <- remove_conflicting_direction(
                     mapped_probes,
                     group_by_val = mir_colname
                    )
            # report some stats from what we've done.
            print(paste("Initial probe num",
                "Initial miRNA num",
                "miRNA num with consensus",
                "miRNA num with single probe",
                "Probe num with consensus",
                sep = " ; "
            ))
            print(paste(unlist(temp$stats),
                collapse = " ; "
            ))
            if (temp$stats$uniprotsWithProbeConsensus ==  temp$stats$uniprotsWithSingleProbe){
                consistDirScore <- 1.0
            }else{
                consistDirScore <- (temp$stats$uniprotsWithProbeConsensus - temp$stats$uniprotsWithSingleProbe) / (temp$stats$totalUniprots - temp$stats$uniprotsWithSingleProbe)
            }
            mapped_probes <- temp$data
            mirProbe2gene <- get_mirProbe2gene(mapped_probes, mirmap)
            idx <- findMirCol(mapped_probes)
            orig <- length(unique(mapped_probes[,idx]))
            new <- length(unique(mirProbe2gene[,1]))
            map_score <- new / orig
            print(paste('Originally there were',
                        nrow(mapped_probes),
                        'rows from the mapped probes corresponding to',
                        orig,
                        'miRNAs.',
                        new,
                        'of those miRs mapped to genes, a percentage of',
                        map_score * 100
                       ))
            if(debug) print(head(mirProbe2gene))
            protmap <- read.delim(file = EnsemblToUniprotMap,
                                  header = FALSE,
                                  sep = '\t'
                                 )
            colnames(protmap) <- c('UNIPROTKB', 'Ensembl')
            mapped_probes2prots <- unique(merge(mirProbe2gene, protmap, by = 'Ensembl'))
# The miRNA to protein scores are already stored such that bigger is better
# add direction so that below we can take into account both up and down reg'd miRNAs that taregt the same protein
            mapped_probes2prots$finalConfScore <- (1 - mapped_probes2prots$`q-value`) * mapped_probes2prots$score * mapped_probes2prots$direction
            reportMirResults(mapped_probes2prots)
            # giving a custom name proved to be an easier way to handle this
            colnames(mapped_probes2prots)[findMirCol(mapped_probes2prots)] <- 'miRNA_to_use'
# in mapping from miR to proteins, we want to take the maximal final confidence score
            temp <- data.frame(mapped_probes2prots %>%
# first remove any redundant protein-miR connections by taking the absoluate value max score (without losing the sign)
                group_by(UNIPROTKB,
                         miRNA_to_use
                        ) %>%
                filter(finalConfScore == absmax(finalConfScore)
                      ) %>%
# now aggregate across multiple miRs for a final prot level score
                group_by(UNIPROTKB
                        ) %>%
                summarise(uni_score = sum(finalConfScore)
                      ) %>%
                arrange(UNIPROTKB, uni_score)
                )
# The score sum may not be the most appropriate for these probabilities as they may exceed one
# but we just cap those here
            temp$uni_score[which(temp$uni_score > 1)] <- 1
            temp$uni_score[which(temp$uni_score < -1)] <- -1
            inds <- which(abs(temp$uni_score) > 0)
            significantUniprots <- temp[inds,]
            colnames(significantUniprots)[2] <- 'score'
            significantUniprots$direction <- sign(significantUniprots$score)
            significantUniprots$score <- abs(significantUniprots$score)
# FC isn't really relevant at the level of proteins as there are too many variables between prots and the miR FC
            significantUniprots[,'Fold Change'] <- 0 
        }else{
            # Now convert the significant probes, to Uniprot IDs
            tempList <- probesToUniprot(probes.FCFiltered,
                                        mappingInfo$table,
                                        mappingInfo$chipType,
                                        TRUE
                                       )
            map_score <- tempList$mappingScore
            tempList2 <- condenseAndFilterGenes(tempList$genes.all)
            consistDirScore <- tempList2$consistDirScore
            significantUniprots <- tempList2$significantUniprots
            significantUniprots$score <- 1 - asNum(significantUniprots[,"q-value"])
            print(paste("Number of proteins with p less than 0.05:",
                     length(which(significantUniprots[,"q-value"] <= 0.05))
                   ))
            print(paste("And there are",
                     nrow(significantUniprots),
                     "total proteins. That is",
                     length(which(significantUniprots[,"q-value"] <= 0.05)
                           ) / nrow(significantUniprots)
                   ))
        }
        #===============================================================================
        # insert results back into database
        #===============================================================================
        if (nrow(significantUniprots)>0){
           toPrintOut <- as.data.frame(matrix(nrow = nrow(significantUniprots), ncol = 7))
           toPrintOut[,1] <- NULL
           toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
           toPrintOut[,3] <- paste(significantUniprots$totalProbes,
                                   significantUniprots$portionOfProbesInThisDirection,
                                   sep = ";"
                                  )
           toPrintOut[,4] <- formatC(significantUniprots$score,
                                     format = 'e',
                                     digits = 3
                                    )
           toPrintOut[,5]<- significantUniprots$direction
           toPrintOut[,6]<- tisID
           toPrintOut[,7]<- round(asNum(significantUniprots[,"Fold Change"]), digits = 4)
           outFile <- paste0(outdir, "/", databaseTable, ".tsv")
           write.table(toPrintOut,
                       file = outFile,
                       quote = FALSE,
                       row.names = FALSE,
                       col.names = FALSE,
                       sep = "\t"
                      )
        }else{
            print(sprintf("%s has no significant expression", geoID))
        }
    }
    return(list(consistDirScore = consistDirScore, mappingScore = map_score ))
}

get_mirProbe2gene <- function(mapped_probes, mirmap){
    idx <- findMirCol(mapped_probes)
    mirProbe2gene <- unique(merge(mirmap,
                                  mapped_probes,
                                  by.x = 'miRNA',
                                  by.y = idx
                                  )
                            )
   # this is a pretty specific fix, but it addresses the only examples I've seen
    if (length(unique(mirProbe2gene[,1])) == 0){
        mapped_probes[,idx] <- as.character(mapped_probes[,idx])
        print('Unable to connect probes to proteins via miRNAs. Seeing if we can fix that...')
        strip_prefix_and_suffix <- function(x, idx){
            return(gsub("_st$", "", gsub("^hp_", "", x[idx], perl = TRUE), perl = TRUE))
        }
        mapped_probes[,idx] <- apply(mapped_probes, 1, function(x) {strip_prefix_and_suffix(x, idx)})
        print(head(mapped_probes))
        mirProbe2gene <- unique(merge(mirmap,
                                  mapped_probes,
                                  by.x = 'miRNA',
                                  by.y = idx
                                  )
                            )
    }
    return(mirProbe2gene)
}


absmax <- function(x) { x[which.max( abs(x) )][1]}

find_ideal_idx <- function(probes_vec, map_df){
    max_overlap <- 0
    yIdx <- 0
    for (i in 1:ncol(map_df)){
        overlap <- length(intersect(map_df[,i], probes_vec))
        if (overlap > max_overlap){
            yIdx <- i
            max_overlap <- overlap
        }
    }
    return(yIdx)
}

reportMirResults <- function(mapped_probes){
    cn <- colnames(mapped_probes)
    header <- c(findMirCol(mapped_probes),
                grep('^Ensembl$', cn),
                grep('^Gene ID$', cn),
                grep('^Fold Change$', cn),
                grep('^q-value$', cn),
                grep('^direction$', cn),
                grep('^score$', cn),
                grep('^UNIPROTKB$', cn),
                grep('^finalConfScore$', cn)
               )
    outFile <- paste0(outdir, "/miRNAResults.tsv")
    write.table(mapped_probes[,header],
                file = outFile,
                quote = FALSE,
                row.names = FALSE,
                col.names = TRUE,
                sep = "\t"
               )
}

defineMirPrefix <- function(){
    return('hsa-')
}
findMirCol <- function(x){
    mirPrefix <- defineMirPrefix()
    idx <- 1 # default to the first column
    maxN <- 0
    for (i in 1:ncol(x)){
        cnt <- length(grep(mirPrefix, x[,i]))
        if (cnt > maxN){
            idx <- i
            maxN <- cnt
        }
    }
    print(paste(maxN, 'matched',
                mirPrefix, 'in the miR matrix. Out of',
                nrow(x)
              ))
    if (debug && maxN == 0){
        print('Unable to find any column. Here is the whole matrix for inspection:')
        print(x)
    }
    return(idx)
}
splitMultMirs <- function(mapTable, idx, sep = "/"){
    toRet <- processMult(mapTable[1,], idx, sep)
    for (i in 2:nrow(mapTable)){
        toRet <- rbind(toRet, processMult(mapTable[i,], idx, sep))
    }
    colnames(toRet) <- colnames(mapTable)
    return(toRet)
}
processMult <- function(row, idx, sep){
    row <- as.character(row)
    distinctMirs <- unlist(strsplit(row[idx], sep, fixed = TRUE))
    toRet <- do.call("rbind", replicate(length(distinctMirs),
                                        row,
                                        simplify = FALSE
                     ))
    toRet[,idx] <- distinctMirs
    return(toRet)
}


remove_conflicting_direction <- function(x, group_by_val = "UNIPROTKB", rnaseq = FALSE){
    print(paste('Removing conflicting directions using', group_by_val))
    library(rlang)
    # set some parameters that will depend on what the data is
    if(rnaseq){
        q_colname = 'FDR'
        fc_colname = 'logFC'
    }else{
        q_colname = 'q-value'
        fc_colname = 'Fold Change'
    }
    # evaluate if 2/3 (or whatever it is set to in settings.R) of directions agree,
    # if so take mean or median FC, then combine q-values with fishers method
    totalMappedProbes <- nrow(x)
    # Get a count of directions for each uniprot (or whatever is keying this)
    directionTallies <- x %>% group_by_at(vars(one_of(c(group_by_val, 'direction')))) %>% tally
    # And a total number of probes for each uniprot
    totals <- x %>%  group_by_at(vars(one_of(group_by_val))) %>% summarise(n())
    uniprotsWithSingleProbe<- length(which(totals[,ncol(totals)] == 1 ))
    totalUniprots <- nrow(totals)

    # Then combine and get a portion for each direction
    # ID'ing the colname proved necessary once we started using a variable colname
    xIdx <- grep(group_by_val, colnames(directionTallies))
    yIdx <- grep(group_by_val, colnames(totals))
    summarised <- merge(data.frame(directionTallies), data.frame(totals), by.x = xIdx, by.y = yIdx)
    colnames(summarised) <- c(group_by_val, 'direction' , "directionTallies", "totalProbes")
    summarised$directionPortions <- summarised$directionTallies/summarised$totalProbes

    # Only keep those that meet our minimum portion
    # and for the others, we can't have any confidence of change.
    # so default all their values (q=1, FC=0)
    toKeep <- subset(summarised,summarised$directionPortions >= minProbePortionAgreeingOnDirection)
    toModify <- subset(summarised,summarised$directionPortions < minProbePortionAgreeingOnDirection)
    toModify <- subset(toModify, !(toModify[,group_by_val] %in% toKeep[,group_by_val]))
    print(paste("Found",
                nrow(toModify),
                group_by_val,
                "entries with inconsistent directions.",
                "Their q-values are being set to 1 and FC to 0.",
                "To view these entries, re-run with debug enabled"
                ))
    if (debug){
        print('The entries that are being defaulted:')
        print(toModify)
    }
    uniprotsWithProbeConsensus <- nrow(toKeep)

    # Now bring back in the original data so we can start to summarise
    keepersWithOrigInfo <- unique(merge(x, toKeep, by = group_by_val))
    probesWithConsensus <- nrow(keepersWithOrigInfo)
    for(i in 2:4){keepersWithOrigInfo[,i] <- asNum(keepersWithOrigInfo[,i])}
    summarisedConsensusUniprots <- data.frame(subset(keepersWithOrigInfo
                                            , keepersWithOrigInfo$direction.x == keepersWithOrigInfo$direction.y
                                       )%>%
                                       group_by_at(vars(one_of(c(
                                                group_by_val,
                                                'direction.x',
                                                'totalProbes',
                                                'directionPortions'
                                       ))))%>%
# the !! and sym are used to interpret the variable in use with summarise
                                       summarise(combinedQ = combineQs(!!sym(q_colname)),
                                                 medianFC = combineFCs(asNum(!!sym(fc_colname)))
                                       )
                                    )
    if(debug){print(colnames(summarisedConsensusUniprots))}
    preferred_cnames <- c(group_by_val
                         , "direction"
                         , "totalProbes"
                         , "portionOfProbesInThisDirection"
                         , "q-value"
                         , "Fold Change"
                         )
    colnames(summarisedConsensusUniprots) <- preferred_cnames
    # now attach the entries which we default b/c of their inconsistent direction
    defaulted_entries <- data.frame(toModify %>%
                                    group_by_at(vars(one_of(c(group_by_val, "totalProbes"
                                    )))) %>%
                                    summarise(direction = 0,
                                              portionOfProbesInThisDirection = max(directionPortions),
                                              q = 1,
                                              fc = 0
                                             )
                                   )
    defaulted_entries <- defaulted_entries[,c(1,3,2,4:6)]
    colnames(defaulted_entries) <- preferred_cnames
    print(head(defaulted_entries))
    return(list(data = rbind(summarisedConsensusUniprots, defaulted_entries),
                stats = list('totalMappedProbes' = totalMappedProbes,
                             'totalUniprots' = totalUniprots,
                             'uniprotsWithProbeConsensus' = uniprotsWithProbeConsensus,
                             'uniprotsWithSingleProbe' = uniprotsWithSingleProbe,
                             'probesWithConsensus' = probesWithConsensus
                            )
               )
           )
}

# This function takes all of the probe differential expression information,
# groups it by UNIPROT, and reports protein level Q-values and fold changes.
### I'm leaving the rnaseq option in here, though I don't think this is used by RNA-seq anymore
condenseAndFilterGenes <- function(genes.all, rnaseq = FALSE){
    temp  <- remove_conflicting_direction(genes.all, rnaseq = rnaseq)
    summarisedConsensusUniprots <- temp$data
    filtered <- subset(summarisedConsensusUniprots
                       , summarisedConsensusUniprots$"q-value" <= (qmax / 100)
                         & abs(summarisedConsensusUniprots$"Fold Change") > abs(log2(minFC))
                       )
    numFilteredUniprot <- nrow(filtered)
    # report some stats from what we've done.
    print(paste("Initial probe num",
                "Initial uniprot num",
                "Uniprot num with consensus",
                "Uniprot num with a single probe",
                "Probe num with consensus",
                "Uniprot num after q and FC filter",
                sep = " ; "
              ))
    print(paste(c(unlist(temp$stats),numFilteredUniprot),
                collapse = " ; "
              ))
    for(i in 2:4){filtered[,i] <- asNum(filtered[,i])}
    # Still need to calculate consistDirScore
    # what portion of uniprots with multiple probes have a consensus
    consistDirScore <- (temp$stats$uniprotsWithProbeConsensus - temp$stats$uniprotsWithSingleProbe) / (temp$stats$totalUniprots - temp$stats$uniprotsWithSingleProbe)
    return(list( significantUniprots = filtered, consistDirScore = consistDirScore))
    # Should we also report inconsistent genes in a separate file?
}

combineFCs <- function(fcs){
    return(median(fcs))
# other options would include:
# absolute value max (being sure to return the direction to the value)
# mean
}
combineQs <- function(qVals){
    return(min(qVals))
#    return(fishersMethod(qVals))
}

# a function to combine Pvals
fishersMethod <- function(pVals){
    return(pchisq( -2*sum(log(pVals)), 2*length(pVals), lower.tail = FALSE))
}

# function to combine GeoDE's chrdir
# another option would be to just take the average of the chrdirs,
# but I don't like that idea as much
combine_chrdir <- function(mod_chrdir){
    chrdir <- 1 - mod_chrdir # the 1 - is b/c I initially took 1- so it would fit with our current setup
    return (1 - mean(chrdir))
}

myGetElement <- function(myList, myName, default=NULL){
    if (myName %in% names(myList)) return(myList[[myName]])
    default
}

qcSig <- function(columnAssignments, sigPor, perfectPor, mappingScoresList, callingScoresList){
    print("############################################################################")
    print("QC'ing sig results")
    print("############################################################################")
    mappingScore <- myGetElement(mappingScoresList,"mappingScore", 0)
    consistDirScore <- myGetElement(mappingScoresList,"consistDirScore", 0)
    ccCorScore <- myGetElement(callingScoresList,"ccCorScore", 0)
    caseCorScore <- myGetElement(ccCorScore,"case", 0)
    controlCorScore <- myGetElement(ccCorScore,"contol", 0)
#    concordScore <- myGetElement(callingScoresList,"concordScore", 0)
    caseSampleScores <- scoreSampleNum(length(getElement(columnAssignments,"case")))
    contSampleScores <- scoreSampleNum(length(getElement(columnAssignments,"cont")))
# Not sure how to do this yet
#    sepScore <- separationScore()
    sigProbesScore <- scoreSigProbePor(sigPor)
    perfectProbesScore <- scoreSigProbePor(perfectPor, bestPor = 1e-3, worstWithScore = 0.05)
    scores <- c(mappingScore, consistDirScore, caseCorScore, controlCorScore
                #, concordScore
                , caseSampleScores, contSampleScores, sigProbesScore, perfectProbesScore
               )
    total <- mean(scores)
    scores <- c(scores, total)
    names(scores) <- c('mappingScore', 'consistDirScore', 'caseCorScore', 'controlCorScore'
# we're no longer using the concordance score as we're only using one program
                       #, 'concordScore'
                       , 'caseSampleScores', 'contSampleScores'
                       , 'sigProbesScore', 'perfectProbesScore', 'finalScore'
                      )
    print(scores)
    write.table(scores, file = paste0(outdir, '/sigqc.tsv'), sep = "\t", quote = FALSE)
}

scoreSampleNum <- function(n){
    return( 1 - (1 / (1 + exp((n - 8)/2))))
}

# this is totally arbitrary, but represents roughly the numbers we think work well with what we do
scoreSigProbePor <- function(sigPor, bestPor = 0.05, worstWithScore = 0.5){
    if(sigPor <= bestPor){
        # linear increase from 0.5 to 1 while x goes from 1 to 750
        return(0.5 + sigPor*0.5/bestPor)
    }else if(sigPor <= 2*bestPor){
        # linear decrease from 1 to 0.5 while x goes from 751 to 1500
        return(1 + (sigPor - bestPor )*-0.5/bestPor)
    }else if(sigPor <= worstWithScore){
        # linear increase from 0.5 to 0 while z goes from 1501 to 10000
        return(0.5 + (sigPor - 2*bestPor)*-0.5/(worstWithScore - 2*bestPor) )
    }else{
        return(0)
    }
}

check_cols <- function(control.cols, case.cols, x){
    # occasionally some samplea don't download/isn't available
    # this is here to report which one it is
    if (! all(c(control.cols, case.cols) %in% x)){
        inInds <- c(control.cols, case.cols) %in% x
        missing <- c(control.cols, case.cols)[! inInds]
        print(paste('The following columns were missing in our expression data:',
                     paste(missing, collapse = ", ")
                    )
             )
        print(paste('And the coulmn names are:',
                    paste(x,
                          collapse = ', '
                    )))
        return(FALSE)
    }
    return(TRUE)
}
