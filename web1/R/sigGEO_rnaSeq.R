# one large function that handles all of the RNA-seq differential calls
analyzeAndReportRNASeqData <- function(){
    debug <- TRUE;
#================================================================================
# Get setup
#================================================================================
    # read in the file that was created with metaGEO and the whole bash pipeline.
    txi <- readRDS(paste0(microarrayDir,"/rnaSeqAnalysis/uniprot_expression_data.rds"))
    countsData <- txi$counts
    if(any(grepl("_", colnames(countsData)))){
        countsColNames <- gsub("_", "", colnames(txi$counts))
        colnames(txi$counts) <- countsColNames
        colnames(txi$length) <- countsColNames
    }else{
         countsColNames <- colnames(txi$counts)
    }

    if(file.exists(paste0(microarrayDir, "/", fullGeoID, "_sampleToSrrConversion.csv"))){
        # This maps the column names between how they are listed in our expression data table
        # and how we've selected case/control data.
        # The uniprot_expression_data.rds will have columns corresponding to merged SRRs
        # (i.e. there will be one merged SRR per SRX)
        # The case/control IDs are either GSMs (GEO) or SRXs (BIO)
        gsmToSrxConversion <- read.csv(paste0(microarrayDir,
                                              "/",
                                              fullGeoID,
                                              "_sampleToSrrConversion.csv"
                                             ),
                                        header = TRUE
                                      )
        if(debug){print(paste("before:", countsColNames))}
        print(gsmToSrxConversion)
        for ( i in 1:length(countsColNames)){
            indexToUse <- which(as.character(gsmToSrxConversion[,2]) == countsColNames[i])
            print(countsColNames[i])
            print(indexToUse)
            countsColNames[i] <- as.character(gsmToSrxConversion[indexToUse, 1])
        }
        colnames(txi$counts) <- countsColNames
        colnames(txi$length) <- countsColNames
        if(debug){print(paste("after:", countsColNames))}
    }

    # find the cases or controls
    columnAssignments <- parseDataColumns()
    if(debug){print(columnAssignments)}

    # Now reorganize the countsData so case and controls are grouped
    x <- buildRNASeqX(txi$counts, columnAssignments$case,columnAssignments$cont)
    temp <- filterNonexpressedGenes(x)
    x <- temp$x
    removed_samples <- temp$rmvd
    xForPlotting <- thoroughLog2ing(x)
    # these are the labels necessary for edgeR
    groups<-rep("filler",ncol(x))
    final_controls <- list()
    final_cases <- list()
    for (i in 1:ncol(x)){
        samp <- colnames(x)[i]
        if (samp %in% columnAssignments$cont){
            groups[i] <- 'control'
            final_controls[[length(final_controls)+1]]=samp
        }else if (samp %in% columnAssignments$case){
            groups[i] <- 'case'
            final_cases[[length(final_cases)+1]]=samp
        }else{
            print('There was an error in the case/control assignments. Quitting.')
            quit('no', as.numeric(exitCodes['usageError']))
        }
    }
    names(groups) <- colnames(x)
    final_columnAssignments <- list(case=unlist(final_cases), cont=unlist(final_controls))
    if (! scRNAseq){
        print('Running bulk RNA-seq gene expression analysis...')
        significantUniprots <- run_edgeR_tximport(x, txi$length, final_columnAssignments, groups)
    }else{
        print('Running single cell RNA-seq gene expression analysis...')
        temp <- run_scde(x, groups)
        significantUniprots <- temp$sig
        removed_samples <- c(removed_samples, temp$rmvd)
    }
    plotDiagnostics(xForPlotting, final_columnAssignments$cont, final_columnAssignments$case, removed_samples)
    # write out in the proper format for putting into the table
    if (nrow(significantUniprots) > 0){
        toPrintOut <- as.data.frame(matrix(nrow = nrow(significantUniprots), ncol = 7))
        toPrintOut[,1] <- NULL
        toPrintOut[,2] <- significantUniprots[,"UNIPROTKB"]
        # this isn't really applicable for RNA-seq
        toPrintOut[,3] <- '1;1'
        toPrintOut[,4] <- formatC((1 - asNum(significantUniprots[,"q-value"])),
                                   format = 'e',
                                   digits = 3
                                  )
        toPrintOut[,5] <- significantUniprots[,"direction"]
        toPrintOut[,6] <- tisID
        toPrintOut[,7] <- significantUniprots[,"Fold Change"]
        outFile <- paste0(outdir, "/", databaseTable, ".tsv")
        write.table(toPrintOut, file = outFile, quote = FALSE,
                    row.names = FALSE, col.names = FALSE, sep = "\t")
        pdf(paste0(pubdir, "/", fullGeoID, "_", tisID, "_sigPCAMDS.pdf"))
        templist <- cleanUpSigProbeDFs(significantUniprots
                                       , 'RNAseq'
                                       , xForPlotting
                                       , final_columnAssignments$case
                                       , final_columnAssignments$cont
                                       , calcScore = TRUE
                                       , geneColName = "UNIPROTKB"
                                      )
        dev.off()
    }else{
        print(sprintf("%s has no significant expression", fullGeoID))
        templist <- list(ccCorScore = list(control = 0, case = 0))
    }
    qcSig(final_columnAssignments
          , length(which(significantUniprots[,"q-value"] <= 0.05)) / nrow(significantUniprots)
          , length(which(significantUniprots[,"q-value"] == 0)) / nrow(significantUniprots)
          , list(consistDirScore = 1.0
                 , mappingScore = 1.0
                 )
          , list(ccCorScore = templist$ccCorScore, concordScore = 1.0)
         )
}


run_scde <- function(x, groups){
    groups <- factor(groups)
# to ensure the counts are stored as integers
    x <- apply(x,2,function(y) {storage.mode(y) <- 'integer'; y})
# taken straight from http://hms-dbmi.github.io/scde/diffexp.html
# plus the FDR calc from https://hms-dbmi.github.io/scw/differential-expression.html
    wd <- getwd()
    setwd(pubdir)
    o.ifm <- scde.error.models(counts = x,
                               groups = groups,
                               n.cores = 1,
                               threshold.segmentation = TRUE,
                               save.crossfit.plots = FALSE,
                               save.model.plots = TRUE,
                               )
    valid.cells <- o.ifm$corr.a > 0
    o.ifm <- o.ifm[valid.cells, ]
    if(sum(valid.cells) != length(valid.cells)){
        print(paste(length(valid.cells) - sum(valid.cells),
                    'cells were filtered out due to poor fit',
                    '- usually indicative of poor quality samples.',
                    'The following cells/samples were removed:'
                    ))
        print(paste(rownames(o.ifm)[!valid.cells], collapse = ', '))
        rmvd <- rownames(o.ifm)[!valid.cells]
    }else{
        rmvd <- c()
    }
    o.prior <- scde.expression.prior(models = o.ifm, counts = x, length.out = 400, show.plot = FALSE)
    # run differential expression tests on all genes.
    ediff <- scde.expression.difference(o.ifm, x, o.prior, groups = groups, n.randomizations = 100, n.cores = 1)
    p.values.adj <- 2*pnorm(abs(ediff$cZ),lower.tail = FALSE) # Adjusted to control for FDR
    de <- cbind(rownames(ediff),abs(ediff[,2]),p.values.adj, sign(ediff[,2]))
    colnames(de) <- c("UNIPROTKB","Fold Change","q-value", 'direction')
    setwd(wd)
    return(list(sig = de, rmvd = rmvd))
}

run_edgeR_tximport <- function(x, txi_length, columnAssignments, groups, local_minCPM = minCPM){
    library(edgeR)
    #================================================================================
        # Run edgeR
    #================================================================================
    # first normalize using data from tximport
    # pulled from https://bioconductor.org/packages/devel/bioc/vignettes/tximport/inst/doc/tximport.html#edger
    normMat <- txi_length[match(rownames(x), rownames(txi_length)),]
    normMat <- normMat[,match(colnames(x), colnames(normMat))]
    normMat <- normMat/exp(rowMeans(log(normMat)))
    o <- log(calcNormFactors(x/normMat)) + log(colSums(x/normMat))
    y <- DGEList(x)
    y$offset <- t(t(log(normMat)) + o)
    results <- run_core_edgeR(groups, y)
    results$UNIPROTKB <- rownames(results) # make the protein names a column
    results$direction <- sign(results$logFC)
    results$logFC <- abs(results$logFC)
    results[,'Fold Change'] <- results$logFC
    results['q-value'] <- results$FDR
    return(results)
}    

run_core_edgeR <- function(groups, y){
    # Now run a general linear model analysis 
    # first create a design matrix
    design <- model.matrix(~groups)
    rownames(design) <- colnames(y)
    
    # Next we estimate the overall dispersion for the dataset,
    # to get an idea of the overall level of biological variability:
    print(paste('estimateGLMCommonDisp: estimate overall dispersion for the dataset,',
                'to get an idea of the overall level of biological variability'))
    dgeList.countFiltered.normd.glmDisp <- estimateGLMCommonDisp(y, design, verbose = TRUE)
    
    # The square root of the common dispersion
    # gives the coefficient of variation of biological variation.
    # Then we estimate gene-wise dispersion estimates,
    # allowing a possible trend with average count
    dgeList.countFiltered.normd.glmDisp <- estimateGLMTrendedDisp(dgeList.countFiltered.normd.glmDisp,
                                                                  design
                                                                 )
    dgeList.countFiltered.normd.glmDisp <- estimateGLMTagwiseDisp(dgeList.countFiltered.normd.glmDisp,
                                                                  design
                                                                  )
    
    # Now proceed to determine differentially expressed genes. Fit genewise glms:
    fit <- glmFit(dgeList.countFiltered.normd.glmDisp, design)
    
    # Finally get the differential genes, with p-values
    d <- dgeList.countFiltered.normd.glmDisp
    options(max.print = 1e7) # just to make sure everything gets printed
    comparisonsName <- "control_vs_case"
    # actually run the glm
    lrt <- glmLRT(fit, coef = 2)
    # the p-value and fold change for all genes, as a df for easy printing
    return(as.data.frame(topTags(lrt, n = nrow(lrt))))
}


# I'm trying out a scaling of the q-values,
# but I'm doing it after everything else is said and done, just to avoid having to change settings.
# this way this will only effect pathsum, which is what uses this output file
scale_rnaseq_q <- function(x){
 # heuristically this got the distribution of scores to look like what we had for chrDir
    return((1/(1+exp(-x)) - 1/(1+exp(-0.95))) * 10 + 0.9)
}

buildRNASeqX <- function(expData, case.cols, control.cols){
    # Build an array with all of the data
    ignore <- check_cols(case.cols,control.cols,colnames(expData))
    x <- cbind(expData[,control.cols], expData[,case.cols])
    for (j in 1:ncol(x)) x[,j] <- asNum(x[,j])
    x <- as.matrix(x)
### Not currently doing this, but we could if need be.
### If we did we would need to take care of the row names
#    if(runSVA) x <- cleanMatViaSVA(x, control.cols, case.cols)
    if (debug) print(summary(x))
    return(x)
}

filterNonexpressedGenes <- function(x){
    library(scde)
    filtered <- clean.counts(x)
    print(paste("Filtered", nrow(x) - nrow(filtered), "transcripts as too lowly expressed"))
    print(paste("Filtered", ncol(x) - ncol(filtered), "samples as too poorly profiled"))
    rmvd <- setdiff(colnames(x), colnames(filtered))
    return(list(x = filtered, rmvd = rmvd))
# Prior to plat2295 (implementing scRNA-seq, this was the manual filtering we did
    library(matrixStats)
    firstClass <- rowMedians(x[,1:firstClassCnt])
    secondClass <- rowMedians(x[,-1*seq(1, firstClassCnt)])    
    filtered <- x[which(firstClass > 0 & secondClass > 0),]
    print(paste("Filtered", nrow(x) - nrow(filtered), "transcripts as too lowly expressed"))
    return(filtered)
}
