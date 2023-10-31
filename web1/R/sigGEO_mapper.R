#===========================================
# Settings for converting ID types to Entrez via a general mapper
#==========================================
# These are the possible, case insensitive, GPL column names that are currently recognized
# They are in order of preference.
# If you find a GPL that is unsuccessful with these, but you see a different column name that should work,
# just add it to this list, and also update the function below,
# which matches the column names to their respective conversion objects.
# Both of these are used solely in the map2EntrezViaOrgHsEg function

possibleIDTypes_Ranked <- c("entrez_gene", "ENTREZ_GENE_ID", "Entrez_ID"
                            ,"ensembl"
                            ,"gb_acc", "GB_LIST", "GB_ACC (ref)"
                            , "ref_seq", "refseq", "RefSeq_ID"
                            ,"unigene", "UniGeneHsID", "Unigene ID"
                            ,"uniprot", "swissprot"
                            ,"gene_symbol", "symbol", "geneName", "GenName", "Gene symbol", "GeneSymbol", "Gene Name", "SYMBOL", "ORF"
                            )

speciesNameFromEntrezMapFileName <- function(){
    parts <- unlist(strsplit(EntrezToUniprotMap,'.', fixed=TRUE))
    return(parts[2])
}

# This is for converting the idType to the actual mapping
idTypeToOrgHsEg <- function (idType){
    if(idType=="entrez_gene" ||
       idType=="ENTREZ_GENE_ID" ||
       idType=="Entrez_ID"
      ){
        return("notNeeded")
    }
### if there are any more idTypes added to this one,
### they should also be added within map2EntrezViaOrgHsEg,
### where ensembl is checked to see if it is a transcript or gene ID
#
# Determine the species we're using and then, essentially, alias the appropriate package
    species <- speciesNameFromEntrezMapFileName()
    if (species == 'mouse'){
        library(org.Mm.eg.db)
        org <- 'org.Mm'
    }else if (species == 'rat'){
        library(org.Rn.eg.db)
        org <- 'org.Rn'
    }else if (species == 'dog'){
        library(org.Cf.eg.db)
        org <- 'org.Cf'
    }else if (species == 'zebrafish'){
        library(org.MDr.eg.db)
        org <- 'org.Dr'
    }else{
        library(org.Hs.eg.db)
        org <- 'org.Hs'
    }
    if (idType=="ensembl"){
        return(as.data.frame(eval(parse(text=paste(org,'egENSEMBL', sep='.')))))
    }else if (idType=="ensemblTranscript"){ 
        return(as.data.frame(eval(parse(text=paste(org,'egENSEMBLTRANS', sep='.')))))
    }else if(idType=="gb_acc" ||
             idType=="GB_ACC" ||
             idType=="GB_LIST" ||
             idType=="GB_ACC (ref)"
           ){
        return(as.data.frame(eval(parse(text=paste(org,'egACCNUM2EG', sep='.')))))
    }else if(idType=="ref_seq" ||
             idType=="refseq" ||
             idType =="RefSeq_ID"
            ){
        return(as.data.frame(eval(parse(text=paste(org,'egREFSEQ2EG', sep='.')))))
    }else if(idType=="unigene" ||
             idType=="UniGeneHsID" ||
             idType=="Unigene ID"
            ){
        return(as.data.frame(eval(parse(text=paste(org,'egUNIGENE2EG', sep='.')))))
    }else if(idType=="uniprot" ||
             idType=="swissprot"
            ){
        return(as.data.frame(eval(parse(text=paste(org,'egUNIPROT', sep='.')))))
    }else if(idType=="gene_symbol" ||
             idType=="symbol" ||
             idType=="geneName" ||
             idType=="GenName" ||
             idType=="Gene symbol" ||
             idType=="GeneSymbol" ||
             idType=="Gene Name" ||
             idType=="SYMBOL" ||
             idType=="ORF"
           ){
        return(as.data.frame(eval(parse(text=paste(org,'egALIAS2EG', sep='.')))))
    }else{
        return(NULL)
    }
}

#=================================================
# The main function that calls the specific mappers below,
# To convert the probes to uniprot IDs via Entrez
# It has a recursive feature so that if the converted proportion isn't high enough,
# it will try again with a different ID type.
#=================================================

# After the significant probes are found, this will map them to uniprot, through Entrez.
# If Entrez is not available in that gpl, it will convert an ID type that is available to Entrez,
# and then go to Uniprot
# It also makes sure a high enough portion of the significant probes are mapping as well
probesToUniprot <- function (probes.all, gplTable, chipType,
                             canLoop, colsNotToUse = NA, canUseAlt = TRUE){
    # for debugging purposes I want see what the probes.all looks like
    if(debug){
        print("probes.all head")
        print(head(probes.all))
    }
    # Use the gpl table to get conversions for all of the probes
    probemapAndColInd <- gplTableToProbeMap(gplTable, chipType, colsNotToUse, canUseAlt, canLoop)
    # This happens if we've run out of possible converting types without have a successful conversion
    ### I'm trying a new approach, and I've been saving the best mapper,
    ### so now I should just be able to assign it to genes.all,
    ### and things should go back through the recursive calls, but with now need to re-run
    if(is.null(probemapAndColInd)){
        print("Finished trying all mappers ")
        genes.all <- bestMappingYet

    # Otherwise run normally
    }else{
        results <- probesToGenes(probemapAndColInd, probes.all)
        genes.all <- results$genes.all
        succesfulPortion <- results$succesfulPortion

        # See if we should try again with a different conversion
        if( is.nan(succesfulPortion) || (succesfulPortion) < minProbeToUniprotProportion){
            
            print(paste("Successful portion: ",
                        succesfulPortion,
                        ". And max mapping proportion ",
                        maxMappingPortion
                      ))
            if(succesfulPortion > maxMappingPortion){
                assign("maxMappingPortion", succesfulPortion, envir = .GlobalEnv)
                if(succesfulPortion > absoluteMinUniprotProportion){
                    assign("bestMappingYet", genes.all, envir = .GlobalEnv)
                }
            }
            print(paste("Too few probes mapped from Entrez to UniProt.",
                        " Re-trying with a different conversion"
                       ))
            #
            # If we used the altmapper, the easiest way I had to do this
            # was just have a separate function
            #
            if(probemapAndColInd$colIndex == "altMapperUsed"){
                print(paste("The alternative mapper was used,",
                           "but did not have a high Entrez to Uniprot conversion.",
                           "Now trying using just the column names and ID types."
                          ))
                genes.all <- probesToUniprot(probes.all, gplTable, chipType,
                                             canLoop, canUseAlt = FALSE
                                            )
            #
            # Otherwise Re-run the mapping, but from the gpl table,
            # cut out the column we just used, that resulted in a low mapping proportion
            # There's no reason to check for the alternative mappers now
            #
            }else{
                if(is.na(colsNotToUse)){
                    genes.all <- probesToUniprot(probes.all, gplTable, chipType,
                                                 canLoop,
                                                 colsNotToUse = probemapAndColInd$colIndex,
                                                 canUseAlt = FALSE
                                                )
                }else{
                    genes.all <- probesToUniprot(probes.all, gplTable, chipType,
                                                 canLoop,
                                                 colsNotToUse = c(colsNotToUse, probemapAndColInd$colIndex),
                                                 canUseAlt = FALSE
                                                )
                }
            }
        }
    }
    # the recursion is resulting in list of lists, so unwind it all
    while(class(genes.all) == "list"){
        genes.all <- genes.all$genes.all
    }
    return(list(genes.all = genes.all, mappingScore = maxMappingPortion))
}


# This uses the probemap and converts the probes to uniprot genes
probesToGenes <- function (probemapAndColInd, probes.all){
    probemap <- probemapAndColInd$probemap
    genes.all <- merge(probes.all, probemap, by.x = "Gene ID", by.y = 1)
    
    # Parse through the Entrez Gene IDs, and if there are multiple, create more lines
    entrezColInd <- grep("ENTREZ_GENE_ID", colnames(genes.all))
    if( nrow(genes.all) >= 1 && is.numeric(entrezColInd)){
        genes.all <- parseMultipleEntrezEntries(genes.all, entrezColInd)
    }
    #Print out the whole probemap, for debugging
    if(debug){
        options(max.print = 1000000)
        print(paste("probemap_begin:",
                    genes.all[,c("Gene ID", "ENTREZ_GENE_ID", "q-value")],
                    "probemap_end"
                   ))
        options(max.print = 10000)
    }
    # Now convert these differential probes to genes, via Entrez to Uniprot
    # This is a table pulled down and separately processed that maps uniProt to Entrez
    protmap <- read.delim(file = EntrezToUniprotMap, header = FALSE, sep = '\t')
    colnames(protmap)[1] <- "UNIPROTKB"
    colnames(protmap)[2] <- "ENTREZ_GENE"
    
    # Quickly check how many of the differential probes we were actually able to get to UniProt
    geneNumberBeforeMerge<-length(unique(probes.all[,"Gene ID"]))
    print(paste("Before merging there were",
                geneNumberBeforeMerge,
                "probes with significant differences"
               ))
    genes.all <- merge(genes.all, protmap, by.x = "ENTREZ_GENE_ID", by.y = "ENTREZ_GENE")
    geneNumberAfterMerge <- length(unique(genes.all[,"Gene ID"]))
    succesfulPortion <- geneNumberAfterMerge/geneNumberBeforeMerge
    print(paste(succesfulPortion, 
                "portion of significant probes successfully converted to UniProt."
               ))
    genes.all <- unique(genes.all[,c("Gene ID", "UNIPROTKB", "q-value", "direction", "Fold Change")]) 
    
    return(list(succesfulPortion = succesfulPortion, genes.all = genes.all))
}

parseMultipleEntrezEntries <- function (genes.all, entrezColInd){
    for (i in 1:nrow(genes.all)){
        if(grepl("///", genes.all[i,entrezColInd], fixed = TRUE)){
            individualEntrez <- strsplit(as.character(genes.all[i,entrezColInd]), "///", fixed = TRUE)
            
            firstEntrez <- c(genes.all[i,])
            firstEntrez[entrezColInd] <- individualEntrez[[1]][1]
            
            names(firstEntrez) <- colnames(genes.all)
            genes.all[i,] <- firstEntrez
            
            for(j in 2:length(individualEntrez[[1]])){
                toAdd <- c(genes.all[i,])
                toAdd[entrezColInd] <- individualEntrez[[1]][j]
                names(toAdd) <- colnames(genes.all)
                genes.all <- rbind(genes.all[1:i,], toAdd, genes.all[(i+1):nrow(genes.all),])
            }
        }
    }
    return(genes.all)
}


#=================================================
# Functions that convert from probes to Entrez
#=================================================

# This function checks for alternative mappers, and uses them if they are there
# Otherwise it calls a similar program that uses a general mapper

gplTableToProbeMap <- function (gplTable, chipType, colsNotToUse, canUseAlt, canLoop){
    # Sometime there are libraries available to convert, those are preferable
    #
    if(canUseAlt && grepl("HuGene-1_0-st", chipType) ) {
        probemapAndCol <- HuGeneMapper(gplTable) 
    }else if(canUseAlt && grepl("HG-U133B", chipType) ) {
        probemapAndCol <- HgU133BMapper(gplTable)
     }else if(canUseAlt && grepl("HG-U133_Plus_2", chipType) ) {
        probemapAndCol <- HgU133Plus2Mapper(gplTable)
     }else if(canUseAlt && grepl("HuEx-1_0-st", chipType) ) {
        probemapAndCol <- HuEx10Mapper(gplTable)
    }else{
    # If none of the alt mappers worked, or we didn't even try them
    #, try the column headers, and a general converter
    #         
    # Check to see if any columns have been tried and have failed.
    # If so, remove them, and pass it to the mapper, otherwise pass as is
    #
        if(is.na(colsNotToUse)){
            probemapAndCol <- map2EntrezViaOrgHsEg(gplTable, canLoop, chipType)
        }else{
            tableToUse <- gplTable[,-1*colsNotToUse]
            probemapAndCol <- map2EntrezViaOrgHsEg(tableToUse, canLoop, chipType)
            # The col index may actually not be the correct original column index due to recursion,
            # so I need to find out what it is
            if(! is.null(probemapAndCol)){
                idTypeUsed <- colnames(tableToUse)[probemapAndCol$colIndex]
                probemapAndCol$colIndex <- grep(paste0("^",idTypeUsed,"$"),
                                                colnames(gplTable),
                                                ignore.case = TRUE,
                                                value = FALSE
                                               )
            }
        }
    }
    
    # Make sure the name is correct, and return is
    if(! is.null(probemapAndCol)){
        colnames(probemapAndCol$probemap)[2] <- "ENTREZ_GENE_ID"
    }
    return(probemapAndCol)
}

#=================================================
# Alternative mapper functions
#=================================================
# These are microarrays where there is an existing mapper
# Each of the below is a different library/microarry type
HuGeneMapper <- function(gpl){
    print("To Entrez mapper: hugene10sttranscriptcluster")
    library(hugene10sttranscriptcluster.db)
    e <- hugene10sttranscriptclusterENTREZID
    return(generalAltMapper(gpl, e))
}

HgU133BMapper <- function(gpl){
    print("To Entrez mapper: hgu133b")
    library(hgu133b.db)
    e <- hgu133bENTREZID
    return(generalAltMapper(gpl, e))
}

HgU133Plus2Mapper <- function(gpl){
    print("To Entrez mapper: hgu133Plus2")
    library(hgu133plus2.db)
    e <- hgu133plus2ENTREZID
    return(generalAltMapper(gpl, e))
}

# This one wasn't working
HuEx10Mapper <- function(gpl){
    print("To Entrez mapper: HuEx1")
    library(huex10stprobeset.db)
    e <- huex10stprobesetENTREZID
    return(generalAltMapper(gpl, e))
}

# These alternative mappers use the same approach
# Once the specifics are done, this function will do the repetitive bits
generalAltMapper <- function(gplTable,e){
    mapped_probes <- mappedkeys(e)
    eList <- as.list(e[mapped_probes])
    emap <- cbind(names(eList), eList)
    colnames(emap)[1] <- "ID"
    colnames(emap)[2] <- "ENTREZ_GENE_ID"
    probemap <- gplTable["ID"]
    probemap <- merge(probemap, emap, by="ID")
    
    successfulMergePortion <- (nrow(probemap) / nrow(gplTable))

    probemap[2] <- factor(unlist(probemap[2]))

    print(paste("In using the altmapper to convert to Entrez Gene ID",
                successfulMergePortion,
                "were successfully converted."
               ))

    return(list(probemap = probemap, colIndex = "altMapperUsed"))
}

#=================================================
# A general function to map to Entrez
#=================================================
# This function takes a gpl file without all of the header
# It searches through the names of the columns looking for
# specific strings corresponding to ID types (like genBank, RefSeq, etc)
# Currently the ID types supported are defined at the top of this file
map2EntrezViaOrgHsEg <- function(gplTable, canLoop, chipType){
    columnNames<- as.vector(colnames(gplTable))
    # Look through the table, and find the best heading, get it, and it's column index
    listWithIDAndCol <- findIDType(possibleIDTypes_Ranked,columnNames)
    idType <- listWithIDAndCol$ID
    colIndex <- listWithIDAndCol$columnIndex

    # This is what happens if we've looked through all of the columns
    if(is.null(idType)){
        print("Unable to find an ID type to convert to Entrez")
        print(paste("The max seen",
                    maxMappingPortion,
                    "And the absolute min",
                    absoluteMinUniprotProportion
                   ))
        if(canLoop && maxMappingPortion > absoluteMinUniprotProportion){
            return(NULL)
        }else{
            print(paste("WARNING: Unable to find an ID type",
                        "we support to convert the probes to Uniprots IDS,",
                        "and unable to loop back.  Giving up now."
                       ))
            quit('no', as.numeric(exitCodes['unknownIDTypeError']))
        }
    }    
    
    # Now pull out the relevant parts from the gpl:
    # column1 contains the probe IDs, and the other the ID type IDs
    # The problem is that agilent microarrays don't
    # have the probenames in the first column
    # Check the chipType for Agilent,
    # and if it is don't select column one,
    # but instead do a string match.
    probeColumnInd <- 1
    if(debug){print(paste0("The header for the GPL is: ", columnNames))}
    #if (grepl("ProbeName", columnNames)){
    #    probeColumnInd <- grep("ProbeName", columnNames)
    #}else 
    if(grepl('agilent', chipType, ignore.case = TRUE)){
        if(any(grepl('ProbeName', columnNames, ignore.case = TRUE))){
            probeColumnInd <- grep('ProbeName', columnNames, ignore.case = TRUE)
        }else if(any(grepl('SPOT_ID', columnNames, ignore.case = TRUE))){
            probeColumnInd <- grep('SPOT_ID', columnNames, ignore.case = TRUE)
        }else if(any(grepl('Reporter.Name', columnNames,ignore.case = TRUE))){
            probeColumnInd <- grep('Reporter.Name', columnNames, ignore.case = TRUE)
        }else if(any(grepl('name', columnNames,ignore.case = TRUE))){
            if (! any(grepl('gene name', columnNames, ignore.case = TRUE))){
                probeColumnInd <- grep('name', columnNames, ignore.case = TRUE)
            } else {
                print("Found Agilent column headers that include mulitple instances of 'name'. Trying to avoid using 'Gene Name'")
                name_match <- grep('name', columnNames, ignore.case = TRUE)
                gn_match1 <- grep('gene name', columnNames, ignore.case = TRUE)
                gn_match2 <- grep('gene_name', columnNames, ignore.case = TRUE)
                gn_match <- c(gn_match1, gn_match2)
                for (i in name_match){
                    if (!(i %in% gn_match)){
                        print(paste0("Using the first, non-'Gene Name' header:", columnNames[i]))
                        probeColumnInd <- i
                        break
                    }
                }
            }
        }else{
            print(paste("WARNING: This is an Agilent microarray,",
                        "but unable to find a column header matching 'Reporter.Name' or any other preferred terms.",
                        "Using the first column as the probe name."
                      ))
        }
        if(debug){print(paste0("probeColumnInd is: ", probeColumnInd, " and the header corresponding is ", columnNames[probeColumnInd]))}
    }
    if(length(probeColumnInd) > 1){
        probeColumnInd <- probeColumnInd[1]
    }
    map <- gplTable[,c(probeColumnInd,colIndex)]
    for (i in 1:ncol(map)){
        map[,i] <- trimws(map[,i])
    }
    # For some chips, there are a portion of probes that have multiple genes listed for that probe
    # This will parse through those and make each gene listed its own row
    # so far the only separating characters I've seen are // and ,
    # I handle each individually
    if(any(grepl("//", as.character(map[,2]), fixed = TRUE))){
        map <- splitUpMultipleMapIDs(map, "//")
    }
    if(any(grepl(",", as.character(map[,2]), fixed = TRUE))){
        map <- splitUpMultipleMapIDs(map, ",")
    }
    
    if(debug){
        print(head(map))
        print(tail(map))
    }
    # For ensembl IDs transcripts are totally different,
    # and instead we need to check for them and change the idType accordingly
    # Just in case it's a mix of gene and transcript
    print("Removing transcript information , if applicable")
    if(idType == "ensembl" &&
       sum(grep('^ENST', as.vector(map[,2]))) > sum(grep('^ENSG', as.vector(map[,2])))
    ){
        idType <- "ensemblTranscript"
    }else{
        # Remove transcript information from a GB_ACC or refseq
        # e.g. NM_014332.1 is the same as NM_014332,
        # except that it includes transcript info, and that will keep it from mapping
        map[,2] <- apply(map, 1, function(x) ifelse(grepl("\\.\\d+$",
                                                          as.character(x[2]),
                                                          perl = TRUE
                                                         ),
                                                    strsplit(as.character(x[2]),
                                                             "\\.\\d+$",
                                                             perl = TRUE
                                                            )[[1]],
                                                    x[2]
                                                   )
                        )
    }
   
    # And figure out how we're going to do the conversions, given this idType
    print(paste0("To Entrez mapper: ", idType))
    toEG <- idTypeToOrgHsEg(idType)

    # Now that we have a mapper,
    # go through and see what proportion actually successfully map.
    # First we check the only situation where we won't be using OrgHsEg
    # In this case a single string was returned, not a dataframe
    if(! is.data.frame(toEG)){
        if(toEG=="notNeeded"){ 
            # This happens if the ID type we found was already Entrez
            print("Using Entrez column in GPL table")
            probemap <- gplTable[,grepl("^id$|entrez", colnames(gplTable), ignore.case = TRUE)]
            return(list(probemap=probemap, colIndex=colIndex))
        }else{
            print(paste("WARNING: toEG is not a dataFrame,",
                        "but it doesn't match the flags we support.",
                        "Something went wrong in finding the ID type."
                      ))
            quit('no', as.numeric(exitCodes['toEGError']))
        }
    }else if(nrow(toEG) > 0){
        # Otherwise we have an idType, that we want to convert to Entrez
        xMatch <- colnames(toEG)[2]
        yMatch <- colnames(map)[2]
        merged <- merge(toEG, map, by.x=xMatch,by.y=yMatch)
        successfulMergePortion <- (nrow(merged) / nrow(map))
        print(paste("In converting from ",
                    idType,
                    "to Entrez Gene ID",
                    successfulMergePortion,
                    "were successfully converted."
                  ))
        return(list(probemap = merged[,c(3,2)], colIndex = colIndex))
    }else{
        print(paste("WARNING: Unable to load",
                    idType,
                    "to Entrez mapping package:",
                    mapper,
                    ". So we cannot convert probes to proteins."
                  ))
        quit('no', as.numeric(exitCodes['mapperLoadingError']))
    }
}

# For some chips, there are a portion of probes that have multiple genes listed for that probe
# This will parse through those and make each gene listed its own row
splitUpMultipleMapIDs <- function (map, sepChar){
    print(paste("Found at least once list of protein IDs mapping",
                "to a single probe that were separated by",
                sepChar,
               ". Attempting to split now"))
    splittingFunction <- function(rowToSplit, sepChar){
        if( any(grepl(sepChar, rowToSplit[2], fixed = TRUE))){
            individualIDs <- unlist(strsplit(rowToSplit[2], sepChar, fixed = TRUE))
            # some of them have 3 /s, so gsub out any remaining as well as any spaces
            individualIDs <- gsub("/", "", individualIDs, fixed = TRUE)
            individualIDs <- gsub(" ", "", individualIDs, fixed = TRUE)
            toReturn <- cbind(rep(rowToSplit[1],length(individualIDs)), individualIDs)
            colnames(toReturn) <- names(rowToSplit)
            return(toReturn)
        }else{
            return(rowToSplit)
        }
    }   
    mapList <- apply(map, 1, function(x){splittingFunction(x, sepChar)})
    expandedMap <- do.call("rbind", mapList)
    return(expandedMap)
}

# This function loops through the potential IDs you provide,
# in a ranked order (i.e. it will try to map the first entry first)
# and looked for the same string in the columns provided
# It returns a list that contains 2 things:
# The idType that it settled on (i.e. what matched)
# And the index of that column with the match
findIDType <- function (potentialIDs, columnNames){
    for (i in 1:length(potentialIDs)){
        colIndex <-grep(paste0("^", potentialIDs[i],"$"),
                        columnNames,
                        ignore.case = TRUE,
                        value = FALSE
                       )
        if(length(colIndex) > 0){
            # this should return the column index of the ID matching one of our ID types
            return(list(ID = potentialIDs[i], columnIndex = colIndex))
        }
    }
    return(NULL) # return NULL if unable to find anything
}
