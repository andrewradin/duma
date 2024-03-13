#====================================
# Load settings and generally set-up
#====================================
source("settings.R")
# read and process command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
    warning("usage: samGEO.R GDSID REcID")
    quit(10)
}

geoID <- args[1]
tisID <- as.integer(args[2])

# this is the proportion of significant genes that are successfully converted to Uniprot IDs
minEnsemblToUniprotProportion<-0.9
minIDTypeToEntrezPortion<-0.58 # Unfortunately I've had to just set this empirically

# open log files
outfn <- sprintf("%s/sigGEO_%s_%d.log", logdir, geoID, tisID)
outlog <- file(outfn, open="wt")
errfn <- sprintf("%s/sigGEO_%s_%d.err", logdir, geoID, tisID)
errlog <- file(errfn, open="wt")
sink(outlog, append=FALSE, type="output", split=FALSE)
sink(errlog, append=FALSE, type="message", split=FALSE)

# load necessary libraries
library(RMySQL)
library(GEOquery)
library(samr)
library(org.Hs.eg.db)
#library(Biobase)

# setup macros for database connection
mychannel <- dbConnect(MySQL(), user=mysql_user, host=mysql_host)
query <- function(...) dbGetQuery(mychannel, ...)

#=================================================
# Define functions
#=================================================

# Give this function a tring to search for, and a table with column names, and it will return a logical:
# true of the string is in the column names, False if not.
checkIfStringIsInColnames <- function(string, table){
	if(length(grep(string, colnames(table), ignore.case=TRUE,value=F))>0){
		return(TRUE)
	}else{
		return(FALSE)
	}
	
}

# This function takes the output of Table(gpl) (i.e. a gpl file without all of the header)
# It searches through the names of the columns looking for specific strings corresponding to ID types (like genBank, RefSeq, etc)
# It has a recursive feature so that if the converted proportion isn't high enough it will try again with a different ID type.
# Currently the ID types supported are defined in settings.R
map2EntrezViaOrgHsEg <- function(gplTable,origCols=NA){

		columnNames<- as.vector(colnames(gplTable))
		
		# This is a hacky little fix to make sure we print out the original column names, even if we have to do several iterations(and thus remove some of the column names)
		if(is.na(origCols)){
			origCols=columnNames
		}
		
		listWithIDAndCol <- findIDType(possibleIDTypes_Ranked,columnNames)
		idType <- listWithIDAndCol$ID
		colIndex <- listWithIDAndCol$columnIndex

		if(is.null(idType)){
			print("Unable to find an ID type to convert to Entrez.")
			print(paste("Column names from the GPL table are: ", origCols,sep=""))
			quit()
		}
		
		map <- gplTable[,grepl(paste("^id$|^",idType,"$",sep=""), colnames(gplTable), ignore.case=TRUE)]
		
		toEG <- idTypeToOrgHsEg(idType)

		# Now that we have a mapper, go through and see what proportion actually successfully map.
               if(nrow(toEG) > 0){
			xMatch <- colnames(toEG)[2]
			yMatch <- colnames(map)[2]
                        merged <- merge(toEG, map, by.x=xMatch,by.y=yMatch)
			successfulMergePortion <- (nrow(merged) / nrow(map))
			print(paste("In converting from ", idType ," to Entrez Gene ID ", successfulMergePortion, " were successfully converted.", sep=""))
			
			if(successfulMergePortion < minIDTypeToEntrezPortion){
				# rerun, but remove this IDtype
				# the hacky way I did this was to just remove the current column I was working with, which didn't map a high enough portion
				# But to ensure that we still have all of the original column names I pass that along
				return(map2EntrezViaOrgHsEg(gplTable[,-colIndex],origCols=columnNames))
			}else{
	        	        return(merged[,c(3,2)])
			}
   		}else{
                        print(paste("Unable to load ", idType, " to Entrez mapping package: ", mapper, sep=""))
                        quit()
                }
}


# This function loops through the potential IDs you provide, in a ranked order (i.e. it will try to map the first entrry first)
# and looked for the same string in the columns provided
# It returns a list that contains 2 things:
# The idType that it settled on (i.e. what matched)
# And the index of that column with the match
findIDType <- function (potentialIDs, columnNames){

	for (i in 1:length(potentialIDs)){
		if(length(grep(potentialIDs[i],columnNames, ignore.case=T, value=F))>0){
				colIndex <- grep(potentialIDs[i],columnNames, ignore.case=T, value=F) 
				# this should return the column index of the ID matching one of our ID types
				return(list(ID=potentialIDs[i], columnIndex=colIndex))
			}
		}
	return(NULL) # return NULL if unable to finnd anything
}



#==================================================
# pull down data sets from geo and annotation files
#=================================================
if (grepl("GDS", geoID)) { # if GDS, simply pull down the data
    print("Parsing GDS")
    gds <- getGEO(geoID, destdir=softDir)
    gpl <- getGEO(Meta(gds)$platform, destdir=softDir) 
    data <- Table(gds)
} else if (grepl("GSE", geoID)) { # GSEs take a little extra processing
    print("Parsing GSE")
    gse <- getGEO(geoID, GSEMatrix=FALSE, destdir=softDir)
    platforms <- lapply(GSMList(gse),function(x) {Meta(x)$platform})
    if (length(unique(platforms)) != 1) {
        print ("No support for multiple platforms in GSE")
        quit(11)
    } else {
        gpl <- getGEO(as.character(unique(platforms)), destdir=softDir)
    }

    data <- Table(GSMList(gse)[[1]])["ID_REF"]
    samples <- length(GSMList(gse))
    for (i in 1:samples) {
        data <- cbind(data, Table(GSMList(gse)[[i]])["VALUE"])
    }   
    colnames(data)[2:(samples+1)] <- names(GSMList(gse))
} else { # We can only work with those 2 types
    print ("Unknown GEO data type. Exiting")
    quit(12)
} 

#================================================================
# create a table for mapping probe ids to entrez gene ids
#================================================================= 
# check out what we have
#colnames(Table(gpl))

altmapper <- FALSE
# create mappers for gpl files without entrez ids
# This is a one off for a particular array that can use this .db to map to entrez
if (grepl("HuGene-1_0-st", Meta(gpl)$title)) {
    library(hugene10sttranscriptcluster.db)
    e <- hugene10sttranscriptclusterENTREZID
    altmapper <- TRUE
}

if (altmapper) { # we loaded a library above, this is only if the above if went through the above if statement
    mapped_probes <- mappedkeys(e)
    eList <- as.list(e[mapped_probes])
    emap <- cbind(names(eList), eList)
    colnames(emap)[1] <- "ID"
    colnames(emap)[2] <- "ENTREZ_GENE_ID"
    probemap <- Table(gpl)["ID"]
    probemap <- merge(probemap, emap, by="ID")
    probemap[2] <- factor(unlist(probemap[2]))
} else { # map exists in gpl file, so we will use it
	# this is the easy case, Entrez is already used by the m.a.
	if(checkIfStringIsInColnames("entrez_gene", Table(gpl))){ 
                probemap <- Table(gpl)[,grepl("^id$|entrez_gene", colnames(Table(gpl)), ignore.case=TRUE)]
	# the first case where entrez is not used, genBank is, so we will map to Entrez, and then plug into what we already have
	}else{
		probemap <- map2EntrezViaOrgHsEg(Table(gpl))
	}
}

colnames(probemap)[2] <- "ENTREZ_GENE_ID"

#==================================================================================
# Now get to work on finding differential expression
#==================================================================================

# choose case and control from the database
q <- paste("select sample_id from ", metaTable, " where classification = \"",
    CONT, "\" and tissue_id =\"", tisID, "\"", sep="")
control.cols <- query(q)

q <- paste("select sample_id from ", metaTable, " where classification = \"",
    CASE, "\" and tissue_id =\"", tisID, "\"", sep="")
case.cols <- query(q)


# create the matrix to pass on to SAM
x <- cbind(data[,unlist(control.cols)], data[,unlist(case.cols)])
for (j in 1:ncol(x)) x[,j]<-as.numeric(as.character(x[,j]))
x <- as.matrix(x)

# write this matrix to a file for inspection/debugging
#write.table(x, file=sprintf("%s.dat", geoID), quote= FALSE,
#    sep="\t", row.names=FALSE)

# create map of case and control for SAM
#y <- c(rep(1,nrow(control.cols)),rep(2,nrow(case.cols)))
y <- c(rep(1,nrow(control.cols)),rep(2,nrow(case.cols)))
#print(y)
#print(colnames(x))
#print(dim(x))

# create data structure for SAM
# d <- list(x=x, y=y, geneid=as.character(data[["IDENTIFIER"]]),
#    genenames=as.character(data[["ID_REF"]]), logged2=1)
d <- list(x=x, y=y, geneid=as.character(data[["ID_REF"]]),
    genenames=as.character(data[["ID_REF"]]), logged2=1)

# run SAM
samr.obj <- samr(d, resp.type="Two class unpaired", assay.type="array",
    nperms=100, random.seed=seed)

delta.table <- samr.compute.delta.table(samr.obj)
#print(head(delta.table,3))

siggenes.table <- samr.compute.siggenes.table(samr.obj, delta, d,
    delta.table)

# pull out significantly expressed genes
genes.up <- as.data.frame(siggenes.table$genes.up)
genes.up$direction <- rep(1,nrow(genes.up))
genes.dn <- as.data.frame(siggenes.table$genes.lo)
genes.dn$direction <- rep(-1,nrow(genes.dn))

print(head(genes.up,1))
print(head(genes.dn,1))
print(tail(genes.up,1))
print(tail(genes.dn,1))

# filter out genes below q-value
genes.up.i <- as.numeric( as.character(genes.up$"q-value(%)")) < qmax
genes.dn.i <- as.numeric( as.character(genes.dn$"q-value(%)")) < qmax
genes.all <- rbind(genes.up[genes.up.i,], genes.dn[genes.dn.i,])

genes.all <- merge(genes.all, probemap, by.x="Gene ID", by.y="ID")
options(max.print=1000000)
print("probemap_begin")
print(genes.all[,c("Gene ID", "ENTREZ_GENE_ID", "Gene Name", "q-value(%)")])
print("probemap_end")
options(max.print=10000)

#=================================================================================
# Now convert these differential probes to genes, via Entrez to Uniprot
#=================================================================================
# convert probe set identifiers to ENTREZ gene id
# This is a table pulled down and separately processed that maps uniProt to Entrez
protmap <- read.delim(file="/tmp/HUMAN_9606_Protein_Entrez.tsv",header=FALSE,sep='\t')
colnames(protmap)[1] <- "UNIPROTKB"
colnames(protmap)[2] <- "ENTREZ_GENE"

# Quickly check how many of the differential probes we were actually able to get to UniProt 
geneNumberBeforeMerge<-nrow(genes.all)
genes.all <- merge(genes.all, protmap, by.x="ENTREZ_GENE_ID", by.y="ENTREZ_GENE")
genes.all <- unique(genes.all[,c("Gene Name", "UNIPROTKB", "q-value(%)", "direction")])
geneNumberAfterMerge<-nrow(genes.all)
print(paste(geneNumberAfterMerge/geneNumberBeforeMerge, " portion of significant probes successfully converted to UniProt.", sep=""))

# And die if it is below the fraction set above
if((geneNumberAfterMerge/geneNumberBeforeMerge) < minEnsemblToUniprotProportion){
	print("I'm losing too many genes in the Entrez to UniProt step")
	quit()
}


#===============================================================================
# insert results back into database
#===============================================================================
if (nrow(genes.all) > 0) {
    for (k in 1:nrow(genes.all)){
#        gid = genes.all[k,"Gene ID"]
        gid = genes.all[k,"Gene Name"]
        pid = genes.all[k,"UNIPROTKB"]
        qval = (100.0 - (as.numeric(as.character(genes.all[k,"q-value(%)"])))) / 100.0
        dir = genes.all[k,"direction"]
	print(genes.all[k,])
        q <- sprintf("insert into %s values (NULL, \"%s\", \"%s\", %.4f, %d, %d)",
            geoTable, pid, gid, qval, dir, tisID)
        print(q)
        junk <- query(q)
    }
} else {
    print(sprintf("%s has no significant expression", geoID)) 
}

