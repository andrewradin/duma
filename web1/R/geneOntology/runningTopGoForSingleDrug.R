# settings
nodeSize <- 5 # 10 is the default

library(topGO)
# This library does a lot more than what we are currently doing with it, but for our current means this simple script will be enough

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
    warning("runningTopGoForSingleDrug.R <csv: Uniprot genes to consider (column1) with pvalues (column2)> <max pvalue to consider significant> <annotation file (created via parseUniprotToGOTerms.sh)> <ontology type: BP, CC, or MP. Must agree with annotation file> <output file to write table to>")
    quit('no',10) # 10 is what we've been using as usage error in other R scripts
}
geneListFile <- args[1]
maxP <- as.numeric(args[2])
annotFile <- args[3]
ontologyType <- args[4]
outputFileHandle <- args[5]

# geneList should be a vector of q-values named by the uniprot they represent
if(grepl(".csv", geneListFile)){
    geneListData <- read.csv(geneListFile, header=FALSE)
    geneList <- as.numeric(as.character(geneListData[,2]))
    names(geneList) <- as.character(geneListData[,1])
}else if(grepl(".rds", geneListFile)){
   geneList <- readRDS(geneListFile)
}else{
    warning("Gene list file is not csv or rds. Those are the only acceptable formats.")
    quit('no',10) # 10 is what we've been using as usage error in other R scripts
}
# We also need to define a function to select significant genes from the list of all genes
# here I just take those that have a qval cut off that we've been using (we basically are working with 1- a normal qval)
topDiffGenes <- function(allScore) {return(allScore < maxP)}

geneID2GO <- readMappings(file = annotFile)

myGOData <- new("topGOdata", description = "Simple session", ontology = ontologyType, allGenes = geneList, geneSel = topDiffGenes, nodeSize = nodeSize, annot = annFUN.gene2GO, gene2GO = geneID2GO)

# We will use the most simple type of test statistics: Fisher's exact test which is based on gene counts

# *** The p-values computed by the runTest function are unadjusted for multiple testing *****

# First, we perform a classical enrichment analysis by testing the over-representation of GO terms within the group of differentially expressed genes. For the method classic each GO category is tested independently.
resultFisher <- runTest(myGOData, algorithm = "classic", statistic = "fisher")

# Next analyse and report results
totalNumberOfTerms <- length(myGOData@graph@nodes)

# print all of the terms
allRes <- GenTable(myGOData, classicFisher = resultFisher, orderBy = "classicFisher", topNodes = totalNumberOfTerms)

# And take care of multiple hypothesis testing
allRes$classicFisher <- p.adjust(as.numeric(gsub("<","",allRes$classicFisher)), method='fdr')

outputPrefix <- paste(outputFileHandle, ontologyType,"GOTerms",sep="_")
write.table(allRes, file=paste(outputPrefix, "_allWithFDR.tsv.txt",sep=""),sep="\t",quote=FALSE, row.names=FALSE)



