library(org.Hs.eg.db)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  warning("uniprotToGO.R <output file to write table to>")
  quit('no',80) # why 80? Because that's where my fingers fell
}
outputFile <- args[1]

# Create an annotation file for Uniprot IDs to GO terms for topGO
# First thing we do is go back to Entrez gene IDs (it might make some sense to do our condensing and such at the level of Entrez IDs, then we wouldn't have to back map)
entrezToUniprot <- as.data.frame(org.Hs.egUNIPROT)
# We then go from entrez to GO terms
entrezToGO <- as.data.frame(org.Hs.egGO)
# We can then combine those and have uniprot to go via entrez
uniprotToGo <- unique(merge(entrezToUniprot, entrezToGO, by="gene_id"))

# Now simply store these in a flat file
write.table(uniprotToGo, file=outputFile, sep="\t", col.names=FALSE, row.names=FALSE, quote=FALSE)

