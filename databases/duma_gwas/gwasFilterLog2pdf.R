#!/usr/bin/env Rscript
library(dplyr)
library(tidyr)
library(gplots)


args = commandArgs(trailingOnly=TRUE)
#args=c('C:/Users/Nikki/Downloads/filter.log', 'C:/Users/Nikki/Downloads/filter.pdf')

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("Usage: Rscript gwasFilterLog2pdf.R filter.log filter.pdf studies.tsv snps.tsv.gz", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = "filter.pdf"
}


filter <- read.delim(args[1], header=FALSE)
#keep <- read.delim(args[2], header=FALSE)

pdf(args[2], width=15, height=10)

#count how many snps and studies made it through
sysret<-system(sprintf('wc -l %s',args[3]), intern=TRUE)
nkeptstudies<-strsplit(sysret," ")[[1]][1]

sysret<-system(sprintf('zcat %s | wc -l',args[4]), intern=TRUE)
nkeptsnps<-strsplit(sysret," ")[[1]][1]

# histogram of snps lost per study, header of total number of studies lost
slps<-filter %>% count(V1)
mh<-hist(log10(slps$n),breaks=c(-1,0,1,2,3,4,5,6,7,8,9), main="Frequency of log of snps lost per phenotype|pmid", sub=sprintf("%d snps were lost from %d phenotype|pmid",length(filter$V1),length(slps$n)))

# histogram of number of snps lost per pmid and total number
splitfilter<-separate(filter, V1, c("pheno","pmid"), sep = "\\|")
slpp<-splitfilter %>% count(pmid)
mh<-hist(log10(slpp$n),breaks=c(-1,0,1,2,3,4,5,6,7,8,9), main="Frequency of log of snps lost per pmid", sub=sprintf("%d snps were lost from %d pmids. %s snps from %s studies passed filters",length(filter$V1),length(slpp$n), nkeptsnps, nkeptstudies))


#barplot of number of things snagged by filter in order of filter
#order that things are being filtered:
#  check position:
#  size-position check (is the number larger than the chromosome length)
#int-position check (is the position actually a number?)
#chr-position check (is the chromosome appropriately named)
#check snp stats:
#  check p:
#  p-value check (is the p-value actually a number)
#check maf:
#  allele check (actg)
#minMAF check (is the minor allele frequency above some number, currently 0, good to see that hasn’t caught anything)
#minMAC check (is the maf*n/2>25, minimum sample size)
#format-MAF check (happens if anything else fails, seems to happen when the allele info is blank)

filterorder<-c("size-position", "int-position", "chr-position", "p-value", "allele", "minMAF", "minMAC", "format-MAF")
nfilter<-filter %>% count(V8)
nfilter$V8 <- gsub("failed the ", "", nfilter$V8)
nfilter$V8 <- gsub(":.*", "", nfilter$V8)
nfilter$V8 <- as.factor(gsub(" check", "", nfilter$V8))
nfilter<-nfilter[match(filterorder, nfilter$V8),]
nfilter$V8<-filterorder

barplot(nfilter$n,names.arg=nfilter$V8,main="Number of SNPS removed by each filter, steps in order", sub=sprintf("%s captured %d snps. %s snps passed all filters", nfilter[which(nfilter$n==max(nfilter$n,na.rm=TRUE)),]$V8, max(nfilter$n,na.rm=TRUE), nkeptsnps))

minmac<-filter[which(filter$V8=="failed the minMAC check:"),]
minmac<-separate(minmac, V6, c("allele","maf"), sep = ";")
minmac$V9 <- as.numeric(as.character(minmac$V9))
minmac$maf <- as.numeric(as.character(minmac$maf))

#what the minmac filter actually catches
plot(log10(minmac$V9),log10(minmac$maf), ylab="log of minor allele frequency", xlab = "log of nsamples", main="SNPs with too few minor allele counts", type = "p")
hist((minmac$V9*minmac$maf)/2,main="How many snps you would rescue by increasing the minMAC filter")

#studies
#on a per study basis, how often do things get flagged by multiple filters?
fc<-filter %>% count(V8,V1) %>% count(V1)
hist(fc$n,breaks=c(0,1,2,3,4,5,6,7,8),main="How many filters each pheno|pmid gets snagged by",xlab="number of filters")

nfilter<-filter %>% count(V8,V1) %>% count(V8)
nfilter$V8 <- gsub("failed the ", "", nfilter$V8)
nfilter$V8 <- gsub(":.*", "", nfilter$V8)
nfilter$V8 <- as.factor(gsub(" check", "", nfilter$V8))
nfilter<-nfilter[match(filterorder, nfilter$V8),]
nfilter$V8<-filterorder

barplot(nfilter$n,names.arg=nfilter$V8,main="Number of different studies removed by each filter, steps in order", sub=sprintf("%s captured %d studies", nfilter[which(nfilter$n==max(nfilter$n,na.rm=TRUE)),]$V8, max(nfilter$n,na.rm=TRUE)))

#and what is that particular line we keep losing

textplot(head(filter[which(grepl(nfilter[which(nfilter$n==max(nfilter$n,na.rm=TRUE)),]$V8,filter$V8)),]))

dev.off()


