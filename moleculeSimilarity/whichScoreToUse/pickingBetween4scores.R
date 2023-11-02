setwd("C:/Users/Aaron/Desktop/2xar/")
randomPartialAll4Scores <- read.table("RandomPartial_drugbankTani.tsv", header=F)
colnames(randomPartialAll4Scores)<-c('drug1','drug2','score1','score2','score3','score4')

formatted<-as.data.frame(matrix(ncol=4, nrow=nrow(randomPartialAll4Scores)))
for (i in 3:6){
  formatted[,(i-2)] <- as.numeric(as.vector(randomPartialAll4Scores[,i]))
}
rownames(formatted)<-paste(randomPartialAll4Scores[,1], randomPartialAll4Scores[,2], sep='_')

no.nas<-na.omit(formatted)

plot(density(no.nas[,1]))
plot(density(no.nas[,2]))
plot(density(no.nas[,3]))
plot(density(no.nas[,4]))

correlations<-matrix(nrow=4, ncol=4)
variance<-vector(mode='numeric', length=4)
for (i in 1:4){
  for (j in 1:4){
    correlations[i,j]<-cor(no.nas[,i], no.nas[,j])
  }
  variance[i]<-var(no.nas[,i])
}

plot(no.nas[,1:4])

summary(no.nas)
pheatmap(correlations)

plot(variance)


# Summary
# 1 and 2, and 3 and 4 are all but identical, and all 4 are very correlated.
# I'm going with 3 because it has the lowest variance, and thus ought to be able to more easily identify those that are the same
# The cutoff I'm going with is the 99th percentile of the 3rd score:
quantile(no.nas[,3], 0.99)
# 0.5500276 


