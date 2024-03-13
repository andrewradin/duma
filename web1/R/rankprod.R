RP <- function(data, x, y, num.perm = 100, gene.names = NULL, plot = FALSE, rand = NULL){
    library(matrixStats)
    cat("Using minimal version of RP\n");
    num.gene <- nrow(data)
    data <- as.matrix(data)
    mode(data) <- "numeric"

    ##set seed for random number generator
    if (!is.null(rand)) set.seed(rand)

    data1 <- as.matrix(data[, x])
    data2 <- as.matrix(data[, y])

    data1.ave <- rowMeans(data1)
    data2.ave <- rowMeans(data2)

    fold.change <- data1.ave - data2.ave
    exponent <- 1/(ncol(data1)*ncol(data2))

    ##original rank product and rank of each rank product
    #study up-regulated genes under condition2
    RP.ori.upin2 <- RankProd1v2(data1, data2, exponent)$RP
    rank.ori.upin2 <- rank(RP.ori.upin2)
    ##study down-regulated genes under condition2
    RP.ori.downin2 <- RankProd1v2(data2, data1, exponent)$RP
    rank.ori.downin2 <- rank(RP.ori.downin2)

    cat("Starting", num.perm, "permutations...", "\n")
   
    ## fb: Ranks are now computed iteratively, i.e. updated with each new row of permuted data. Initialization is done here.
    temp2.up <- rank(RP.ori.upin2)  ##rank.ori.upin2
    temp2.down <- rank(RP.ori.downin2)  ##rank.ori.downin2

    for (p in 1:num.perm) {
        temp.data <- Newdata(data1, data2)
        temp2.up <- updateRank(temp2.up
                               , RP.ori.upin2
                               , RankProd1v2(temp.data$new.data1, temp.data$new.data2, exponent)$RP
                               )
        temp2.down <- updateRank(temp2.down
                                 , RP.ori.downin2
                                 , RankProd1v2(temp.data$new.data2, temp.data$new.data1, exponent)$RP
                                )
    }
    rm(temp.data)

    ##address significance level
    cat("Computing pfp ..", "\n")

    ##for up-regulation under class2,data1<data2, two-sample
    
    temp2 <- temp2.up ##the rank of original RP in the increasing order
    
    order.temp <- match(temp2, sort(temp2))
    count.perm <- (sort(temp2) - c(1:num.gene))[order.temp]
    exp.count <- count.perm/num.perm
    pval.upin2 <- count.perm/(num.perm * num.gene)
    pfp.upin2 <- exp.count/rank.ori.upin2

    if (plot) {
        par(mfrow = c(2, 1))
        plot(rank.ori.upin2, pfp.upin2, xlab = "sorted gene rank of the original rank product",
            ylab = "estimated PFP")
        title("Identification of Up-regulated genes under class 2")
    }

    ##for down-regulation under class2, data1> data2, two-sample
   
    temp2 <- temp2.down

    order.temp <- match(temp2, sort(temp2))
    count.perm <- (sort(temp2) - c(1:num.gene))[order.temp]
    exp.count <- count.perm/num.perm
    pval.downin2 <- count.perm/(num.perm * num.gene)
    pfp.downin2 <- exp.count/rank.ori.downin2

    if (plot) {
        plot(rank.ori.downin2, pfp.downin2, xlab = "sorted gene rank of the original rank product",
            ylab = "estimated PFP")
        title("Identification of down-regulated genes under class 2")
    }
    rm( temp2, order.temp, count.perm, exp.count)

    ##output the estimated pfp and ranks of all genes
    cat("Outputing the results ..", "\n")

    pfp <- data.frame(pfp.upin2, pfp.downin2)
    pval <- data.frame(pval.upin2, pval.downin2)
    RPs <- data.frame(RP.ori.upin2, RP.ori.downin2)
    RPrank <- data.frame(rank.ori.upin2, rank.ori.downin2)
    colnames(RPrank) <- colnames(RPs) <- colnames(pfp) <- colnames(pval) <- c("class1 < class2", "class1 > class 2")

    fold.change <- t(t(fold.change))
    colnames(fold.change) <- "log/unlog(class1/class2)"

    if (!is.null(gene.names)) {
        rownames(pfp) <- rownames(pval) <- rownames(RPs) <- rownames(RPrank) <- rownames(fold.change) <- gene.names
    }
    list(pfp = pfp, pval = pval, RPs = RPs, RPrank = RPrank, AveFC = fold.change)
}

RankProd1v2 <- function(data1,data2, exponent){
  ##version 2: For each Fold-change, get rank, then rankprod, only keep the rankprod updated from each Fold-change
  ##compute all possible pairwise comparison for two sample
  rank.prod <- rep(1, nrow(data1))
  for( k in 1:ncol(data1)) rank.prod <- rank.prod * rowProds(t(colRanks((data1[,k] - data2), ties.method = 'average')) ^ exponent)
  list(RP=rank.prod)
}

Newdata <- function(data1,data2){
  k1 <- ncol(data1)
  k2 <- ncol(data2)
  num.gene <- nrow(data1)
  new.data1 <- matrix(NA,num.gene,k1)
  new.data2 <- matrix(NA,num.gene,k2)
  for (k_1 in 1:k1){
    new.data1[,k_1] <- data1[sample.int(num.gene),k_1]
  }
  for (k_2 in 1:k2){
    new.data2[,k_2] <- data2[sample.int(num.gene),k_2]
  }
  return(list(new.data1 = new.data1,new.data2 = new.data2))
}

## fb: This function does most of the hard work - computing matrix ranks iteratively column by column
updateRank <- function( oldRank, col1, nextCol ) {
  ord <- order(col1)
  col1 <- col1[ord]
  oldRank <- oldRank[ord]

  nextCol <- sort.int(nextCol,method = 'quick')

  i <- length(col1)
  j <- length(nextCol)
  while(i>0 && j>0) {
    while(j>0 && nextCol[j]>=col1[i]) j<-j-1 # go up nextCol until the value is less than col1[i]
    k <- i
    while(i>0 && j>0 && nextCol[j]<col1[i]) i<-i-1 # go up col1 until the value is less than or equal to nextCol[j]
    oldRank[(i+1):k] <- oldRank[(i+1):k] + j
  }
  return(oldRank[order(ord)])
}

topGene <- function(x,cutoff=NULL,method="pfp",num.gene=NULL,logbase=2,gene.names=NULL){  
   ##input is x: an RP object
   pfp <- as.matrix(x$pfp)
   FC <- as.matrix(x$AveFC)  ##data1/ data2
   pval <- as.matrix(x$pval)
   
   if (is.null(x$RPs) ){  ##Rank Sum
      RP <- as.matrix(x$RSs)
      rank <- as.matrix(x$RSrank)
    } else {
      RP <- as.matrix(x$RPs)
      rank <- as.matrix(x$RPrank)
    }  
 
   if (is.null(num.gene) & is.null(cutoff)) 
       stop("No selection criteria is input, please input either cutoff or num.gene")
   
   ##for up-regulation under class2,data1<data2, two-sample
   ## under denominator class, one-sample 
   RP.sort.upin2 <- sort(RP[,1],index.return=TRUE)
   RP.sort.downin2 <- sort(RP[,2],index.return=TRUE)

   if (!is.null(cutoff) ) {

      if (method == "pfp") {
      cutgenes.upin2<-which(pfp[RP.sort.upin2$ix,1]<cutoff)
      cutgenes.downin2<-which(pfp[RP.sort.downin2$ix,2]<cutoff)
        } else {
          if (method == "pval") {
            cutgenes.upin2<-which(pval[RP.sort.upin2$ix,1]<cutoff)
            cutgenes.downin2<-which(pval[RP.sort.downin2$ix,2]<cutoff)
          } else {
          stop("No criterion is input to select genes, please select either pfp(fdr) or pval(P-value)")
          }
        }


      if (length(cutgenes.upin2)>0) {
              numTop<-max(cutgenes.upin2)  
              gene.sel.upin2<-RP.sort.upin2$ix[1:numTop]  
              rm(numTop)
        }else {
        gene.sel.upin2<-c()
       }
         
      if (length(cutgenes.downin2)>0) {
              numTop<-max(cutgenes.downin2)  
              gene.sel.downin2<-RP.sort.downin2$ix[1:numTop]  
              rm(numTop)
        }else {
        gene.sel.downin2<-c()
       }

   }
       
   if (is.null(cutoff) & !is.null(num.gene)) {     
     if (num.gene>0) {
        gene.sel.upin2<-RP.sort.upin2$ix[1:num.gene] 
                gene.sel.downin2<-RP.sort.downin2$ix[1:num.gene] 

    } else {
        gene.sel.upin2<-c()
                gene.sel.downin2<-c()
    }
   }

   if (!is.null(gene.names)) {
         if (dim(pfp)[1]!=length(gene.names) ){
         cat("Warning: gene.names should have the same length as the gene vector.","\n")
         cat("No gene.names are assigned","\n") 
         } else { 
         rownames(pfp)<-gene.names
         #cat("gene.names are assigned","\n")
         } 
   }


   pfp<-round(pfp,4)
   pval<-round(pval,4)
   RP<-round(RP,4)
   FC<-round(logbase^FC,4)


   ##for up-regulation under class2,data1<data2, two-sample
   ## under denominator class, one-sample 
   
   if(length(gene.sel.upin2)>0) {
     Out.table.upin2<-cbind(gene.sel.upin2,RP[gene.sel.upin2,1],FC[gene.sel.upin2],
                   pfp[gene.sel.upin2,1],pval[gene.sel.upin2,1])
     rownames(Out.table.upin2)<-rownames(pfp)[gene.sel.upin2]
     colnames(Out.table.upin2)<-c("gene.index","RP/Rsum","FC:(class1/class2)","pfp","P.value")  

     cat("Table1: Genes called significant under class1 < class2","\n\n")
    } else { 
     cat("No genes called significant under class1 < class2","\n\n") 
     Out.table.upin2<-NULL
    }

   ##for down-regulation under class2, data1> data2, two-sample
   ## under numeratorclass, one-sample 
   
   if(length(gene.sel.downin2)>0) {
     Out.table.downin2<-cbind(gene.sel.downin2,RP[gene.sel.downin2,2],FC[gene.sel.downin2],
                   pfp[gene.sel.downin2,2],pval[gene.sel.downin2,2])
     rownames(Out.table.downin2)<-rownames(pfp)[gene.sel.downin2]
     colnames(Out.table.downin2)<-c("gene.index","RP/Rsum","FC:(class1/class2)","pfp","P.value")  

     cat("Table2: Genes called significant under class1 > class2","\n\n")  
    } else { 
      cat("No genes called significant under class1 > class2","\n\n") 
      Out.table.downin2<-NULL
    }


   list(Table1=Out.table.upin2,Table2=Out.table.downin2)
    
}


