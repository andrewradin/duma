evalModel <- function(knowns, predictions, listToSaveTo){
    # use the ROCR package to save all of the quality metrics for each set of models
    labels <- as.character(knowns)
    labels[which(labels=="2:False")] = 0
    labels[which(labels=="1:True")] = 1
    labels <- as.numeric(labels)
    pred <- prediction(predictions, labels )
    # Append these results to the current in listToSaveTo
    i=length(listToSaveTo$roc)+1
    listToSaveTo$roc[[i]] <-performance(pred, measure='tpr',x.measure='fpr')
    listToSaveTo$auc[[i]] <- performance(pred, measure='auc')
    listToSaveTo$precRecall[[i]] <- performance(pred, measure='prec',x.measure='rec')
    
    return(listToSaveTo)
}

plotAllOfTheMetrics <- function(everything, colors, lwds){
    par(family='serif')
    png('roc.png')
    plot(everything$roc[[1]],
       col=colors[1],
       lwd=lwds[1],
       main='Receiver Operating Characteristic',
       xlim=c(0,1),
       ylim=c(0,1)
    )
    if( length( everything$roc ) > 1 ){
        for (iter in 2:length(everything$roc)){
            plot(everything$roc[[iter]], col=colors[iter], lwd=lwds[iter], add=TRUE)
        }
    }
    abline(0,1)
    legend('bottomright',
           c(seq(1,length(everything$roc)-1), "Final"),
           col=colors, lwd=2)
    legend(0.7, 0.3,
           paste("Final AUC: ",round(everything$auc[[length(everything$auc)]]@y.values[[1]],digits=4),sep=""),
           bty='n')
    dev.off()


    png('precRecall.png')
    plot(everything$precRecall[[1]],
         col=colors[1],
         lwd=lwds[1],
         main='Precision/Recall',
         xlim=c(0,1),
         ylim=c(0,1)
    )
    if( length( everything$precRecall ) > 1 ){
        for (iter in 2:length(everything$precRecall)){
            plot(everything$precRecall[[iter]], col=colors[iter], lwd=lwds[iter], add=TRUE)
        }
    }
    legend('bottomleft',
           c(seq(1,length(everything$roc)-1), "Final"),
           col=colors, lwd=2
    )

    dev.off()

}


plotRankings <- function(listOfAll){
    lengthOfList <- length(listOfAll)
    # probability of association ranks
    png("probabilityOfAssociation.png", height=lengthOfList*480, res=120)
    par(mfrow=(c(lengthOfList,1)))
    for (i in 1:lengthOfList){
        currentToUse <- listOfAll[[i]]
        if(i==lengthOfList){
            mainTitle <- "Final"
        }else{
            mainTitle <- i
        }
        plot(currentToUse$orderedByProb, pch=16, ylab='Association Probability', xlab='Drug Rank',col='blue', main=mainTitle)
        points(currentToUse$allTreatX,currentToUse$allTreatY, col='red3', pch=16)
        if(! all(is.na(currentToUse$testTreatX))){
            points(currentToUse$testTreatX,currentToUse$testTreatY, col='gold', pch=16)
            legend('topright',c("Known - training", "Known - testing", "Discovered"), pch=16,col=c('red3','gold', 'blue'))
        }else{
            legend('topright',c("Known", "Discovered"), pch=16,col=c('red3', 'blue'))
        }
    }
    dev.off()
    
    # probability of association ranks - top 100
    png("probabilityOfAssociation_top100.png", height=lengthOfList*480, res=120)
    par(mfrow=(c(lengthOfList,1)))
    for (i in 1:lengthOfList){
        currentToUse <- listOfAll[[i]]
        if(i==lengthOfList){
            mainTitle <- "Final"
        }else{
            mainTitle <- i
        }
        plot(currentToUse$orderedByProb[1:100], pch=16, ylab='Association Probability', xlab='Drug Rank',col='blue', main=mainTitle)
        points(currentToUse$allTreatX,currentToUse$allTreatY, col='red3', pch=16)
        if(! all(is.na(currentToUse$testTreatX))){
            points(currentToUse$testTreatX,currentToUse$testTreatY, col='gold', pch=16)
            legend('topright',c("Known - training", "Known - testing", "Discovered"), pch=16,col=c('red3','gold', 'blue'))
        }else{
            legend('topright',c("Known", "Discovered"), pch=16,col=c('red3', 'blue'))
        }
    }
    dev.off()
}
