#setwd('../Desktop//padre')
#library(scales)
all_atrs_selected <- read.table('_attrs_selected.tsv', sep="\t", stringsAsFactors=F, header=F, comment.char="")
all_atrs <- read.table('plat761_all_atr_adr_portions.tsv', sep="\t", stringsAsFactors=F, header=F, comment.char="")
together <- merge(all_atrs, all_atrs_selected, by=1, all.x = T)
together[,2] <- as.numeric(as.character(together[,2]))
together[,3] <- as.numeric(as.character(together[,3]))
together_nonas <- together
together_nonas[is.na(together_nonas)] <- 0
par(mar=c(4.1, 6.1, 2.1, 2.1))
plot(together_nonas[,2:3], pch=20 
     , xlab = 'Portion of ADRs attribute is associated with\nNAPA identified attributes'
     , ylab = 'Weka selected attributes\nPortion of ADR-predicting models attribute was selected for')
cor(together_nonas[,2], together_nonas[,3])

hist(together_nonas[,3], breaks=100)
quantile(together_nonas[,2], 0.05)

barplot(together_nonas[order(together_nonas[,3], decreasing=T)[1:10],3]
        , names.arg = together_nonas[order(together_nonas[,3], decreasing=T)[1:10],1]
        )


sum_atrs_selected <- read.table('summaryStats_attrs_selected.tsv', sep="\t", stringsAsFactors=F, header=F, comment.char="")
barplot(sum_atrs_selected[order(sum_atrs_selected[,2], decreasing=T),2],
         , xlab = 'Weka selected attributes\nPortion of ADR-predicting models attribute was selected for'
         , xlim=c(0,1)
         , horiz=T)
