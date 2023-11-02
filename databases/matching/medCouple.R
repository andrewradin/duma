library(robustbase) #for calculation of medcouple coefficient (using mc())

args <- commandArgs(trailingOnly = TRUE)

input <- read.table(args[1],header=F,stringsAsFactors=F)

result <- mc(as.numeric(input[,1]))

result
