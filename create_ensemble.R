
rm(list=ls())

# set working directory
setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

# rounding thresholds
master.round <- c(-Inf, 1.9, 2.7, 3.5, Inf)
int.round <- c(-Inf, 1.999, 2.999, 3.999, Inf)

id.test <- as.numeric(read.csv(paste( "data/best_svm_Sergio_Alejandro.csv"), header=TRUE)[,1])

# test predictions
p1g.test <- read.csv(paste( "data/ANN10b-wm-250r2-union-qtd-250tandd-psim-prod1234-test.csv", sep=""), header=FALSE)[,1]
e1.test <- as.numeric(read.csv(paste( "data/best_svm_Sergio_Alejandro.csv"), header=TRUE)[,2])
e2c.test <- as.numeric(read.csv(paste( "data/ANN10b-wm-250r2-union-qtd-250tandd-psim-pred4-test.csv"), header=FALSE)[,1])

# second ensemble with the nn 2nd layer
pp.test <- e2c.test*0.70 + p1g.test*0.30
pp.test.rounded <- as.numeric(cut(pp.test, breaks=master.round, labels=c(1,2,3,4)))  

pp.test <- 0.8*pp.test.rounded  + 0.35*e1.test
pp.test.rounded <- as.numeric(cut(pp.test, breaks=int.round, labels=c(1,2,3,4)))  

write.csv(data.frame(id=id.test, prediction=pp.test.rounded),"data/winning_entry.csv",row.names=F, quote=FALSE)







