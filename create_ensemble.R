
rm(list=ls())

# set working directory
# setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

# rounding thresholds
master.round <- c(-Inf, 1.9, 2.7, 3.5, Inf)
int.round <- c(-Inf, 1.999, 2.999, 3.999, Inf)

id.test <- as.numeric(read.csv(paste( "C:/Data/Data Science/Kaggle/CrowdFlower/models/svm_sergio_alejandro/best_svm_Sergio_Alejandro.csv"), header=TRUE)[,1])

# 1st layer ann
p1g.test <- read.csv(paste( "models/ann_1234_7_ver2-test.csv", sep=""), header=FALSE)[,1]

# 2nd layer ann
e2c.test <- read.csv(paste( "models/ANN_2level_pred4-test.csv"), header=FALSE)[,1]

# svm classifiers
svm01 <- read.csv(paste( "models/submission_e.csv"), header=TRUE)[,2]
svm02 <- read.csv(paste( "models/SVM10b_final_test.csv"), header=FALSE)[,1]

# winning ensemble

# step 1
pp.test <- e2c.test*0.70 + p1g.test*0.30
pp.test.rounded <- as.numeric(cut(pp.test, breaks=master.round, labels=c(1,2,3,4)))  

# step 2
pp.test <- 0.8*pp.test.rounded  + 0.35*(svm01 + svm02)/2
pp.test.rounded <- as.numeric(cut(pp.test, breaks=int.round, labels=c(1,2,3,4)))  

write.csv(data.frame(id=id.test, prediction=pp.test.rounded),"models/winning_entry.csv",row.names=F, quote=FALSE)







