rm(list=ls())

library(Metrics)


# set working directory
setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

# Choose modeling dataset (just to get labels y)
load("data/modelingset08.Rdata")

keys <- read.csv("data/cv_5fold_keys.csv")

# *********************** madcap *************************************

m1 <- read.csv("models/xgboost_v10_cv_5fold.csv", header=T)[,1]
m1.test <- read.csv("models/xgboost_v10_test_unrounded.csv", header=T)[,2]

m2 <- read.csv("models/xgboost_v10b_cv_5fold.csv", header=T)[,1]
m2.test <- read.csv("models/xgboost_v10b_test_unrounded.csv", header=T)[,2]

m3 <- read.csv("models/xgboost_v10c_cv_5fold.csv", header=T)[,1]
m3.test <- read.csv("models/xgboost_v10c_test_unrounded.csv", header=T)[,2]

m4 <- read.csv("models/xgboost_v11_cv_5fold.csv", header=T)[,1]
m4.test <- read.csv("models/xgboost_v11_test_unrounded.csv", header=T)[,2]

m5 <- read.csv("models/xgboost_v12_cv_5fold.csv", header=T)[,1]
m5.test <- read.csv("models/xgboost_v12_test_unrounded.csv", header=T)[,2]

m6 <- read.csv("models/H2O.gbm_20150616_cv_5fold.csv", header=T)[,1]
m6.test <- read.csv("models/H2O.GBM_20150616_unrounded.csv", header=T)[,2]

m7 <- read.csv("models/svm.reg_20150703_cv_5fold.csv", header=T)[,1]
m7.test <- read.csv("models/svm.reg_20150703_unrounded.csv", header=T)[,2]


# *********************** roman *************************************

m8 <- rep(0, length(y))
for (k in 1:5) { m8[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann_alt_ngram_wm-", k, ".csv", sep=""), header=F)[,1]}
m8.test <- read.csv("models/ann_alt_ngram_wm-test.csv", header=F)[,1]

m9 <- rep(0, length(y))
for (k in 1:5) { m9[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann_wm_c1r2-", k, ".csv", sep=""), header=F)[,1]}
m9.test <- read.csv("models/ann_wm_c1r2-test.csv", header=F)[,1]

m10 <- rep(0, length(y))
for (k in 1:5) { m10[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann_alt-", k, ".csv", sep=""), header=F)[,1]}
m10.test <- read.csv("models/ann_alt-test.csv", header=F)[,1]

m11 <- rep(0, length(y))
for (k in 1:5) { m11[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ANN10b-", k, ".csv", sep=""), header=F)[,1]}
m11.test <- read.csv("models/ANN10b-test.csv", header=F)[,1]

m12 <- rep(0, length(y))
for (k in 1:5) { m12[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann_1234_7_ver2-", k, ".csv", sep=""), header=F)[,1]}
m12.test <- read.csv("models/ann_1234_7_ver2-test.csv", header=F)[,1]

m13 <- rep(0, length(y))
for (k in 1:5) { m13[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann_250_tfidf-", k, ".csv", sep=""), header=F)[,1]}
m13.test <- read.csv("models/ann_250_tfidf-test.csv", header=F)[,1]

m14 <- rep(0, length(y))
for (k in 1:5) { m14[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann_tfidf-", k, ".csv", sep=""), header=F)[,1]}
m14.test <- read.csv("models/ann_tfidf-test.csv", header=F)[,1]

m15 <- rep(0, length(y))
for (k in 1:5) { m15[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann10b_ver2-", k, ".csv", sep=""), header=F)[,1]}
m15.test <- read.csv("models/ann10b_ver2-test.csv", header=F)[,1]

m16 <- rep(0, length(y))
for (k in 1:5) { m16[keys[keys[,1]==k,2]] <-  read.csv(paste("models/ann10b_noamazon-", k, ".csv", sep=""), header=F)[,1]}
m16.test <- read.csv("models/ANN10b_noamazon-test.csv", header=F)[,1]


m17 <- rep(0, length(y))
for (k in 1:5) { m17[keys[keys[,1]==k,2]] <-  read.csv(paste("models/KNN-5-bagging-", k, ".csv", sep=""), header=F)[,1]}
m17.test <- read.csv("models/KNN-5-bagging-test.csv", header=F)[,1]


m18 <- rep(0, length(y))
for (k in 1:5) { m18[keys[keys[,1]==k,2]] <-  read.csv(paste("models/sub_d_cv_", k, ".csv", sep=""), header=T)[,1]}
m18.test <- read.csv("models/submission_e.csv", header=F)[,2]


m19 <- rep(0, length(y))
for (k in 1:5) { m19[keys[keys[,1]==k,2]] <-  read.csv(paste("models/SVM10b_final_", k, ".csv", sep=""), header=F)[,1]}
m19.test <- read.csv("models/SVM10b_final_test.csv", header=F)[,1]


m20 <- rep(0, length(y))
for (k in 1:5) { m20[keys[keys[,1]==k,2]] <-  read.csv(paste("models/RF5b_final-", k, ".csv", sep=""), header=F)[,1]}
m20.test <- read.csv("models/RF5b_final-test.csv", header=F)[,1]



#cor(cbind(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16))
#cor(cbind(m1.test,m2.test,m3.test,m4.test,m5.test,m6.test,m7.test,m8.test,m9.test,m10.test,m11.test,m12.test,m13.test,m14.test,m15.test,m16.test))


# *********************** sergio/alejandro *************************************

# mstr3 <- read.csv("models\\stacking\\stacking_master_train_V3.csv", header=TRUE)
# mstr3.test <- read.csv("models\\stacking\\stacking_master_test_V3.csv", header=TRUE)
# mstr3 <- mstr3[,c("id", "KNN_bagging_5_withOUT.amazon", "SVM.Alejandro", "SVM.Sergio", "RF_bagging_5_without.amazon", "svm_bagging_10_wo_replacement_wo_amazon")]
# mstr3.test <- mstr3.test[,c("id", "KNN_bagging_5_withOUT.amazon", "SVM_Alejandro", "SVM_Sergio", "RF_bagging_5_withOUT.amazon", "svm_bagging_10_wo_replacement_wo_amazon")]


mstr4 <- cbind(m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15, m16,m17, m18, m19, m20 )

mstr4.test <- cbind(m1.test ,m2.test ,m3.test ,m4.test ,m5.test ,m6.test ,m7.test ,m8.test ,
                              m9.test ,m10.test ,m11.test ,m12.test ,m13.test ,m14.test,m15.test,m16.test,
                              m17.test, m18.test, m19.test, m20.test)

write.csv(mstr4, "input/stacking_master_train_V4_rerun.csv")
write.csv(mstr4.test, "input/stacking_master_test_V4_rerun.csv")
