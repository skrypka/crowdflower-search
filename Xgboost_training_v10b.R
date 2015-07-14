# R script to train xgboost model and generate predictions on cv & test sets


# ****************************** Step 0 ******************************
# ******************* Loading libraries, data, etc. ******************

rm(list=ls())

library(caret)
library(xgboost)
library(Metrics)

# set working directory
# setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

# Choose modeling dataset
load("data/modelingset08b.Rdata") 

# Define custom objective function (kappa)
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds.rounded <- round(preds) #as.numeric(cut(preds, breaks=c(-Inf, 0.8, 1.8, 2.4, +Inf), labels=c(0,1,2,3)))
  err <- ScoreQuadraticWeightedKappa(as.numeric(labels),preds.rounded)
  # plot(as.numeric(labels), preds)
  return(list(metric = "kappa", value = err))
}



# ****************************** Step 1 *******************************
# *********************** Tuning by random search ********************* 

# skip and load params from tuning log file
load("tuning/xgboost_logfile10b.RData") # tuning parameters



# ****************************** Step 2 *******************************
# **************** Bagging & Generating cv predictions **************** 

# 5fold set
models <- 20
yhat <- rep(0, length(y))
keys <- read.csv(paste("data/cv_5fold_keys.csv", sep=""), header=TRUE)
for (k in 1:5) {
  tr <- keys$index[!keys$fold==k]
  vl <- keys$index[keys$fold==k]
  for (i in 1:models) {
    set.seed(1000*k + 100*i)
    xgboost.mod <- xgboost(objective = "reg:linear", data = X[tr,], label = y[tr], nround = logfile$rounds[i],
                           eta = logfile$shrinkage[i],  max.depth = logfile$depth[i], min.child.weight=logfile$min.child[i],
                           gamma=logfile$gamma[i], subsample=logfile$subsample[i], colsample.bytree= logfile$colsample.bytree[i],
                           nthread = 4,  verbose=0)
    yhat[vl] <- yhat[vl] +  predict(xgboost.mod, X[vl,]) 
  }
  yhat[vl] <- yhat[vl]/models
}
yhat.vl.rounded <- as.numeric(cut(yhat, breaks=c(-Inf, 0.9, 1.7, 2.5, Inf), labels=c(0,1,2,3)))-1
ScoreQuadraticWeightedKappa(y, yhat.vl.rounded)
write.csv(yhat+1, paste("models/xgboost_v10b_cv_5fold.csv", sep=""), row.names=FALSE)



# ****************************** Step 3 *******************************
# ** Building model on full train set & Generating test predictions *** 

load("tuning/xgboost_logfile10b.RData") # tuning parameters
load("data/modelingset08b.Rdata")
models <- 20
yhat.test  <- rep(0,nrow(X.test))
for (i in 1:min(nrow(logfile),models)){
  set.seed(100*i)
  xgboost.mod <- xgboost(data = X, label = y, max.depth = logfile$depth[i], eta = logfile$shrinkage[i],
                         nround = logfile$rounds[i], nthread = 4, objective = "reg:linear", subsample=logfile$subsample[i],
                         colsample_bytree=logfile$colsample.bytree[i], gamma=logfile$gamma[i], min.child.weight=logfile$min.child[i])
  yhat.test  <- yhat.test + predict(xgboost.mod, X.test)  
}
yhat.test <-  yhat.test/models 
yhat.test <- yhat.test +1
yhat.test.rounded <- as.numeric(cut(yhat.test, breaks=c(-Inf, 1.9, 2.7, 3.5, +Inf), labels=c(1,2,3,4)))
pred1 <- data.frame(id=id.test, prediction=yhat.test)
pred2 <- data.frame(id=id.test, prediction=yhat.test.rounded)
write.csv(pred1,"tuning/xgboost_v10b_test_unrounded.csv",row.names=F, quote=FALSE)