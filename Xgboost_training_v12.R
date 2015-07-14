# R script to train xgboost model and generate predictions on cv & test sets


# ****************************** Step 0 ******************************
# ******************* Loading libraries, data, etc. ******************

rm(list=ls())

library(caret)
library(xgboost)
library(Metrics)

# set working directory
setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

# Choose modeling dataset
load("data/modelingset10.Rdata") 


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

load("data/xgboost_logfile12.RData")

if (FALSE) {
    for (i in (nrow(logfile)+1):50) {
        n <- sample(c(40, 60, 80, 100, 120, 140),1)
        d <- sample(5:20,1)
        g <- sample(seq(0.1,2,0.2),1)
        m <- sample(seq(0,5,0.2),1)
        s <- 1 # sample(seq(0.5,1,0.05),1)
        cs <-1 # sample(seq(0.5,1,0.05),1)
        et <- sample(c(0.1),1)
        err <- rep(0,15)
        for (st in 1:5) {
            keys <- read.csv(paste("data/set", st, "_3foldcv_keys.csv", sep=""), header=TRUE)[,1]
            for (k in 1:3) {
              X.tr <- X[keys==k,]
              X.vl <- X[!keys==k,]
              y.tr <- y[keys==k]
              y.vl <- y[!keys==k]
              xgboost.mod <- xgboost(objective = "reg:linear", data = X.tr, label = y.tr, nround = n , eta = et, 
                                     max.depth = d, min.child.weight=m, gamma=g, subsample=s, colsample.bytree= cs,
                                      nthread = 4,  verbose=0)
              yhat.vl <- predict(xgboost.mod, X.vl)  
              err[(st-1)*3+k] <- ScoreQuadraticWeightedKappa(y.vl, round(yhat.vl))
              write.csv(yhat.vl, paste("data/run", i, "_set", st, "_fold", k, ".csv", sep=""), row.names=FALSE)  
              print(err[(st-1)*3+k])
              flush.console()
            } 
        }
        logfile <- rbind(logfile, data.frame(run=i, shrinkage=et, rounds=n, depth=d, gamma=g, min.child=m, subsample=s, 
                                             colsample.bytree= cs, err.val= mean(err), err.sd.val=sd(err), note="v08 all features") )
        save(list=c("logfile"), file="data/xgboost_logfile12.RData")
    }
    logfile <- logfile[order(-logfile$err.val),]
    save(list=c("logfile"), file="data/xgboost_logfile12.RData")
}


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
write.csv(yhat+1, paste("models/xgboost_v12_cv_5fold.csv", sep=""), row.names=FALSE)


# ****************************** Step 3 *******************************
# ** Building model on full train set & Generating test predictions *** 

load("data/xgboost_logfile12.RData")
load("data/modelingset10.Rdata")
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
write.csv(pred1,"models/xgboost_v12_test_unrounded.csv",row.names=F, quote=FALSE)
# write.csv(pred2,"models\\xgboost\\v12\\sub\\xgboost_v12_test_rounded.csv",row.names=F, quote=FALSE)
