rm(list=ls())

######################################################################
## Step 1 - Import Data and create Train/Validation Splits
######################################################################

# set working directory
setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

library(caret)
library(Metrics)
library(h2o)

load("data/modelingset08.Rdata") 

ds <- cbind(as.data.frame(X), label=y)
names(ds) <- paste("V", 1:297, sep="")
ds.test <- as.data.frame(X.test)
names(ds.test) <- paste("V", 1:296, sep="")

write.csv(ds, "data/modelingset08.csv", row.names=FALSE)
write.csv(ds.test, "data/modelingset08_test.csv", row.names=FALSE)

## Launch h2o on localhost, using all cores
h2oServer = h2o.init(nthreads=-1)

## Point to directory where the Kaggle data is
dir <- paste0("data/")

train.hex <- h2o.importFile("data/modelingset08.csv", destination_frame="train.hex")
test.hex <- h2o.importFile("data/modelingset08_test.csv", destination_frame="test.hex")
dim(train.hex)
summary(train.hex)

predictors <- 1:(ncol(train.hex)-1) 
response <- ncol(train.hex)



######################################################################
## Step 2 - Tuning  
######################################################################

if (FALSE) { # no need to run this again (load optimum parameters from h2o_log_01.RData)
      nn <- 1000
      keys <- read.csv("cv_keys\\cv_5fold_keys.csv")
      # models <- matrix(rep(0,nn*11), nrow=nn)
      load("models\\h2o\\h2o_log_01.RData")
      for (i in 1:nn) {
          rand_numtrees <- sample(c(40,50,60,70, 80),1) ## 1 to 50 trees
          rand_max_depth <- sample(5:20,1) ## 5 to 15 max depth
          rand_min_rows <- sample(1:10,1) ## 1 to 10 min rows
          rand_learn_rate <- 0.025*sample(2:6,1) ## 0.025 to 0.25 learning rate
          models[i, 1:4] <- c(rand_numtrees, rand_max_depth, rand_min_rows, rand_learn_rate)
          model_name <- paste0("GBMModel_",i,
                               "_ntrees",rand_numtrees,
                               "_maxdepth",rand_max_depth,
                               "_minrows",rand_min_rows,
                               "_learnrate",rand_learn_rate)
          err_log <- rep(0,5)
          for (k in 1:5) {
            train_holdout.hex <-  h2o.assign(train.hex[keys[!keys$fold==k,2][1:sum(!keys$fold==k)],], "train_holdout.hex")
            valid_holdout.hex <- h2o.assign(train.hex[keys[keys$fold==k,2][1:sum(keys$fold==k)],], "valid_holdout.hex")
            model <- h2o.gbm(x=predictors, 
                             y=response, 
                             training_frame=train_holdout.hex,
                             validation_frame=valid_holdout.hex,
                             model_id=model_name,
                             distribution="gaussian",
                             ntrees=rand_numtrees, 
                             max_depth=rand_max_depth, 
                             min_rows=rand_min_rows, 
                             learn_rate=rand_learn_rate)    
            err_log[k] <- ScoreQuadraticWeightedKappa(round(as.data.frame(h2o.predict(model, valid_holdout.hex))[,1]),as.data.frame(valid_holdout.hex)[,ncol(valid_holdout.hex)])
          }
          models[i, 5:9] <- err_log
          models[i,10] <- mean(err_log)
          models[i,11] <- sd(err_log)
          save(list=("models"), file="h2o_log_01.RData")
      }
      models <- models[order(-models[,10]),]
#      save(list=c("models", "pred_all", "pred_final"), file="h2o_log_01.RData")
}




######################################################################
## Step 3 - Bagging of top 10 best models 
######################################################################
keys <- read.csv("data/cv_5fold_keys.csv")
load("data/h2o_log_01.RData")
nn = 10
err_log <- rep(0,5)
pred_all <- rep(0,length(y))
for (k in 1:5) {
  
    tr <- keys$index[which(!keys$fold==k)]
    ind.tr <-rep(0,length(y))
    ind.tr[tr] <- 1
    ind.tr.hex <- as.h2o(ind.tr)
    
    vl <- keys$index[which(keys$fold==k)]
    ind.vl <- rep(0,length(y))
    ind.vl[vl] <- 1
    ind.vl.hex <- as.h2o(ind.vl)
    
    train_holdout.hex <-  h2o.assign(train.hex[ind.tr.hex==1,], "train_holdout.hex")
    valid_holdout.hex <- h2o.assign(train.hex[ind.vl.hex==1,], "valid_holdout.hex")
    
    for (i in 1:nn) {
      set.seed(1000*k + 100*i)
      rand_numtrees <-    models[i, 1]
      rand_max_depth <-   models[i, 2]
      rand_min_rows <-    models[i, 3]
      rand_learn_rate <-  models[i, 4]
      
      model_name <- paste0("GBMModel_",i,
                           "_ntrees",rand_numtrees,
                           "_maxdepth",rand_max_depth,
                           "_minrows",rand_min_rows,
                           "_learnrate",rand_learn_rate)
          
    model <- h2o.gbm(x=predictors, 
                     y=response, 
                     training_frame=train_holdout.hex,
                     validation_frame=valid_holdout.hex,
                     model_id=model_name,
                     distribution="gaussian",
                     ntrees=rand_numtrees, 
                     max_depth=rand_max_depth, 
                     min_rows=rand_min_rows, 
                     learn_rate=rand_learn_rate)
    pred_all[ind.vl==1] <-  pred_all[ind.vl==1] + as.data.frame(h2o.predict(model, valid_holdout.hex))[,1]
    }
}

pred_all <-  pred_all /nn
# performance on top 10 models (0.6936311), 0.6880739 second time
ScoreQuadraticWeightedKappa(round(pred_all),y)
write.csv(pred_all+1, paste("data/H2O.gbm_20150616_cv_5fold.csv", sep=""), row.names=FALSE, quote=FALSE)



######################################################################
## Step 4 - Build Final Model using the Full Training Data
######################################################################
load("data/h2o_log_01.RData")   # model parameters
nn = 10                                 # number of bagged models
pred_final <- rep(0,nrow(test.hex))     # vector where test prediction will be saved
for (i in 1:nn) {
  set.seed(100*i)
      rand_numtrees <-    models[i, 1]
      rand_max_depth <-   models[i, 2]
      rand_min_rows <-    models[i, 3]
      rand_learn_rate <-  models[i, 4]
      
      model_name <- paste0("GBMModel_",i,
                           "_ntrees",rand_numtrees,
                           "_maxdepth",rand_max_depth,
                           "_minrows",rand_min_rows,
                           "_learnrate",rand_learn_rate)
      
      model <- h2o.gbm(x=predictors, 
                       y=response, 
                       training_frame=train.hex,
                       model_id=model_name,
                       distribution="gaussian",
                       ntrees=rand_numtrees, 
                       max_depth=rand_max_depth, 
                       min_rows=rand_min_rows, 
                       learn_rate=rand_learn_rate)
                       pred_final <-  pred_final + as.data.frame(h2o.predict(model, test.hex))[,1]
}
pred_final  <- pred_final/nn +1
pred_final.rounded <- as.numeric(cut(pred_final, breaks=c(-Inf, 1.85, 2.80, 3.50, Inf), labels=c(1,2,3,4)))
write.csv(data.frame(id=id.test, predictions=pred_final ), "data/H2O.GBM_20150616_unrounded.csv", row.names=FALSE,  quote = FALSE)
#write.csv(data.frame(id=id.test, predictions=pred_final.rounded), "models\\h2o\\H2O.GBM_20150616_rounded.csv", row.names=FALSE,  quote = FALSE)



## prepare cv files
ScoreQuadraticWeightedKappa(round(pred_all),y) # double check this makes sense
