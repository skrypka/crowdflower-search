# svm regressor (default = radial kernel)
rm(list=ls())

# Load libraries
library(caret)  
library(e1071)
library(Metrics)

# Set working folder
setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

# Load data (convert to data.frame and rename variables)
load("data/modelingset08.Rdata") 


# rename columns
X <- as.data.frame(X)                        # Full training set
names(X) <- paste("V", 1:ncol(X), sep="")
X.test <- as.data.frame(X.test)              # Test
names(X.test) <- paste("V", 1:ncol(X.test), sep="")


# Center, scale
preProcValues1 <- preProcess(X, method = c("center", "scale"))
X <- predict(preProcValues1, X)
X.test <- predict(preProcValues1, X.test)


# step 1. Tune 
# Random search for best parameters (use 80/20 split for tuning)
if (FALSE) {  # (skip and load svm_radial_reg_02.Rdata because we already have tuning parameters)
    nn <- 100 # Number of searches (I only went through about 50 then stopped the script...)
    load("data/svm_radial_reg_02.Rdata") # if running for first time, initialize log_file (line above)
    for (i in which(log_file[,1]==0)[1]:nn) {
        cc <- sample(seq(2,10,1), 1)              # cost
        gg <- sample(seq(0.001, 0.007, 0.001), 1)   # gamma
        
        mod.svm <- svm( y[tr]~., data=X.tr, cost=cc, gamma = gg, typr="eps-regression")
        yhat.vl <- predict(mod.svm, X.vl)
        err <- ScoreQuadraticWeightedKappa(y[-tr], round(yhat.vl)) # kappa
      
       # update log_file, save to disk
       log_file[i,] <- c(cc, gg, err)
       save(list=c("log_file"), file="data/svm_radial_reg_02.Rdata")
      
       # show progress
       print(paste("radial kernel, regression: cost=", cc, ", gamma=", gg, ", kappa=", err, sep=""))      
       flush.console()
    }
    # sort, starting with best model on top
    log_file <- log_file[order(-log_file[,3]),]
    save(list=c(log_file, file="data/svm_radial_reg_02.Rdata")
}



# Let's check how best xx models perform on the 5-fold cv set
load("data/svm_radial_reg_02.Rdata")
if (FALSE) { # optional step. Not needed to generate final predictions
    xx <- 10 # models
    load("models\\svm.reg\\20150624\\svm_radial_reg_02.Rdata")
    pred_all <- matrix(rep(0,length(y)*xx),ncol=xx) # Save all prediction here, for later use
    for (i in 1:xx) {
          cc <- log_file[i, 1]   # cost 
          gg <- log_file[i, 2]   # gamma
          
          key <- read.csv("cv_5fold_keys.csv") # cv common keys
          err <- rep(0,5)
          for (k in 1:5) { # cv loop
            tr <-  key$index[!key$fold==k]  # training indices
            vl <- key$index[key$fold==k]    # validation indices
            mod.svm <- svm(y[tr]~., data=X[tr,], cost=cc, gamma = gg, typr="eps-regression")
            yhat.vl <- predict(mod.svm, X[vl,])
            pred_all[vl,i] <- yhat.vl
            err[k] <- ScoreQuadraticWeightedKappa(y[vl], round(yhat.vl)) 
           }  

          # show progress
          print(paste("radial kernel, regression: cost=", cc, ", gamma=", gg, ", kappa(vl)=", log_file[i,3], ", kappa(cv)=", mean(err), sep=""))      
          flush.console()
    }
}


# Pick one model and generate cv predictions (not necessary if pred_all was saved in step before)


# generate 5fold cv predictions 
load("data/svm_radial_reg_02.Rdata")
key <- read.csv("data/cv_5fold_keys.csv")
prd <- rep(0,length(y))
err <- rep(0,5)
for (k in 1:5) {
  # cv loop
    tr <-  key$index[!key$fold==k]
    vl <- key$index[key$fold==k]
    for (i in 1:10) {
        cc <- log_file[i, 1]
        gg <- log_file[i, 2]
        mod.svm <- svm( y[tr]~., data=X[tr,], cost=cc, gamma = gg, typr="eps-regression")
        yhat.vl <- predict(mod.svm, X[vl,])
        prd[vl] <- prd[vl] +  yhat.vl
      }
      prd[vl] <- prd[vl]/10
      err[k] <- ScoreQuadraticWeightedKappa(y[vl] , round(prd[vl]))
      print(paste("radial kernel, regression: ", cc, ", gamma=", gg, ", kappa(cv", k,")=", err[k], sep=""))      
      flush.console()
}
# These metrics should be similar
mean(err)
ScoreQuadraticWeightedKappa(y, round(prd)) # double check, should be around 0.7
write.csv(prd+1, paste("data/svm.reg_20150703_cv_5fold.csv", sep=""), row.names=FALSE)






# Apply to test data
load("data/svm_radial_reg_02.Rdata")
bst.mod <- 1:10
bst.rounding <- 1 #    (-Inf 0.90 1.75 2.55)   0.7223563    old(-Inf  0.9 1.75 2.50 Inf 0.7152178)
pred.test <- rep(0,length(y))
for (i in bst.mod) {
    cc <- log_file[i, 1]
    gg <- log_file[i, 2]
    mod.svm <- svm( y~., data=X, cost=cc, gamma = gg, type="eps-regression")
    pred.test <- pred.test + predict(mod.svm, X.test)
    print(i)
    flush.console()
}
pred.test <- pred.test/10 +1
pred.test.rounded <- as.numeric(cut(pred.test, breaks= c(-Inf, 0.90, 1.75, 2.55, Inf)+1, labels=c(1,2,3,4)))
write.csv(data.frame(id=id.test, predictions=pred.test), "data/svm.reg_20150703_unrounded.csv", row.names=FALSE,  quote = FALSE)
# write.csv(data.frame(id=id.test, predictions=pred.test.rounded), "models\\svm.reg\\20150624\\svm_regression_x10_20150703_rounded.csv", row.names=FALSE,  quote = FALSE)




