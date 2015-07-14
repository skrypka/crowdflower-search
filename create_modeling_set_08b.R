rm(list=ls())

setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/final-code/')

# engineered features
load(file = "data/ngramFeatures07.RData")


# features from Roman
X1 <- read.csv("data/X250_2.csv", header = FALSE)
X1.test <- read.csv("data/X250_test_2.csv",header = FALSE)

# more features from Roman
f1 <- read.csv("data/train1234.csv", header = FALSE)
f2 <- read.csv("data/train1234_2.csv", header = FALSE)
f3 <- read.csv("data/train1234_3.csv", header = FALSE)


f1.test <- read.csv("data/test1234.csv", header = FALSE)
f2.test <- read.csv("data/test1234_2.csv", header = FALSE)
f3.test <- read.csv("data/test1234_3.csv", header = FALSE)

# okapi
f4 <- read.csv("data/Okapi_train.csv", header = TRUE)
f4.test <- read.csv("data/Okapi_test.csv", header = TRUE)

# word2vec
f5 <- read.csv("data/w2vec_train.csv", header = FALSE)
f5.test <- read.csv("data/w2vec_test.csv", header = FALSE)


# this's just to get the label
load(file = "data/cleanData02.RData")
rm(list=c("ds1.clean", "ds2.clean"))



# prepare data for xgboost
X <- as.matrix(cbind(X2, X1, f1, f2, f3, f4, f5))
X.test <- as.matrix(cbind(X2.test, X1.test, f1.test, f2.test, f3.test, f4.test, f5.test))
y <- target - 1 # because xgboost expects levels to be named 0,1,...,n-1

colnames(X) <- paste("V", 1:ncol(X), sep="")
colnames(X.test) <- paste("V", 1:ncol(X.test), sep="")

# save modleing data for future use
save(list=c("X", "X.test", "y", "id.test"), file="data/modelingset08b.Rdata")

