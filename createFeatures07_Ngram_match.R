# R script -- for generating similarity features between query & title/description
# Input: cleanData02.RData
# Output: features saved as ngramFeatures07.RData as well as csv version (ngramMatch_07.csv, ngramMatch_test_07.csv)
# Version: 7
# Author: madcap (Maher Harb)


rm(list=ls())

# setwd('C:\\Data\\Data Science\\kaggle\\CrowdFlower\\')

library(RWeka)
library(stringdist)
library(combinat)
source("ngramMatches07.R") # The core feature extraction function

# ************** CHANGE THIS TO POINT TO CLEAN DATA ***************

# cleaned versions of train.csv & test.csv, stemmed, stop words removed
load(file = "data/cleanData02.RData")

# ***************************************************************

# Add features to train data
X2 <- data.frame()
for (i in 1:nrow(ds1.clean)) {
  temp <- ngramMatches(ds1.clean$query[i], ds1.clean$product_title[i], ds1.clean$product_description[i])
  X2 <- rbind(X2, temp)
}

# rename columns
names(X2) <- c("Q.len", "PT.len", "PD.len", "PT.1grms", "PT.2grms", "PT.3grms", "PT.4grms",
               "PD.1grms", "PD.2grms", "PD.3grms", "PD.4grms", "PT.ngrm.matchscore", "PD.ngrm.matchscore",
               "PT.ngrm.simscore", "PD.ngrm.simscore", "PT.2grms.2ndorder", "PD.2grms.2ndorder",
               "PT.similarity", "PD.similarity")


# Impute where title doesn't have alternative reordered bigrams
X2$PT.2grms.2ndorder[is.na(X2$PT.2grms.2ndorder)] <- mean(X2$PT.2grms.2ndorder, na.rm=T)

# Impute empty description
indx = (X2$PD.len==0)
X2$PD.2grms.2ndorder[indx] <- NA
X2$PD.2grms.2ndorder[is.na(X2$PD.2grms.2ndorder)] <- mean(X2$PD.2grms.2ndorder, na.rm=T)
X2$PD.1grms[indx] <- NA
X2$PD.2grms[indx] <- NA
X2$PD.3grms[indx] <- NA
X2$PD.4grms[indx] <- NA
X2$PD.ngrm.matchscore[indx] <- NA
X2$PD.ngrm.simscore[indx] <- NA
X2$PD.similarity[indx] <- NA
X2$PD.1grms[indx] <- mean(X2$PD.1grms, na.rm=T)
X2$PD.2grms[indx] <- mean(X2$PD.2grms, na.rm=T)
X2$PD.3grms[indx] <- mean(X2$PD.3grms, na.rm=T)
X2$PD.4grms[indx] <- mean(X2$PD.4grms, na.rm=T)
X2$PD.ngrm.matchscore[indx] <- mean(X2$PD.ngrm.matchscore, na.rm=T)
X2$PD.ngrm.simscore[indx] <- mean(X2$PD.ngrm.simscore, na.rm=T)
X2$PD.similarity[indx] <- mean(X2$PD.similarity, na.rm=T)

# Add features to test data
X2.test <- data.frame()
for (i in 1:nrow(ds2.clean)) {
  temp <- ngramMatches(ds2.clean$query[i], ds2.clean$product_title[i], ds2.clean$product_description[i])
  X2.test <- rbind(X2.test, temp)
}

# Rename columns
names(X2.test) <- c("Q.len", "PT.len", "PD.len", "PT.1grms", "PT.2grms", "PT.3grms", "PT.4grms",
                    "PD.1grms", "PD.2grms", "PD.3grms", "PD.4grms", "PT.ngrm.matchscore", "PD.ngrm.matchscore",
                    "PT.ngrm.simscore", "PD.ngrm.simscore", "PT.2grms.2ndorder", "PD.2grms.2ndorder",
                    "PT.similarity", "PD.similarity")

# Impute where title doesn't have alternative reordered bigrams
X2.test$PT.2grms.2ndorder[is.na(X2.test$PT.2grms.2ndorder)] <- mean(X2.test$PT.2grms.2ndorder, na.rm=T)

# Impute empty description
indx = (X2.test$PD.len==0)
X2.test$PD.2grms.2ndorder[indx] <- NA
X2.test$PD.2grms.2ndorder[is.na(X2.test$PD.2grms.2ndorder)] <- mean(X2.test$PD.2grms.2ndorder, na.rm=T)
X2.test$PD.1grms[indx] <- NA
X2.test$PD.2grms[indx] <- NA
X2.test$PD.3grms[indx] <- NA
X2.test$PD.4grms[indx] <- NA
X2.test$PD.ngrm.matchscore[indx] <- NA
X2.test$PD.ngrm.simscore[indx] <- NA
X2.test$PD.similarity[indx] <- NA
X2.test$PD.1grms[indx] <- mean(X2.test$PD.1grms, na.rm=T)
X2.test$PD.2grms[indx] <- mean(X2.test$PD.2grms, na.rm=T)
X2.test$PD.3grms[indx] <- mean(X2.test$PD.3grms, na.rm=T)
X2.test$PD.4grms[indx] <- mean(X2.test$PD.4grms, na.rm=T)
X2.test$PD.ngrm.matchscore[indx] <- mean(X2.test$PD.ngrm.matchscore, na.rm=T)
X2.test$PD.ngrm.simscore[indx] <- mean(X2.test$PD.ngrm.simscore, na.rm=T)
X2.test$PD.similarity[indx] <- mean(X2.test$PD.similarity, na.rm=T)

# Inspect correlations between features (if >.9 manually remove duplicate features)
cor(X2)>0.9 & !diag(ncol(X2))
X2$PT.similarity <- NULL
X2.test$PT.similarity <- NULL
X2$PD.ngrm.simscore <- NULL
X2.test$PD.ngrm.simscore <- NULL
X2$PD.similarity <- NULL
X2.test$PD.similarity <- NULL

# various normalizations -- based on enhancing correlation between target and features
X2$PT.1grms <- X2$PT.1grms/X2$Q.len
X2$PT.2grms <- X2$PT.2grms/X2$Q.len
X2$PT.3grms <- X2$PT.3grms/X2$Q.len
X2$PT.4grms <- X2$PT.4grms/X2$Q.len
X2$PD.1grms <- X2$PD.1grms/X2$Q.len
X2$PD.2grms <- X2$PD.2grms/X2$Q.len
X2$PD.3grms <- X2$PD.3grms/X2$Q.len
X2$PD.4grms <- X2$PD.4grms/X2$Q.len
X2$PT.ngrm.matchscore <- X2$PT.ngrm.matchscore
X2$PD.ngrm.matchscore <- X2$PD.ngrm.matchscore^2
X2$PT.ngrm.simscore <- X2$PT.ngrm.simscore
X2$PT.2grms.2ndorder <- X2$PT.2grms.2ndorder^0.5
X2$len.ratio <- X2$Q.len/X2$PT.len
X2$Q.len <- NULL
X2$PT.len <- NULL
X2$PD.len <- NULL

X2.test$PT.1grms <- X2.test$PT.1grms/X2.test$Q.len
X2.test$PT.2grms <- X2.test$PT.2grms/X2.test$Q.len
X2.test$PT.3grms <- X2.test$PT.3grms/X2.test$Q.len
X2.test$PT.4grms <- X2.test$PT.4grms/X2.test$Q.len
X2.test$PD.1grms <- X2.test$PD.1grms/X2.test$Q.len
X2.test$PD.2grms <- X2.test$PD.2grms/X2.test$Q.len
X2.test$PD.3grms <- X2.test$PD.3grms/X2.test$Q.len
X2.test$PD.4grms <- X2.test$PD.4grms/X2.test$Q.len
X2.test$PT.ngrm.matchscore <- X2.test$PT.ngrm.matchscore
X2.test$PD.ngrm.matchscore <- X2.test$PD.ngrm.matchscore^2
X2.test$PT.ngrm.simscore <- X2.test$PT.ngrm.simscore
X2.test$PT.2grms.2ndorder <- X2.test$PT.2grms.2ndorder^0.5
X2.test$len.ratio <- X2.test$Q.len/X2.test$PT.len
X2.test$Q.len <- NULL
X2.test$PT.len <- NULL
X2.test$PD.len <- NULL

# save file
save(list=c("X2", "X2.test"), file="data/ngramFeatures07.RData")

# export to csv (optional)
write.csv(X2, "data/ngramMatch_07.csv", row.names=FALSE)
write.csv(X2.test, "data/ngramMatch_test_07.csv", row.names=FALSE)
