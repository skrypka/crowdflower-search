# R script -- for generating similarity features between alternative query & title
# Input: train.csv, test.csv
# Output: alt_query_features_train_and_test_v01.csv
# Version: 1
# Author: madcap (Maher Harb)

#rm(list=ls())

#setwd('C:/Users/Maher/Google Drive/Crowdflower Competition/madcap_final_scripts/features/alternative_query')

library(RWeka)
library(stringdist)
library(combinat)
library(readr)
library(qdap)
library(tm)
source("ngramMatches07_alt_queries.R")


# get queries & titles from raw data
# ************** CHANGE THIS TO POINT TO RAW DATA ***************

ds1.raw <- read_csv("input/train.csv")
ds2.raw <- read_csv("input/test.csv")

# ***************************************************************
dset <- as.data.frame(rbind(ds1.raw[,1:3], ds2.raw[1:3]))


# dset <- dset [sample(32671,5),]


# do some cleaing on raw titles
i <- 3
dset[,i] <- tolower(dset[,i]) # convert to lower case
dset[,i] <- gsub("[ &<>)(_,.;:!?/-]+", " ", dset[,i]) # replace all punctuation and special characters by space
dset[,i] <- gsub("'s\\b", "", dset[,i]) # remove the apostrophe's
dset[,i] <- gsub("[']+", "", dset[,i]) # remove the apostrophe
dset[,i] <- gsub("[\"]+", "", dset[,i])   # remove the double quotes
dset[,i] <- sapply(dset[,i], function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ")  })  # stem
dset[,i] <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", dset[,i]) # remove stop words stopwords("english")


# combine titles into one corpus (grouped by query)
queries <- unique(dset[,2])
alt_queries <- character()
for (q in queries) {
  titles <- dset$product_title[dset$query==q]
  grams <- NGramTokenizer(titles, Weka_control(min = 3, max = 3, delimiters = " \\r\\n.?!:"))
  wd <- as.data.frame(table(grams))
  wd <- wd[order(-wd$Freq),]
  alt_queries <- c(alt_queries, as.character(wd$grams[1]))
}
qmap <- data.frame(queries, alt_queries)

dset$row.seq <- 1:nrow(dset)
dset <- merge(x=dset, y=qmap, by.x="query", by.y="queries", x.all=TRUE)
dset <- dset[,c(2,1,5,3,4)]
dset <- dset[order(dset$row.seq),]


# generate features based on the ngram similarity function
XXX <- t(apply(dset, 1, function(x){ngramMatches(x[3], x[4])}))


# save
write.csv(XXX[,3:5], "data/alt_query_features_train_and_test_v01.csv", row.names=FALSE)