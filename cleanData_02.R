# R script -- for cleaning the raw data
# Input: train.csv, test.csv
# Output: cleaned versions of the data saved in cleanData02.RData
# Version: 2
# Author: madcap (Maher Harb)


#rm(list=ls())


library(RWeka)
library(stringr)
library(readr)
library(stringdist)
library(tm)
library(qdap)

# ************** CHANGE THIS TO POINT TO RAW DATA ***************

ds1.raw <- read_csv("input/train.csv")
ds2.raw <- read_csv("input/test.csv")

# ***************************************************************

ds1.clean <- ds1.raw
ds2.clean <- ds2.raw
id <- ds2.raw$id

# remove html tags in descriptions
ds1.clean$product_description  <- gsub("<.+?>", "", ds1.clean$product_description )
ds2.clean$product_description  <- gsub("<.+?>", "", ds2.clean$product_description )

# replace &nbsp with space in descriptions
ds1.clean$product_description = gsub("&nbsp", " ", ds1.clean$product_description)
ds2.clean$product_description = gsub("&nbsp", " ", ds2.clean$product_description)

# convert to lower case
ds1.clean$query <- tolower(ds1.clean$query)
ds1.clean$product_title <- tolower(ds1.clean$product_title)
ds1.clean$product_description <- tolower(ds1.clean$product_description)
ds2.clean$query <- tolower(ds2.clean$query)
ds2.clean$product_title <- tolower(ds2.clean$product_title)
ds2.clean$product_description <- tolower(ds2.clean$product_description)

# replace all punctuation and special characters by space
ds1.clean$query <- gsub("[ &<>)(_,.;:!?/-]+", " ", ds1.clean$query)
ds1.clean$product_title <- gsub("[ &<>)(_,.;:!?/-]+", " ", ds1.clean$product_title)
ds1.clean$product_description <- gsub("[ \n\t&<>)(_,.;:!?/-]+", " ", ds1.clean$product_description)
ds2.clean$query <- gsub("[ &<>)(_,.;:!?/-]+", " ", ds2.clean$query)
ds2.clean$product_title <- gsub("[ &<>)(_,.;:!?/-]+", " ", ds2.clean$product_title)
ds2.clean$product_description <- gsub("[ \n\t&<>)(_,.;:!?/-]+", " ", ds2.clean$product_description)

# remove the apostrophe's
ds1.clean$query <- gsub("'s\\b", "", ds1.clean$query)
ds1.clean$product_title <- gsub("'s\\b", "", ds1.clean$product_title)
ds1.clean$product_description <- gsub("'s\\b", "", ds1.clean$product_description)
ds2.clean$query <- gsub("'s\\b", "", ds2.clean$query)
ds2.clean$product_title <- gsub("'s\\b", "", ds2.clean$product_title)
ds2.clean$product_description <- gsub("'s\\b", "", ds2.clean$product_description)

# remove the apostrophe
ds1.clean$query <- gsub("[']+", "", ds1.clean$query)
ds1.clean$product_title <- gsub("[']+", "", ds1.clean$product_title)
ds1.clean$product_description <- gsub("[']+", "", ds1.clean$product_description)
ds2.clean$query <- gsub("[']+", "", ds2.clean$query)
ds2.clean$product_title <- gsub("[']+", "", ds2.clean$product_title)
ds2.clean$product_description <- gsub("[']+", "", ds2.clean$product_description)

# remove the double quotes
ds1.clean$query <- gsub("[\"]+", "", ds1.clean$query)
ds1.clean$product_title <- gsub("[\"]+", "", ds1.clean$product_title)
ds1.clean$product_description <- gsub("[\"]+", "", ds1.clean$product_description)
ds2.clean$query <- gsub("[\"]+", "", ds2.clean$query)
ds2.clean$product_title <- gsub("[\"]+", "", ds2.clean$product_title)
ds2.clean$product_description <- gsub("[\"]+", "", ds2.clean$product_description)


# stem
qq1.stemmed <-  sapply(ds1.clean$query, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ")  })
pt1.stemmed <-  sapply(ds1.clean$product_title, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ")  })
pd1.stemmed <-  sapply(ds1.clean$product_description, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ") })

qq2.stemmed <-  sapply(ds2.clean$query, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ")  })
pt2.stemmed <-  sapply(ds2.clean$product_title, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ")  })
pd2.stemmed <-  sapply(ds2.clean$product_description, function(x){ paste(stemmer(x, capitalize=FALSE), collapse=" ") })


ds1.clean$query <-   qq1.stemmed
ds1.clean$product_title <- pt1.stemmed
ds1.clean$product_description <-   pd1.stemmed

ds2.clean$query <-   qq2.stemmed
ds2.clean$product_title <- pt2.stemmed
ds2.clean$product_description <-   pd2.stemmed



# remove stop words stopwords("english")
ds1.clean$query <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds1.clean$query)
ds1.clean$product_title <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds1.clean$product_title)
ds1.clean$product_description <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds1.clean$product_description)

ds2.clean$query <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds2.clean$query)
ds2.clean$product_title <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds2.clean$product_title)
ds2.clean$product_description <- gsub(paste("(\\b", paste(stopwords("english"), collapse="\\b|\\b"), "\\b)", sep=""), "", ds2.clean$product_description)

ds1.clean$query <- gsub("[ ]+", " ", ds1.clean$query)
ds1.clean$product_title <- gsub("[ ]+", " ",ds1.clean$product_title)
ds1.clean$product_description<- gsub("[ ]+", " ", ds1.clean$product_description)

ds2.clean$query <- gsub("[ ]+", " ", ds2.clean$query)
ds2.clean$product_title <- gsub("[ ]+", " ", ds2.clean$product_title)
ds2.clean$product_description <- gsub("[ ]+", " ", ds2.clean$product_description)


target <- ds1.clean$median_relevance
ds1.clean$id <- NULL
ds1.clean$median_relevance <- NULL

target_variance <- ds1.clean$relevance_variance
ds1.clean$relevance_variance <- NULL
id.test <- ds2.clean$id
ds2.clean$id <- NULL

# save in .RData file
save(list=c("ds1.clean", "ds2.clean", "id.test", "target"), file="data/cleanData02.RData")
