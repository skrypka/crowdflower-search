rm(list=ls())

library(RWeka)
library(stringr)
library(readr)
library(stringdist)
library(tm)
library(qdap)
library(combinat)


ngramMatches <- function(pattern, string1) {
  # pattern is the search query; string1 and string2 are title and description


  # Extract number of words in each string
  w1 <-NGramTokenizer(pattern, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  n1 <- length(w1)
  w2 <-NGramTokenizer(string1, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  n2 <- length(w2)
  out <- c(n1, n2)


  # Tokenize query
  w1 <- NGramTokenizer(pattern, Weka_control(min = 1, max = min(n1,2), delimiters = " \\r\\n.?!:"))
  w1.gramLen <- nchar(gsub("[^ ]", "", w1))+1


  rslt1 <- rep(0,2)           # number of occurences of pattern ngrams of length 2 in string1
  r1 <- rep(0,length(w1))     # binary variable indicating whether the pattern ngram occurs in string1
  r3 <- rep(0,length(w1))     # similarity measure for pattern ngram and string1


  # Populate
  for (i in 1:length(w1)) {
    rslt1[w1.gramLen[i]] <- rslt1[w1.gramLen[i]]  + ntimes(w1[i], string1)
    r1[i] <- ifelse(ntimes(w1[i], string1)>0, 1, 0)
    r3[i] <- simscore(tolower(c(w1[i])), tolower(c(string1)))
  }
  # These are the output features so far
  out <- c(out, sum(r1)/length(r1), sum(r3)/length(r3))



  # Add some similarity measure
  out <- c(out, simscore(pattern, string1))

  return(out)
}


simscore <- function(pattern, string) {
  1-pmin(stringdist( pattern, string, method="lcs")-max(nchar(string)-nchar(pattern),0),nchar(pattern))/nchar(pattern)
}


ntimes <- function (string1, string2){
  string1 <- gsub("[ -]", "[ ]?", string1)
  return((nchar(string2)- nchar(gsub(paste("", string1, "", sep=""), "", string2)))/nchar(string1))
}


ds1.raw <- read_csv("input/train.csv")
ds2.raw <- read_csv("input/test.csv")

target <- ds1.raw[,5]
ds1.raw <- ds1.raw[,2:3]
ds2.raw <- ds2.raw[,2:3]

ds1.raw$product <- ds1.raw$product_title
ds2.raw$product <- ds2.raw$product_title


# Remove single & double apostrophes
ds1.raw$product<- gsub("['\"]+", "", ds1.raw$product)
ds2.raw$product<- gsub("['\"]+", "", ds2.raw$product)


# Remove product codes (long words (>5 characters) that are all caps, numbers or mix pf both)
ds1.raw$product <- gsub("[ ]?\\b[0-9A-Z-]{5,}\\b", "", ds1.raw$product, ignore.case=F)
ds2.raw$product <- gsub("[ ]?\\b[0-9A-Z-]{5,}\\b", "", ds2.raw$product, ignore.case=F)


# Remove descriptions (text between paranthesis/brackets)
ds1.raw$product <- gsub("[ ]?[[(].+?[])]", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("[ ]?[(].+?[)]", "", ds2.raw$product, ignore.case=T)


# Remove "made in..."
ds1.raw$product <- gsub("made in [a-z]+\\b", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("made in [a-z]+\\b", "", ds2.raw$product, ignore.case=T)


# Remove descriptions (hyphen or comma followed by space then at most 2 words, repeated)
ds1.raw$product <- gsub("([,-]( ([a-zA-Z0-9]+\\b)){1,2}[ ]?){1,}$", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("([,-]( ([a-zA-Z0-9]+\\b)){1,2}[ ]?){1,}$", "", ds2.raw$product, ignore.case=T)


# Reemove descriptions (prepositions staring with: with, for, by )
ds1.raw$product <- gsub("\\b(with|for|by|w/) .+$", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("\\b(with|for|by|w/) .+$", "", ds2.raw$product, ignore.case=T)


# colors & sizes
ds1.raw$product <- gsub("size: .+$", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("size: .+$", "", ds2.raw$product, ignore.case=T)
ds1.raw$product <- gsub("size [0-9]+[.]?[0-9]+\\b", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("size [0-9]+[.]?[0-9]+\\b", "", ds2.raw$product, ignore.case=T)
ds1.raw$product <- gsub("([/ -]{1,}(purple|red|blue|white|black|green|pink|yellow|grey|silver|clear|small|large|medium|m|X|2X|xl|navy|aqua|brown|brown leather|sealed|nib|new)[ ]?){1,}$", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("([/ -]{1,}(purple|red|blue|white|black|green|pink|yellow|grey|silver|clear|small|large|medium|m|X|2X|xl|navy|aqua|brown|brown leather|sealed|nib|new)[ ]?){1,}$", "", ds2.raw$product, ignore.case=T)


# dimensions
ds1.raw$product <- gsub("[0-9]{1,}[. ]?[0-9/]{0,}[whl]?[ ]?x[ ]?[0-9]{1,}[. ]?[0-9/]{0,}[whl]?", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("[0-9]{1,}[. ]?[0-9/]{0,}[whl]?[ ]?x[ ]?[0-9]{1,}[. ]?[0-9/]{0,}[whl]?", "", ds2.raw$product, ignore.case=T)


# measurement units
ds1.raw$product <- gsub("[^a-z][0-9.]+[ /-]?(mm|ft|ml|oz|ounce|qt|cu ft|inch|ms|in|pk|ct|ea|pack|cup|pound|fl oz|floz|ghz|pc|gr|g|count|w|v|a|gb|tb)\\b", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("[^a-z][0-9.]+[ /-]?(mm|ft|ml|oz|ounce|qt|cu ft|inch|ms|in|pk|ct|ea|pack|cup|pound|fl oz|floz|ghz|pc|gr|g|count|w|v|a|gb|tb)\\b", "", ds2.raw$product, ignore.case=T)


# other
ds1.raw$product <- gsub("(value bundle|warranty|brand new|excellent condition|one size|new in box|authentic|as is)", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("(value bundle|warranty|brand new|excellent condition|one size|new in box|authentic|as is)", "", ds2.raw$product, ignore.case=T)

# stop words
ds1.raw$product <- gsub("\\b(in)\\b", "", ds1.raw$product, ignore.case=T)
ds2.raw$product <- gsub("(value bundle|warranty|brand new|excellent condition|one size|new in box|authentic|as is)", "", ds2.raw$product, ignore.case=T)


# hyphenated words
ds1.raw$product <- str_replace(ds1.raw$product, "([a-zA-Z])-([a-zA-Z])", "\\1\\2")
ds2.raw$product <- str_replace(ds2.raw$product, "([a-zA-Z])-([a-zA-Z])", "\\1\\2")

# special characters

ds1.raw$product <- gsub("[ &<>)(_,.;:!?/+#*-]+", " ", ds1.raw$product)
ds2.raw$product <- gsub("[ &<>)(_,.;:!?/+#*-]+", " ", ds2.raw$product)

# numbers that are not part of a word
ds1.raw$product <- gsub("\\b[0-9]+\\b", "", ds1.raw$product)
ds2.raw$product <- gsub("\\b[0-9]+\\b", "", ds2.raw$product)


# last two words in product
ds1.raw$product1 <- str_extract(ds1.raw$product, "(([a-zA-Z0-9-]+)\\b[ ]{0,}){2,2}$")
ds1.raw$product1 <- gsub("[ ]+$", "", ds1.raw$product1)

ds2.raw$product1 <- str_extract(ds2.raw$product, "(([a-zA-Z0-9-]+)\\b[ ]{0,}){2,2}$")
ds2.raw$product1 <- gsub("[ ]+$", "", ds2.raw$product1)

f1<- data.frame()
for (i in 1:length(ds1.raw$product1)) {
  f1 <- rbind(f1, ngramMatches(as.character(ds1.raw[i,1]), as.character(ds1.raw[i,4])))
}
indx = (is.na(ds1.raw$product1))

f1[indx,5] <- mean(f1[!indx,5])
f1[indx,4] <- mean(f1[!indx,4])



f2 <- data.frame()
for (i in 1:length(ds2.raw$product1)) {
  f2 <- rbind(f2, ngramMatches(as.character(ds2.raw[i,1]), as.character(ds2.raw[i,4])))
}
indx = (is.na(ds2.raw$product1))

f2[indx,5] <- mean(f2[!indx,5])
f2[indx,4] <- mean(f2[!indx,4])

write.csv(ds1.raw[,c(1,2,4)], "data/product_names.csv", row.names=FALSE)
write.csv(ds2.raw[,c(1,2,4)], "data/product_names_test.csv", row.names=FALSE)

write.csv(f1[,4], "data/product_simscore_train.csv", row.names=FALSE)
write.csv(f2[,4], "data/product_simscore_test.csv", row.names=FALSE)
