
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
        r3[i] <- simscore(w1[i], string1)
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
