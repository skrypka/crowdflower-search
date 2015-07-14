
ngramMatches <- function(pattern, string1, string2) {
# pattern is the search query; string1 and string2 are title and description 
  
  
  # Extract number of words in each string
  w1 <-NGramTokenizer(pattern, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  n1 <- length(w1)
  w2 <-NGramTokenizer(string1, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  n2 <- length(w2)
  w3 <-NGramTokenizer(string2, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  n3 <- length(w3)
  out <- c(n1, n2, n3)
  
  
  # Tokenize query
  w1 <- NGramTokenizer(pattern, Weka_control(min = 1, max = min(n1,4), delimiters = " \\r\\n.?!:"))
  w1.gramLen <- nchar(gsub("[^ ]", "", w1))+1
  
  
  rslt1 <- rep(0,4)           # number of occurences of pattern ngrams of length 1-5 in string1
  rslt2 <- rep(0,4)           # number of occurences of pattern ngrams of length 1-5 in string2
  r1 <- rep(0,length(w1))     # binary variable indicating whether the pattern ngram occurs in string1
  r2 <- rep(0,length(w1))     # binary variable indicating whether the pattern ngram occurs in string2
  r3 <- rep(0,length(w1))     # similarity measure for pattern ngram and string1
  r4 <- rep(0,length(w1))     # similarity measure for pattern ngram and string1
  
  # Populate
  for (i in 1:length(w1)) {
        rslt1[w1.gramLen[i]] <- rslt1[w1.gramLen[i]]  + ntimes(w1[i], string1)
        rslt2[w1.gramLen[i]] <- rslt2[w1.gramLen[i]]  + ntimes(w1[i], string2)
        r1[i] <- ifelse(ntimes(w1[i], string1)>0, 1, 0)
        r2[i] <- ifelse(ntimes(w1[i], string2)>0, 1, 0)
        r3[i] <- simscore(w1[i], string1)
        r4[i] <- simscore(w1[i], string2)
  } 
  # These are the output features so far
  out <- c(out, rslt1, rslt2, sum(r1)/length(r1), sum(r2)/length(r2), sum(r3)/length(r3), sum(r4)/length(r4))
  
  
  # second order bigram matching (reshuffle words in query)
  w1.single <- w1[w1.gramLen==1]
  w1.bigram <- w1[w1.gramLen==2]
  all.comb <- apply(rbind(combn2(w1.single), combn2(w1.single)[,2:1]), 1, paste, collapse =" ")
  all.comb <- all.comb[!all.comb %in% w1.bigram]
  r5 <- rep(0,length(all.comb))
  r6 <- rep(0,length(all.comb))
  for (i in 1:length(all.comb)) {
    r5[i] <- simscore(all.comb[i], string1)
    r6[i] <- simscore(all.comb[i], string2)
  } 
  out <- c(out, sum(r5)/length(r5), sum(r6)/length(r6))
  
  
  # Add some similarity measure                                                                  
  out <- c(out, simscore(pattern, string1), simscore(pattern, string2))
  
  return(out)
}


simscore <- function(pattern, string) {
  1-pmin(stringdist( pattern, string, method="lcs")-max(nchar(string)-nchar(pattern),0),nchar(pattern))/nchar(pattern)
} 

  
ntimes <- function (string1, string2){
  string1 <- gsub("[ -]", "[ ]?", string1)
  return((nchar(string2)- nchar(gsub(paste("", string1, "", sep=""), "", string2)))/nchar(string1))
}
