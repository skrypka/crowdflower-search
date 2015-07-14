#The core score computing script for the create_Okapi Script
#for info about Okapi bm25 read: https://en.wikipedia.org/wiki/Okapi_BM25
Okapi <- function(pattern, string1, string2,data_clean,mn_qlength) {

  #number of queries
  N=nrow(data_clean)
    
  #how many times this query happens in the data  
  nq_data<-length(which(data_clean$query==pattern)) 
  
  w1 <-NGramTokenizer(pattern, Weka_control(min = 1, max = 1, delimiters = " \\r\\n.?!:"))
  
  #Sum the Score, for each token
  score<-0
  
  IDF=log((N-nq_data+0.5)/(nq_data+0.5))
  idf<-rep(0,length(w1))
  
  k=1.6
  b=0.75
  
  for (i in 1:length(w1)) {
    fq=ntimes(w1[i],pattern)/length(w1)
    
    score<-score+(IDF*(fq*(1+k))/(fq+k*(1-b+b*length(w1)/mn_qlength)))
    
  } 

  nq_data<-nq_data/N
  
  out <- c(nq_data, IDF, score)
  return(out)

}

ntimes <- function (string1, string2){
  string1 <- gsub("[ -]", "[ ]?", string1)
  return((nchar(string2)- nchar(gsub(paste("", string1, "", sep=""), "", string2)))/nchar(string1))
}