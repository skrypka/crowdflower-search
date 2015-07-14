# generate cv keys

set.seed(97531)
indx <- sample(1:10158, 10158, replace=FALSE)
keys <- data.frame(fold=c(rep(1,2031), rep(2,2032), rep(3,2031), rep(4,2032), rep(5,2032)),
                   index = indx)
write.csv(keys, "data/cv_5fold_keys.csv", row.names=FALSE)
