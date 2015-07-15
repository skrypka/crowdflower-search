import csv
import re
from collections import defaultdict
import math
import json
import pandas as pd
import numpy as np

def clean(s):
        # Returns unique token-sorted cleaned lowercased text
        return " ".join(sorted(set(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)))).lower()

def index_document(s,d):
        # Creates half the matrix of pairwise tokens
        # This fits into memory, else we have to choose a Count-min Sketch probabilistic counter
        tokens = s.split()
        for x in range(len(tokens)):
                d[tokens[x]] += 1
                for y in range(x+1,len(tokens)):
                        d[tokens[x]+"_X_"+tokens[y]] += 1
        return d

def index_corpus():
        # Create our count dictionary and fill it with train and test set (pairwise) token counts
        d = defaultdict(int)
        for e, row in enumerate( csv.DictReader(open("input/train.csv",'r'))):
                s = clean("medianrellabel%s %s %s"%(row["median_relevance"], row["product_description"],row["product_title"]))
                d = index_document(s,d)
        for e, row in enumerate( csv.DictReader(open("input/test.csv",'r'))):
                s = clean("%s %s"%(row["product_description"],row["product_title"]))
                d = index_document(s,d)
        return d

def nkd(token1, token2, d):
        # Returns the semantic Normalized Kaggle Distance between two tokens
        sorted_tokens = sorted([clean(token1), clean(token2)])
        token_x = sorted_tokens[0]
        token_y = sorted_tokens[1]
        if d[token_x] == 0 or d[token_y] == 0 or d[token_x+"_X_"+token_y] == 0:
                return 2.
        else:
                #print d[token_x], d[token_y], d[token_x+"_X_"+token_y], token_x+"_X_"+token_y
                logcount_x = math.log(d[token_x])
                logcount_y = math.log(d[token_y])
                logcount_xy = math.log(d[token_x+"_X_"+token_y])
                log_index_size = math.log(100000) # fixed guesstimate
                nkd = (max(logcount_x,logcount_y)-logcount_xy) / (log_index_size-min(logcount_x,logcount_y))
                return nkd

def extract_features(data): #the features are the number of times that query tokens appear in the title or description, which is cool
    token_pattern = re.compile(r"(?u)\b\w\w+\b")

    data["query_tokens_in_title"] = 0.0
    nkd_feature=[]
    for i, row in data.iterrows():
        q_words = clean("%s"%(row["query"])).split()
        t_words = clean("%s"%(row["product_title"])).split()
        v=[]
        for q_word in q_words:
            for t_word in t_words:
                if ((len(q_word)>3)& (len(t_word)>3)):
                        v.append(nkd(q_word,t_word,d))
        #print 'Q: '+str(row["query"])
        #print 'Title: '+str(row["product_title"])
        #print 'nkd: '+str((0.125)*sum(sorted(v)[:4]))
        #print 'Label: '+str(row['median_relevance'])
        #print  sorted(v)[-3:]
        #print  sum(sorted(v)[-4:])
        nkd_feature.append((0.125)*sum(sorted(v)[:4]))

    return nkd_feature

d=index_corpus()
print 'Corpus indexed'

train = pd.read_csv('input/train.csv').fillna("")
test = pd.read_csv('input/test.csv').fillna("")

print nkd('shampoo','shampoo',d)

print 'Extracting features for train and test'
nkd_train=extract_features(train)
nkd_test=extract_features(test)

nkd_train=np.asarray(nkd_train)
nkd_test=np.asarray(nkd_test)

print 'Features extracted, dumping to csv files'
np.savetxt("data/kaggle_dist_train.csv", nkd_train, delimiter=",")
np.savetxt("data/kaggle_dist_test.csv", nkd_test, delimiter=",")
