#This script generates a similarity feature using a W2vec model trained with a corpus from google news.
#the W2vec model can be found at: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
#train and test file must be in ../Data/XXXX.csv
#feature csv file is dumped to that folder aswell
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC,SVR
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
from nltk.stem.porter import *
import gensim,logging
#import Word2Vec

def clean(s):
        # Returns unique token-sorted cleaned lowercased text
        return " ".join(sorted(set(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)))).lower()

def extract_features(data,model):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")

        w2vec_feature=[]
        feat_importances=[]
        w2vec_feature_imp=[]
        d_w2vec_feature=[]
        d_feat_importances=[]
        d_w2vec_feature_imp=[]
        for i, row in data.iterrows():
                q_words = clean("%s"%(row["query"])).split()
                t_words = clean("%s"%(row["product_title"])).split()
                d_words = clean("%s"%(row["product_description"])).split()
                prev_q_len=len(q_words)
                prev_t_len=len(t_words)
                prev_d_len=len(d_words)
                q_words = filter(lambda x: x in model.vocab, q_words) #take words that are in the vocabulary, otherwise it crashes
                t_words = filter(lambda x: x in model.vocab, t_words)
                d_words = filter(lambda x: x in model.vocab, d_words)
                q_len=len(q_words)
                t_len=len(t_words)
                d_len=len(d_words)
                feat_importance=float(q_len+t_len)/float(prev_q_len+prev_t_len)
                feat_importances.append(feat_importance)
                d_feat_importance=float(q_len+d_len)/float(prev_q_len+prev_d_len)
                d_feat_importances.append(d_feat_importance)
                for t_word in t_words:
                        if len(t_word)<3: #oh! thats a heart!
                                t_words.remove(t_word)

                sim=model.n_similarity(q_words,t_words) #calculate the similarity between query and title
                d_sim=model.n_similarity(q_words,d_words)
                #print 'Q: '+str(row["query"])+' ----> ' +str(q_words)
                #print 'Title: '+str(row["product_title"])+' ----> '+str(t_words)
                #print 'similarity: '+str(sim)
                #print 'Label: '+str(row['median_relevance'])
                if ((len(q_words)==0)or(len(t_words)==0)): #check if we still have words (maybe no Q words were in the word2vec vocab)
                        w2vec_feature.append(0)
                        w2vec_feature_imp.append(0)
                else:
                        w2vec_feature.append(sim)
                        w2vec_feature_imp.append(sim*feat_importance)

                #print 'Label: '+str(row['median_relevance'])
                if ((len(q_words)==0)or(len(d_words)==0)):
                        d_w2vec_feature.append(0)
                        d_w2vec_feature_imp.append(0)
                else:
                        d_w2vec_feature.append(d_sim)
                        d_w2vec_feature_imp.append(d_sim*d_feat_importance)
                #print  sorted(v)[-3:]
                #print  sum(sorted(v)[-4:])
                #nkd_feature.append((0.125)*sum(sorted(v)[:4]))

        return w2vec_feature,feat_importances,w2vec_feature_imp,d_w2vec_feature,d_feat_importances,d_w2vec_feature_imp



print 'Loading model'
model = gensim.models.Word2Vec.load_word2vec_format('/media/fromto/data/GoogleNews-vectors-negative300.bin', binary=True)
print 'Model loaded'

train = pd.read_csv('input/train.csv').fillna("")
test = pd.read_csv('input/test.csv').fillna("")

print 'Extracting W2vec similarity between query and title'
w2vec_train,f_imp_train,w2vec_train_imp,d_w2vec_train,d_f_imp_train,d_w2vec_train_imp=extract_features(train,model)
w2vec_test,f_imp_test,w2vec_test_imp,d_w2vec_test,d_f_imp_test,d_w2vec_test_imp=extract_features(test,model)

w2vec_train_=np.asarray((w2vec_train,f_imp_train,w2vec_train_imp,d_w2vec_train,d_f_imp_train,d_w2vec_train_imp))
w2vec_test_=np.asarray((w2vec_test,f_imp_test,w2vec_test_imp,d_w2vec_test,d_f_imp_test,d_w2vec_test_imp))
w2vec_train_=np.transpose(w2vec_train_)
w2vec_test_=np.transpose(w2vec_test_)

#print w2vec_train_.shape
#print w2vec_test_.shape

print 'Features extracted, dumping to csv files'
np.savetxt("data/w2vec_train.csv", (w2vec_train_), delimiter=",")
np.savetxt("data/w2vec_test.csv", (w2vec_test_), delimiter=",")
