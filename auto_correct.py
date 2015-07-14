from bs4 import BeautifulSoup
import difflib
from nltk import bigrams
import re
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from myml.files import dump

train = pd.read_csv("input/train.csv").fillna("")
test  = pd.read_csv("input/test.csv").fillna("")

def build_query_correction_map(print_different=True):
        # get all query
        queries = set(train['query'].values)
        correct_map = {}
        if print_different:
            print("%30s \t %30s"%('original query','corrected query'))
        for q in queries:
                corrected_q = autocorrect_query(q,train=train,test=test,warning_on=False)
                if print_different and q != corrected_q:
                    print ("%30s \t %30s"%(q,corrected_q))
                correct_map[q] = corrected_q
        return correct_map

def autocorrect_query(query,train=None,test=None,cutoff=0.8,warning_on=True):
        """
        autocorrect a query based on the training set
        """
        train_data = train.values[train['query'].values==query,:]
        test_data = test.values[test['query'].values==query,:]
        s = ""
        for r in train_data:
                s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
        for r in test_data:
                s = "%s %s %s"%(s,BeautifulSoup(r[2]).get_text(" ",strip=True),BeautifulSoup(r[3]).get_text(" ",strip=True))
        s = re.findall(r'[\'\"\w]+',s.lower())
        s_bigram = [' '.join(i) for i in bigrams(s)]
        s.extend(s_bigram)
        corrected_query = []
        for q in query.lower().split():
                if len(q)<=2:
                        corrected_query.append(q)
                        continue
                corrected_word = difflib.get_close_matches(q, s,n=1,cutoff=cutoff)
                if len(corrected_word) >0:
                        corrected_query.append(corrected_word[0])
                else :
                        if warning_on:
                                print ("WARNING: cannot find matched word for '%s' -> used the original word"%(q))
                        corrected_query.append(q)
        return ' '.join(corrected_query)
query_map = build_query_correction_map()
dump('data/query_auto_correct', query_map)
