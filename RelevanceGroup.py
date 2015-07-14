from __future__ import division

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import numpy.linalg as LA

from myml.files import dump, load
from utils import process_str_replace
from myml.utils import clr_print


# In[3]:

process_str = process_str_replace


# In[4]:

X2, X2_test, y = load("data/XXtestY250_r2td")


# In[5]:

train = pd.read_csv("input/train.csv").fillna("")
test = pd.read_csv("input/test.csv").fillna("")
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)
train.head(3)


# In[6]:

qx = set(train['query'].values)


# In[6]:

def get_str_for_query(train, q, product_title, product_description, median_relevance, id=None):
    #prev_df = train[(train['query']==q) & \
    #           (train['median_relevance']==median_relevance)]
    df = train[(train['query']==q) &                (train['median_relevance']==median_relevance) &                (train['product_title']!=product_title) &                (train['product_description']!=product_description) &                (train.index!=id)]
    #print id, prev_df.shape[0], df.shape[0], median_relevance
    title_set = set()
    for e in df['product_title'].values:
        title_set |= set(process_str(e))
    description_set = set()
    for e in df['product_description'].values:
        description_set |= set(process_str(e))
    return title_set, description_set


# In[7]:

def create_similarity_features(train, row, id=None):
    tx1, dx1 = get_str_for_query(train, row['query'], row['product_title'], row['product_description'], 1, id)
    tx2, dx2 = get_str_for_query(train, row['query'], row['product_title'], row['product_description'], 2, id)
    tx3, dx3 = get_str_for_query(train, row['query'], row['product_title'], row['product_description'], 3, id)
    tx4, dx4 = get_str_for_query(train, row['query'], row['product_title'], row['product_description'], 4, id)
    our_tx = set(process_str(row['product_title']))
    our_dx = set(process_str(row['product_description']))

    return len(tx1 & our_tx), len(dx1 & our_dx), len(tx2 & our_tx), len(dx2 & our_dx), len(tx3 & our_tx), len(dx3 & our_dx), len(tx4 & our_tx), len(dx4 & our_dx)


# In[8]:

train_fx = np.array([create_similarity_features(train, train.iloc[i], i) for i in xrange(train.shape[0])])
dump('data/train1234_c1_r', train_fx)
test_fx = np.array([create_similarity_features(train, test.iloc[i]) for i in xrange(test.shape[0])])
dump('data/test1234_c1_r', test_fx)


# ## Similarity less strict

# In[9]:

def get_str_for_query2(train, q, product_title, product_description, median_relevance, id=None):
    df = train[(train['query']==q) &
               (train['median_relevance']==median_relevance) &
               (train.index!=id)]
    title_set = set()
    for e in df['product_title'].values:
        title_set |= set(process_str(e))
    description_set = set()
    for e in df['product_description'].values:
        description_set |= set(process_str(e))
    return title_set, description_set


# In[10]:

def create_similarity_features2(train, row, id=None):
    tx1, dx1 = get_str_for_query2(train, row['query'], row['product_title'], row['product_description'], 1, id)
    tx2, dx2 = get_str_for_query2(train, row['query'], row['product_title'], row['product_description'], 2, id)
    tx3, dx3 = get_str_for_query2(train, row['query'], row['product_title'], row['product_description'], 3, id)
    tx4, dx4 = get_str_for_query2(train, row['query'], row['product_title'], row['product_description'], 4, id)
    our_tx = set(process_str(row['product_title']))
    our_dx = set(process_str(row['product_description']))

    return len(tx1 & our_tx), len(dx1 & our_dx), len(tx2 & our_tx), len(dx2 & our_dx), len(tx3 & our_tx), len(dx3 & our_dx), len(tx4 & our_tx), len(dx4 & our_dx)


# In[11]:

train_fx = np.array([create_similarity_features2(train, train.iloc[i], i) for i in xrange(train.shape[0])])
dump('data/train1234_2_c1_r', train_fx)
test_fx = np.array([create_similarity_features2(train, test.iloc[i]) for i in xrange(test.shape[0])])
dump('data/test1234_2_c1_r', test_fx)


# ## Similarity seq (mean, max)

# In[12]:

def get_str_for_query3(train, q, product_title, product_description, median_relevance, id=None):
    df = train[(train['query']==q) &
               (train['median_relevance']==median_relevance) &
               (train.index!=id)]
    tx = [' '.join(process_str(e)) for e in df['product_title'].values]
    dx = [' '.join(process_str(e)) for e in df['product_description'].values]

    return tx, dx


# In[13]:

stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words = stopWords)
transformer = TfidfTransformer()
cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)


# In[14]:

def create_similarity_features3(train, row, id=None):
    tx1, dx1 = get_str_for_query3(train, row['query'], row['product_title'], row['product_description'], 1, id)
    tx2, dx2 = get_str_for_query3(train, row['query'], row['product_title'], row['product_description'], 2, id)
    tx3, dx3 = get_str_for_query3(train, row['query'], row['product_title'], row['product_description'], 3, id)
    tx4, dx4 = get_str_for_query3(train, row['query'], row['product_title'], row['product_description'], 4, id)

    our_tx = ' '.join(process_str(row['product_title']))
    our_dx = ' '.join(process_str(row['product_description']))

    b_tx = tx1+tx2+tx3+tx4
    b_tx = vectorizer.fit_transform(b_tx)
    transformer.fit(b_tx)

    testV = transformer.transform(vectorizer.transform([our_tx])).toarray()[0]

    if tx1:
        trainV = transformer.transform(vectorizer.transform(tx1)).toarray()
        tx1out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        tx1out = np.array([0])
    if tx2:
        trainV = transformer.transform(vectorizer.transform(tx2)).toarray()
        tx2out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        tx2out = np.array([0])
    if tx3:
        trainV = transformer.transform(vectorizer.transform(tx3)).toarray()
        tx3out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        tx3out = np.array([0])
    if tx4:
        trainV = transformer.transform(vectorizer.transform(tx4)).toarray()
        tx4out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        tx4out = np.array([0])

    testV = transformer.transform(vectorizer.transform([our_dx])).toarray()[0]

    if dx1:
        trainV = transformer.transform(vectorizer.transform(dx1)).toarray()
        dx1out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        dx1out = np.array([0])
    if dx2:
        trainV = transformer.transform(vectorizer.transform(dx2)).toarray()
        dx2out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        dx2out = np.array([0])
    if dx3:
        trainV = transformer.transform(vectorizer.transform(dx3)).toarray()
        dx3out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        dx3out = np.array([0])
    if dx4:
        trainV = transformer.transform(vectorizer.transform(dx4)).toarray()
        dx4out = np.array([cx(v, testV) for v in trainV]+[0])
    else:
        dx4out = np.array([0])

    return np.nanmax(tx1out),np.nanmean(tx1out),np.nanmax(tx2out),np.nanmean(tx2out),np.nanmax(tx3out),np.nanmean(tx3out),np.nanmax(tx4out),np.nanmean(tx4out),     np.nanmax(dx1out),np.nanmean(dx1out),np.nanmax(dx2out),np.nanmean(dx2out),np.nanmax(dx3out),np.nanmean(dx3out),np.nanmax(dx4out),np.nanmean(dx4out)


# In[15]:

train_fx = np.array([create_similarity_features3(train, train.iloc[i], i) for i in xrange(train.shape[0])])
dump('data/train1234_3_c1_r', train_fx)
test_fx = np.array([create_similarity_features3(train, test.iloc[i]) for i in xrange(test.shape[0])])
dump('data/test1234_3_c1_r', test_fx)


# # Antipattern

# In[16]:

def get_set_title_description(train, q, median_relevance, id=None):
    df = train[(train['query']==q) &
               (train['median_relevance']==median_relevance) &
               (train.index!=id)]
    r = set()
    for l in df['product_title'].values+df['product_description'].values:
        for w in process_str(l):
            r.add(w)
    return r


# In[17]:

def create_similarity_features4(train, row, id=None):
    tx = set(process_str(row['product_title']))

    s1 = get_set_title_description(train, row['query'], 1, id)
    s2 = get_set_title_description(train, row['query'], 2, id)
    s3 = get_set_title_description(train, row['query'], 3, id)
    s4 = get_set_title_description(train, row['query'], 4, id)

    return (
        len((tx & (s1)) - (s2 | s3 | s4)),
        len((tx & (s1 | s2)) - (s3 | s4)),
        len((tx & (s1 | s2 | s3)) - (s4)),

        len((tx & (s4)) - (s3 | s2 | s1)),
        len((tx & (s4 | s3)) - (s2 | s1)),
        len((tx & (s4 | s3 | s2)) - (s1)),
    )


# In[18]:

train_fx = np.array([create_similarity_features4(train, train.iloc[i], i) for i in xrange(train.shape[0])])
dump('data/train_anti_1234_c1_r', train_fx)
test_fx = np.array([create_similarity_features4(train, test.iloc[i]) for i in xrange(test.shape[0])])
dump('data/test_anti_1234_c1_r', test_fx)


# # Ngram similarity

# In[19]:

def get_ngrams(s, n):
    rez = []
    for i in range(0, len(s)-n):
        ss = s[i:i+n]
        rez.append(ss)
    return rez


# In[20]:

def get_str_for_query6(train, q, product_title, product_description, median_relevance, id=None):
    df = train[(train['query']==q) &
               (train['median_relevance']==median_relevance) &
               (train.index!=id)]
    tx = [e.lower().decode("utf8","ignore") for e in df['product_title'].values]
    dx = [e.lower().decode("utf8","ignore") for e in df['product_description'].values]

    return tx, dx


# In[21]:

def create_similarity_features6(train, row, id=None):
    clr_print(id, row['product_title'])

    tx1, dx1 = get_str_for_query6(train, row['query'], row['product_title'], row['product_description'], 1, id)
    tx2, dx2 = get_str_for_query6(train, row['query'], row['product_title'], row['product_description'], 2, id)
    tx3, dx3 = get_str_for_query6(train, row['query'], row['product_title'], row['product_description'], 3, id)
    tx4, dx4 = get_str_for_query6(train, row['query'], row['product_title'], row['product_description'], 4, id)

    our_tx = ' '.join(process_str(row['product_title']))
    our_dx = ' '.join(process_str(row['product_description']))

    rez = []
    for ngrams in [get_ngrams(our_tx, 2),
                   get_ngrams(our_tx, 4),
                   get_ngrams(our_tx, 6),
                   get_ngrams(our_tx, 8),
                  ]:
        for fx in [tx1, dx1, tx2, dx2, tx3, dx3, tx4, dx4]:
            c = 1
            for ngram in ngrams:
                for f in fx:
                    c+= f.count(ngram)
            c /= len(ngrams)+1
            rez.append(c)
    return rez


# In[22]:

#train_fx = np.array([create_similarity_features6(train, train.iloc[i], i) for i in xrange(1)])
train_fx = np.array([create_similarity_features6(train, train.iloc[i], i) for i in xrange(train.shape[0])])
dump('data/train1234_ngram_r', train_fx)
test_fx = np.array([create_similarity_features6(train, test.iloc[i]) for i in xrange(test.shape[0])])
dump('data/test1234_ngram_r', test_fx)


# In[ ]:




# # Main Product

# In[9]:

train['product_names'] = pd.read_csv('data/product_names.csv').fillna('')['product1']
test['product_names'] = pd.read_csv('data/product_names_test.csv').fillna('')['product1']


# In[10]:

def get_str_for_query7(train, q, median_relevance, id=None):
    df = train[(train['query']==q) &
               (train['median_relevance']==median_relevance) &
               (train.index!=id)]
    title_set = set()
    for e in df['product_names'].values:
        title_set |= set(process_str_replace(e))
    return title_set


# In[11]:

def create_similarity_features7(train, row, id=None):
    clr_print(id, row['product_title'])
    tx1 = get_str_for_query7(train, row['query'], 1, id)
    tx2 = get_str_for_query7(train, row['query'], 2, id)
    tx3 = get_str_for_query7(train, row['query'], 3, id)
    tx4 = get_str_for_query7(train, row['query'], 4, id)
    our_tx = set(process_str_replace(row['product_names']))

    return len(tx1 & our_tx), len(tx2 & our_tx), len(tx3 & our_tx), len(tx4 & our_tx)


# In[12]:

train_fx = np.array([create_similarity_features7(train, train.iloc[i], i) for i in xrange(train.shape[0])])
dump('data/train1234_7', train_fx)
test_fx = np.array([create_similarity_features7(train, test.iloc[i]) for i in xrange(test.shape[0])])
dump('data/test1234_7', test_fx)


# # Share

# In[23]:

X_1234 = load('data/train1234_c1_r')
X_test_1234 = load('data/test1234_c1_r')
np.savetxt('data/train1234.csv', X_1234, delimiter=',')
np.savetxt('data/test1234.csv', X_test_1234, delimiter=',')

X_1234_2 = load('data/train1234_2_c1_r')
X_test_1234_2 = load('data/test1234_2_c1_r')
np.savetxt('data/train1234_2.csv', X_1234_2, delimiter=',')
np.savetxt('data/test1234_2.csv', X_test_1234_2, delimiter=',')

X_1234_3 = load('data/train1234_3_c1_r')
X_test_1234_3 = load('data/test1234_3_c1_r')
np.savetxt('data/train1234_3.csv', X_1234_3, delimiter=',')
np.savetxt('data/test1234_3.csv', X_test_1234_3, delimiter=',')

X_anti_1234 = load('data/train_anti_1234_c1_r')
X_test_anti_1234 = load('data/test_anti_1234_c1_r')
np.savetxt('data/train_anti_1234.csv', X_anti_1234, delimiter=',')
np.savetxt('data/test_anti_1234.csv', X_test_anti_1234, delimiter=',')


# In[14]:

X_1234_7 = load('data/train1234_7')
X_test_1234_7 = load('data/test1234_7')
np.savetxt('data/train1234_7.csv', X_1234_7, delimiter=',')
np.savetxt('data/test1234_7.csv', X_test_1234_7, delimiter=',')
