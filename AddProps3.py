# coding: utf-8

# In[1]:


from __future__ import division

import numpy as np
import pandas as pd

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from myml.files import dump
from myml.pipe import ItemSelectorByIndex
from utils import process_str_replace_str, process_str_replace


# In[2]:

train = pd.read_csv("input/train.csv").fillna("")
test = pd.read_csv("input/test.csv").fillna("")
train = train.drop(['id'], axis=1)
test = test.drop(['id'], axis=1)


# In[3]:

y = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

# In[5]:

X = np.hstack((
    np.array(list(train.apply(lambda x:'%s' % process_str_replace_str(x['query']),axis=1))).reshape(-1, 1),
    np.array(list(train.apply(lambda x:'%s' % process_str_replace_str(x['product_title']),axis=1))).reshape(-1, 1),
    np.array(list(train.apply(lambda x:'%s' % process_str_replace_str(x['product_description']),axis=1))).reshape(-1, 1),
))
X_test = np.hstack((
    np.array(list(test.apply(lambda x:'%s' % process_str_replace_str(x['query']),axis=1))).reshape(-1, 1),
    np.array(list(test.apply(lambda x:'%s' % process_str_replace_str(x['product_title']),axis=1))).reshape(-1, 1),
    np.array(list(test.apply(lambda x:'%s' % process_str_replace_str(x['product_description']),axis=1))).reshape(-1, 1),
))


# In[6]:

trf = Pipeline([
    ('union', FeatureUnion([
        ('query', Pipeline([
            ('selector', ItemSelectorByIndex(0)),
            ('tfv', TfidfVectorizer(min_df=3, max_features=None,
                                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                                    ngram_range=(1, 6), use_idf=1,smooth_idf=1,sublinear_tf=1,
                                    stop_words='english')),
        ])),
        ('product_title', Pipeline([
            ('selector', ItemSelectorByIndex(1)),
            ('tfv', TfidfVectorizer(min_df=3, max_features=None,
                                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                                    ngram_range=(1, 6), use_idf=1,smooth_idf=1,sublinear_tf=1,
                                    stop_words='english')),
        ])),
        ('product_description', Pipeline([
            ('selector', ItemSelectorByIndex(2)),
            ('tfv', TfidfVectorizer(min_df=3, max_features=None,
                                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                                    ngram_range=(1, 6), use_idf=1,smooth_idf=1,sublinear_tf=1,
                                    stop_words='english')),
        ])),
    ])),
    ('svd', TruncatedSVD(n_components=250, random_state=45)),
    #('scl', StandardScaler())
])


# In[7]:

X2 = trf.fit_transform(X)
X2.shape


# In[8]:

X2_test = trf.transform(X_test)
X2_test.shape


# In[9]:

#dump("data/XXtestY250_clean", (X2, X2_test, y))
#dump("data/XXtestY250_am_t0", (X2, X2_test, y))

dump("data/XXtestY250_r2td", (X2, X2_test, y))

#dump("data/XXtestY250_ver1", (X2, X2_test, y))
#dump("data/XXtestY250_clean_description", (X2, X2_test, y))
#X2, X2_test, y = load("data/XXtestY250")
np.savetxt("data/X250_3.csv", X2, delimiter=",")
np.savetxt("data/X250_test_3.csv", X2_test, delimiter=",")
