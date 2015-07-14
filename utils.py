from __future__ import print_function, division
import re
import os

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

from myml.metrics import quadratic_weighted_kappa
from myml.files import load

def make_submission(preds, path='sx.csv'):
    print("Making submissions", preds.shape)
    test = pd.read_csv('data/test.csv').fillna('')
    idx = test.id.values.astype(int)
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv(path, index=False)

def make_in_limit(px):
    pxi = np.array(np.rint(px), dtype=int)
    pxi[pxi<1] = 1
    pxi[pxi>4] = 4
    return pxi

def limited_kappa(original, px):
    return quadratic_weighted_kappa(original, make_in_limit(px))

l_kappa_scorer = metrics.make_scorer(limited_kappa, greater_is_better=True)


import enchant
from nltk.metrics import edit_distance
class SpellingReplacer(object):
    def __init__(self, dict_name = 'en_US', max_dist = 2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = 2

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)

        print(suggestions)
        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word
def spell_check(word_list):
    checked_list = []
    for item in word_list:
        replacer = SpellingReplacer()
        r = replacer.replace(item)
        checked_list.append(r)
    return checked_list



#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','style','color','px','margin','left', 'right','font','solid','0px']
stop_words += list(text.ENGLISH_STOP_WORDS)
for i in range(len(stop_words)):
    stop_words[i]=stemmer.stem(stop_words[i])
stop_words = set(stop_words)

def stem_one(w):
    return stemmer.stem(w)

def process_str(s):
    s = s.lower()
    #s = re.sub("- [a-z\/]+$", '', s)
    s = " ".join([z for z in BeautifulSoup(s).get_text(" ").split(" ")])
    s = re.sub("[^a-z0-9]", " ", s)
    wx = [stemmer.stem(z) for z in s.split(" ") if z]
    return wx

long_word_replace = load('data/word_to_2_replacer')

query_auto_correct = load('data/query_auto_correct')
num_to_num = load('data/num_to_num')

def process_str_replace(s):
    s = s.lower()
    if s in query_auto_correct:
        #print(s, query_auto_correct[s])
        s = query_auto_correct[s]
    #s = re.sub("- [a-z\/]+$", '', s)
    s = " ".join([z for z in BeautifulSoup(s).get_text(" ").split(" ")])
    #s = re.sub("([/ -]{1,}(purple|red|blue|white|black|green|pink|yellow|grey|silver|clear|small|large|medium|m|X|2X|xl|navy|aqua|brown|brown leather|sealed|nib|new)[ ]?){1,}$", " ", s)
    s = re.sub("[^a-z0-9]", " ", s)
    sx = re.split(r'( |\b\d+|\d+\b)', s)
    sx = [w.strip() for w in sx]
    wx = [stemmer.stem(z) for z in sx if z]
    rez_wx = []
    for w in wx:
        w = num_to_num.get(w, w)

        if long_word_replace.get(w):
            w1, w2 = long_word_replace.get(w)
            #rez_wx.append(w)
            rez_wx.append(w1)
            rez_wx.append(w2)
        else:
            rez_wx.append(w)
    return rez_wx

def process_str_replace_str(s):
    return ' '.join(process_str_replace(s))

def process_str_str(s):
    return ' '.join(process_str(s))

def bs_str(s):
    return BeautifulSoup(s.lower()).get_text(" ")

def round_with_x(pxold, x):
    px = []
    for p in pxold:
        if p<x[0]:
            px.append(1)
        elif p<x[1]:
            px.append(2)
        elif p<x[2]:
            px.append(3)
        else:
            px.append(4)
    return np.array(px)

rek_to_rat_helper = {
    (1, 0.0): (1, 1, 1),
    (1, 0.471405): (1, 1, 2),
    (1, 0.942809): (1, 1, 3),
    (1, 1.41421): (1, 1, 4),

    (2, 0.471405): (1, 2, 2),
    (2, 0.816497): (1, 2, 3),
    (2, 1.24722): (1, 2, 4),
    (2, 0.0): (2, 2, 2),
    (2, 0.471405): (2, 2, 3),
    (2, 0.942809): (2, 2, 4),

    (3, 0.942809): (1, 3, 3),
    (3, 1.24722): (1, 3, 4),
    (3, 0.471405): (2, 3, 3),
    (3, 0.816497): (2, 3, 4),
    (3, 0.0): (3, 3, 3),
    (3, 0.471405): (3, 3, 4),

    (4, 0.942809): (2, 4, 4),
    (4, 1.41421): (1, 4, 4),
    (4, 0.471405): (3, 4, 4),
    (4, 0.0): (4, 4, 4)
}

def rel_to_rating(m, s):
    diff = 1000000000
    rez = None
    for (mi, si), v in rek_to_rat_helper.items():
        if mi!=m:
            continue
        cdiff = abs(s-si)
        if cdiff < diff:
            diff = cdiff
            rez = v
    return rez

def cv_generate(clf, prefix, X, y, Xtest, generate_test=True, xempty=None):
    df = pd.read_csv('data/cv_5fold_keys.csv')
    r1x = []
    r2x = []
    for k in df['fold'].unique():
        train_idx = df[df['fold']!=k]['index'].values - 1
        test_idx = df[df['fold']==k]['index'].values - 1
        print('Fold', k)
        clf.fit(X[train_idx], y[train_idx])
        px = clf.predict(X[test_idx])

        r1 = limited_kappa(y[test_idx], px)
        r2 = limited_kappa(y[test_idx], round_with_x(px, [1.9, 2.7, 3.5]))

        r1x.append(r1)
        r2x.append(r2)
        print("R:", r1, r2)

        np.savetxt('models/'+prefix+'-' + str(k) + '.csv', px, delimiter=',')
        if xempty is not None:
            z = np.zeros(xempty.flatten().shape)
            z[test_idx] = 1
            idx0 = (z==1) & (xempty.flatten()==0)
            idx1 = (z==1) & (xempty.flatten()==1)
            px = clf.predict(X[idx0])
            print("Description", limited_kappa(y[idx0], px),
                  limited_kappa(y[idx0], round_with_x(px, [1.9, 2.7, 3.5])))
            px = clf.predict(X[idx1])
            print("Empty", limited_kappa(y[idx1], px),
                  limited_kappa(y[idx1], round_with_x(px, [1.9, 2.7, 3.5])))

    r1x = np.array(r1x)
    r2x = np.array(r2x)
    print("RR:", r1x.mean(), r1x.std(), r2x.mean(), r2x.std())

    if generate_test:
        clf.fit(X, y)
        px = clf.predict(Xtest)
        print("Xtest", Xtest.shape, px.shape)
        np.savetxt('models/'+prefix+'-test.csv', px, delimiter=',')
        #make_submission(round_with_x(px, [1.9, 2.7, 3.5]))

def cv_generate2(name, clf, X, y, X_test):
    directory = 'data/E_'+ name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    test = pd.read_csv('data/test.csv')
    idx = test.id.values.astype(int)

    kappa_list = []
    for j in range (1,6):
        indices_file = 'data/fold3/set'+str(j)+'_3foldcv_keys.csv'
        indices = pd.read_csv(indices_file)

        for i in range (1,4):
            X_new = X[(indices.index[indices['x']==i]).values,:]
            y_new = y[(indices.index[indices['x']==i]).values]
            X_cv = X[(indices.index[indices['x']!=i]).values,:]
            y_cv = y[(indices.index[indices['x']!=i]).values]
            clf.fit(X_new, y_new)

            y_pred = clf.predict(X_cv)
            kappa_list.append(limited_kappa(y_cv,y_pred))
            print('set ',j, ' fold ', i, limited_kappa(y_cv,y_pred))

            cv_sub = pd.DataFrame({ "prediction": y_pred.flatten()})
            cv_sub.to_csv(directory+'sub_set'+str(j)+'_cv_fold_'+str(i)+'.csv', index=False)

    print('kappa mean', np.array(kappa_list).mean())

    clf.fit(X,y)
    preds = clf.predict(X_test)

    submission = pd.DataFrame({"id": idx, "prediction": preds.flatten()})
    submission.to_csv(directory+"submission.csv", index=False)

def get_url(url):
    import urllib2

    opener = urllib2.build_opener()
    urllib2.install_opener(opener)

    headers = {
        'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)',
        'Accept': 'text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5',
        'Accept-Language': 'fr-fr,en-us;q=0.7,en;q=0.3',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7'
    }
    #None = GET; set values to use POST
    req = urllib2.Request(url, None, headers)

    response = urllib2.urlopen(req).read()
    return response

def load_cv5(prefix):
    px = None
    df = pd.read_csv('data/cv_5fold_keys.csv')
    pred_idx = df['index'].values - 1
    for k in df['fold'].unique():
        loaded = np.loadtxt('data/'+prefix+'-' + str(k) + '.csv')
        if px is not None:
            px = np.hstack((px, loaded))
        else:
            px = loaded
    idx = np.arange(px.shape[0])
    idx = idx[pred_idx].argsort()
    return px[idx], np.loadtxt('data/'+prefix+'-test.csv')
