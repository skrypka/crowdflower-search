from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor

from myml.nn import NnRegression
from myml.files import load

from utils import cv_generate

X2, X2_test, y = load("data/XXtestY250_r2td")

X_extra = pd.read_csv('data/ngramMatch_07.csv').values
X_extra_test = pd.read_csv('data/ngramMatch_test_07.csv').values
X_1234 = load('data/train1234_c1_r')
X_test_1234 = load('data/test1234_c1_r')
X_1234_2 = load('data/train1234_2_c1_r')
X_test_1234_2 = load('data/test1234_2_c1_r')
X_1234_3 = load('data/train1234_3_c1_r')
X_test_1234_3 = load('data/test1234_3_c1_r')
X_anti_1234 = load('data/train_anti_1234_c1_r')
X_test_anti_1234 = load('data/test_anti_1234_c1_r')

X_union_f, X_test_union_f = load('data/XXunion_f_norm')

train_test_alt = pd.read_csv('data/alt_query_features_train_and_test_v01.csv').values
train_alt = train_test_alt[:10158]
test_alt = train_test_alt[10158:]

train_ngram = load('data/train1234_ngram_r')
test_ngram = load('data/test1234_ngram_r')

train_wm, test_wm = load('data/wm_features')

# psim_train = pd.read_csv('data/product_simscore_train.csv').values
# psim_test = pd.read_csv('data/product_simscore_test.csv').values
psim_train = pd.read_csv('data/product_simscore_train.csv').values
psim_test = pd.read_csv('data/product_simscore_test.csv').values

prod1234_train = load('data/train1234_7')
prod1234_test = load('data/test1234_7')

tf_train = pd.read_csv('data/Tf_idf_train.csv', header=None).values
tf_test = pd.read_csv('data/Tf_idf_test.csv', header=None).values

X = np.hstack((X2, X_extra, X_1234, X_1234_2, X_1234_3, X_union_f, X_anti_1234, train_alt, train_ngram, train_wm, psim_train, prod1234_train, tf_train))
X_test = np.hstack((X2_test, X_extra_test, X_test_1234, X_test_1234_2, X_test_1234_3, X_test_union_f, X_test_anti_1234, test_alt, test_ngram, test_wm, psim_test, prod1234_test, tf_test))

np.random.seed(44)
nn = NnRegression(nb_epoch=40, dropx=[0.3, 0.5, 0.5], nb_neuronx=[1024, 512], validation_split=0., verbose=0)
b = BaggingRegressor(nn, 10, bootstrap=False, verbose=0, random_state=1)

cv_generate(b, "ann_250_tfidf", X, y, X_test, generate_test=True, xempty=None)
