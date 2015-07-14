from __future__ import print_function

import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class M2Mensemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf):
        self.base_clf = base_clf

    def fit(self, SX, y):
        self.classes_ = np.unique(y)
        self.clfx = {}
        sidx = SX[:, 0]
        X = SX[:, 1:]

        uniq_sidx = set(sidx)
        print("Building {0} nb base clf".format(len(uniq_sidx)))
        for sid in uniq_sidx:
            clf = clone(self.base_clf)
            train_idx = sidx==sid
            clf.fit(X[train_idx], y[train_idx])
            self.clfx[sid] = clf
        return self

    def predict(self, SX):
        result = []
        for i in range(SX.shape[0]):
            sid = SX[i,0]
            X = SX[i, 1:]
            result.append( self.clfx[sid].predict(X.reshape((1, -1))) )
        return np.array(result).flatten()

class M2MRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_clf):
        self.base_clf = base_clf

    def fit(self, SX, y):
        self.clfx = {}
        sidx = SX[:, 0]
        X = SX[:, 1:]

        uniq_sidx = set(sidx)
        print("Building {0} nb base clf".format(len(uniq_sidx)))
        for i, sid in enumerate(uniq_sidx):
            clf = clone(self.base_clf)
            train_idx = sidx==sid
            clf.fit(X[train_idx], y[train_idx])
            self.clfx[sid] = clf
        return self

    def predict(self, SX):
        result = []
        for i in range(SX.shape[0]):
            sid = SX[i,0]
            X = SX[i, 1:]
            result.append( self.clfx[sid].predict(X.reshape((1, -1))) )
        return np.array(result).flatten()

class UnsuperM2Mensemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf):
        self.base_clf = base_clf

    def fit(self, SX, y=None):
        self.clfx = {}
        sidx = SX[:, 0]
        X = SX[:, 1:]

        uniq_sidx = set(sidx)
        print("Building {0} nb base clf".format(len(uniq_sidx)))
        for sid in uniq_sidx:
            clf = clone(self.base_clf)
            train_idx = sidx==sid
            clf.fit(X[train_idx])
            self.clfx[sid] = clf
        return self

    def predict(self, SX):
        result = []
        for i in range(SX.shape[0]):
            sid = SX[i,0]
            X = SX[i, 1:]
            result.append( self.clfx[sid].predict(X) )
        return np.array(result)
