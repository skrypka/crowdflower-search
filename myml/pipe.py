from __future__ import print_function

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin


class ItemSelectorByIndex(BaseEstimator, TransformerMixin):
    def __init__(self, i1, i2=None):
        self.i1 = i1
        self.i2 = i2
    def fit(self, x, y=None):
        return self
    def transform(self, data):
        if self.i2:
            return data[:, self.i1:self.i2]
        else:
            return data[:, self.i1]

class RegressionTransformer(BaseEstimator, RegressorMixin):
    def __init__(self, base_clf, ltrain, ltest):
        self.base_clf = base_clf
        self.ltrain = ltrain
        self.ltest = ltest
    def fit(self, X, y):
        X, y = self.ltrain(X, y)
        self.base_clf.fit(X, y)
        return self
    def predict(self, X):
        X = self.ltest(X)
        return self.base_clf.predict(X)
