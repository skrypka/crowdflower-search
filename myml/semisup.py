from __future__ import print_function

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SemiRegression(BaseEstimator, RegressorMixin):
    def __init__(self, base_clf, Xextra, Yextra):
        self.base_clf=base_clf
        self.Xextra=Xextra
        self.Yextra=Yextra

    def fit(self, X, y):
        XX = np.vstack((X,self.Xextra))
        YY = np.hstack((y,self.Yextra))
        print('Semi: {}+{}=>{} {}+{}=>{}'.format(X.shape, self.Xextra.shape, XX.shape,
                                                 y.shape, self.Yextra.shape, YY.shape))
        self.base_clf.fit(XX, YY)

    def predict(self, X):
        return self.base_clf.predict(X)
