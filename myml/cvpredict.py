from __future__ import print_function

import numpy as np
from sklearn.cross_validation import StratifiedKFold

class CVPredictorRegression():
    def __init__(self, clf, cv=5, random_state=None, scoring=None):
        self.clf = clf
        self.cv = cv
        self.random_state = random_state
        self.scoring = scoring

    def fit(self, X, y):
        print('Warning: use fit_transform')
        self.clf.fit(X, y)
        return self

    def fit_transform(self, X, y):
        result = np.zeros(y.shape)
        skf = StratifiedKFold(y, n_folds=self.cv, random_state=self.random_state, shuffle=True)
        for i, (itrain, itest) in enumerate(skf):
            X_train = X[itrain]
            y_train = y[itrain]
            X_test = X[itest]
            y_test = y[itest]

            self.clf.fit(X_train, y_train)

            y_pred = self.clf.predict(X_test)
            result[itest] = y_pred

            score = self.scoring(y_test, y_pred) if self.scoring else ''
            print("Generated k-fold", i, score)
        print('Fitting on full database')
        self.clf.fit(X, y)
        return result

    def predict(self, X):
        return self.clf.predict(X)
