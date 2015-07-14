from __future__ import print_function

from sklearn.base import BaseEstimator, RegressorMixin, clone

class SimpleBaggingRegression(BaseEstimator, RegressorMixin):
    def __init__(self, clf, nb_run=5):
        self.clf = clf
        self.nb_run = nb_run

    def fit(self, X, y):
        self.clfx = []

        for i in xrange(self.nb_run):
            clf = clone(self.clf)
            clf.fit(X, y)
            self.clfx.append(clf)

    def predict(self, X):
        result = 0
        for clf in self.clfx:
            result += clf.predict(X)
        return result / self.nb_run
