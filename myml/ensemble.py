from __future__ import print_function
from itertools import izip

import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin


class BestEnsembleWeights():
    def __init__(self, f=None):
        self.f = f

    def _apply_weights(self, weights, pxx):
        final_prediction = 0
        for weight, px in izip(weights, pxx):
            final_prediction += weight * px
        return final_prediction

    def fit(self, pxx, tryis=100):
        print("Models:", [self.f(px) for px in pxx])

        def weights_loss_func(weights):
            return self.f(self._apply_weights(weights, pxx))

        starting_values = np.ones(len(pxx)) / (len(pxx))
        bounds = tuple((0, 1) for w in starting_values)
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

        res = minimize(weights_loss_func, starting_values,
                       method='SLSQP', bounds=bounds, constraints=cons)

        self.best_score = res['fun']
        self.best_weights = res['x']

        for i in xrange(tryis):
            starting_values = np.random.uniform(0,1,size=len(pxx))
            res = minimize(weights_loss_func, starting_values,
                           method='SLSQP', bounds=bounds, constraints=cons)

            if res['fun']<self.best_score and res['success']:
                self.best_score = res['fun']
                self.best_weights = res['x']

        print('Ensamble: {best_score} = {weights}'.format(
            best_score=self.best_score, weights=self.best_weights))
        return self.best_score

    def transform(self, pxx):
        return self._apply_weights(self.best_weights, pxx)

class Blend(BaseEstimator, ClassifierMixin):
    def __init__(self, clfx, scoring, cv=10, is_max=True, random_state=42):
        self.cv = cv
        self.clfx = clfx
        self.is_max = is_max
        self.scoring = scoring
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        self.best_weights = 0

        skx = list(StratifiedKFold(y, self.cv, random_state=self.random_state))

        score = 0
        for i, (itrain, itest) in enumerate(skx):
            print("Blend fold", i)
            X_train = X[itrain]
            y_train = y[itrain]
            X_test = X[itest]
            y_test = y[itest]
            [clf.fit(X_train, y_train) for clf in self.clfx]
            pxx_test = [clf.predict_proba(X_test) for clf in self.clfx]

            ensemble_weight = BestEnsembleWeights(lambda px: self.scoring(y_test, px))
            score += ensemble_weight.fit(pxx_test)

            self.best_weights += ensemble_weight.best_weights

        print("Train on full dataset")
        [clf.fit(X, y) for clf in self.clfx]
        self.best_ensemble = BestEnsembleWeights()
        self.best_ensemble.best_weights = self.best_weights / np.sum(self.best_weights)
        print("Final:", score/self.cv, self.best_ensemble.best_weights)

    def predict_proba(self, X):
        pxx = [clf.predict_proba(X) for clf in self.clfx]
        return self.best_ensemble.transform(pxx)
