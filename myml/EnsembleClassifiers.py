"""
    A wrapper for different ways of combining models

    Authors: Henning Sperr

    License: BSD-3 clause
"""
from __future__ import print_function
from itertools import izip
import random

from sklearn.base import ClassifierMixin
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss

import numpy as np
from scipy.optimize import minimize


class LinearModelCombination(ClassifierMixin):
    """
        Class that combines two models linearly.

        model1/2 : models to be combined
        metric : metric to minimize
    """

    def __init__(self, model1, model2, metric=log_loss):
        self.model1 = model1
        self.model2 = model2
        self.weight = None
        self.metric = metric

    def fit(self, X, y):
        scores = []
        pred1 = self.model1.predict_proba(X)
        pred2 = self.model2.predict_proba(X)

        for i in xrange(0, 101):
            weight = i / 100.
            scores.append(
                self.metric(y, weight * pred1 + (1 - weight) * pred2))
            # linear surface so if the score gets worse we can stop
            if len(scores) > 1 and scores[-1] > scores[-2]:
                break

        best_weight = np.argmin(scores)

        self.best_score = scores[best_weight]
        self.weight = best_weight / 100.

        return self

    def predict(self, X):
        if self.weight == None:
            raise Exception("Classifier seems to be not yet fitted")

        pred1 = self.model1.predict_proba(X) * self.weight
        pred2 = self.model2.predict_proba(X) * (1 - self.weight)
        return np.argmax(pred1 + pred2)

    def predict_proba(self, X):
        if self.weight == None:
            raise Exception("Classifier seems to be not yet fitted")

        pred1 = self.model1.predict_proba(X) * self.weight
        pred2 = self.model2.predict_proba(X) * (1 - self.weight)
        return pred1 + pred2

    def __str__(self):
        return ' '.join(["LM: ", str(self.model1), ' - ', str(self.model2), ' W: ', str(self.weight)])


class BestEnsembleWeights(ClassifierMixin):

    """
        Use scipys optimize package to find best weights for classifier combination.

        classifiers : list of classifiers
        prefit : if True classifiers will be assumed to be fit already and the data passed to
                 fit method will be fully used for finding best weights
        random_state : random seed
        verbose : print verbose output

    """

    def __init__(self, classifiers, num_iter=50, prefit=False, random_state=None, verbose=0):
        self.classifiers = classifiers
        self.prefit = prefit
        if random_state is None:
            self.random_state = random.randint(0, 10000)
        else:
            self.random_state = random_state
        self.verbose = verbose
        self.num_iter = num_iter

    def fit(self, X, y):
        if self.prefit:
            test_x, test_y = X, y
        else:
            sss = StratifiedShuffleSplit(
                y, n_iter=1, random_state=self.random_state)
            for train_index, test_index in sss:
                break

            train_x = X[train_index]
            train_y = y[train_index]

            test_x = X[test_index]
            test_y = y[test_index]

            for clf in self.classifiers:
                clf.fit(train_x, train_y)

        self._find_best_weights(test_x, test_y)

    def _find_best_weights(self, X, y):
        predictions = []
        for clf in self.classifiers:
            predictions.append(clf.predict_proba(X))

        if self.verbose:
            print('Individual LogLoss:')
            for mn, pred in enumerate(predictions):
                print("Model {model_number}:{log_loss}".format(model_number=mn,
                                                               log_loss=log_loss(y, pred)))

        def log_loss_func(weights):
            ''' scipy minimize will pass the weights as a numpy array '''
            final_prediction = 0
            for weight, prediction in izip(weights, predictions):
                final_prediction += weight * prediction

            return log_loss(y, final_prediction)

        # the algorithms need a starting value, right not we chose 0.5 for all weights
        # its better to choose many random starting points and run minimize a
        # few times
        starting_values = np.ones(len(predictions)) / (len(predictions))
        # This sets the bounds on the weights, between 0 and 1
        bounds = tuple((0, 1) for w in starting_values)

        # adding constraints  and a different solver as suggested by user 16universes
        # https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

        res = minimize(log_loss_func, starting_values,
                       method='SLSQP', bounds=bounds, constraints=cons)

        self.best_score = res['fun']
        self.best_weights = res['x']

        for i in xrange(self.num_iter):
            starting_values = np.random.uniform(0,1,size=len(predictions))

            res = minimize(log_loss_func, starting_values,
                           method='SLSQP', bounds=bounds, constraints=cons)

            if res['fun']<self.best_score:
                self.best_score = res['fun']
                self.best_weights = res['x']

                if self.verbose:
                    print('')
                    print('Update Ensamble Score: {best_score}'.format(best_score=res['fun']))
                    print('Update Best Weights: {weights}'.format(weights=self.best_weights))

        if self.verbose:
            print('Ensamble Score: {best_score}'.format(best_score=self.best_score))
            print('Best Weights: {weights}'.format(weights=self.best_weights))

    def predict_proba(self, X):
        prediction = 0
        for weight, clf in izip(self.best_weights, self.classifiers):
            prediction += weight * clf.predict_proba(X)
        return prediction

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class LogisticModelCombination(ClassifierMixin):

    """
        Combine multiple models using a Logistic Regression
    """

    def __init__(self, classifiers, cv_folds=1, use_original_features=False, random_state=None, verbose=0):
        self.classifiers = classifiers
        self.cv_folds = cv_folds
        self.use_original_features = use_original_features
        self.logistic = LogisticRegressionCV(
            Cs=[10, 1, 0.1, 0.01, 0.001], refit=True)

        if random_state is None:
            self.random_state = random.randint(0, 10000)
        else:
            self.random_state = random_state

    def fit(self, X, y):
        sss = StratifiedShuffleSplit(
            y, n_iter=self.cv_folds, random_state=self.random_state)
        for train_index, test_index in sss:
            train_x = X[train_index]
            train_y = y[train_index]

            test_x = X[test_index]
            test_y = y[test_index]

            self._fit_logistic(train_x, train_y)

    def _fit_logistic(self, X, y):
        pred_X = self.convert_data(X)
        self.logistic.fit(pred_X, y)
        return self

    def convert_data(self, X):
        preds = []
        for i, clf in enumerate(self.classifiers):
            class_proba = clf.predict(X)
            preds.append(class_proba)
        pred_X = np.vstack(preds).T

        if self.use_original_features:
            pred_X = np.concatenate([X, pred_X], axis=1)
        return pred_X

    def predict_proba(self, X):
        pred_X = self.convert_data(X)
        return self.logistic.predict_proba(pred_X)
