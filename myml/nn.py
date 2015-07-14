from __future__ import print_function

import numpy as np
from keras.models import Sequential
import keras.optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class NnClassification(BaseEstimator, ClassifierMixin):
    def __init__(self, apply_standart_scaling=True,
                 dropx=[0.2, 0.5, 0.3], nb_neuronx=[1024, 512], nb_epoch=20):
        self.apply_standart_scaling = apply_standart_scaling
        self.dropx = dropx
        self.nb_neuronx = nb_neuronx
        self.nb_epoch = nb_epoch

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        nb_classes = len(self.classes_)
        nb_features = X.shape[1]

        y = np_utils.to_categorical(y)

        self.standart_scaling = StandardScaler() if self.apply_standart_scaling else None
        if self.standart_scaling:
            X = self.standart_scaling.fit_transform(X)

        model = Sequential()
        model.add(Dropout(self.dropx[0]))
        model.add(Dense(nb_features, self.nb_neuronx[0], init='glorot_uniform'))
        model.add(PReLU((self.nb_neuronx[0],)))
        model.add(BatchNormalization((self.nb_neuronx[0],)))
        model.add(Dropout(self.dropx[1]))

        model.add(Dense(self.nb_neuronx[0], self.nb_neuronx[1], init='glorot_uniform'))
        model.add(PReLU((self.nb_neuronx[1],)))
        model.add(BatchNormalization((self.nb_neuronx[1],)))
        model.add(Dropout(self.dropx[2]))

        model.add(Dense(self.nb_neuronx[1], nb_classes, init='glorot_uniform'))
        model.add(Activation('softmax'))

        optz = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optz)
        model.fit(X, y, batch_size=32, nb_epoch=self.nb_epoch,
                  #validation_data=None,
                  validation_split=0.2,
                  verbose=1)

        self.model = model

    def predict_proba(self, X):
        if self.standart_scaling:
            X = self.standart_scaling.transform(X)
        return self.model.predict_proba(X, verbose=0)

class NnRegression(BaseEstimator, RegressorMixin):
    def __init__(self, apply_standart_scaling=True,
                 dropx=[0.2, 0.5, 0.5], nb_neuronx=[1024, 512], nb_epoch=20, validation_split=0.,
                 verbose=1, callbacks=[], loss='mean_squared_error', batch_size=32):
        self.apply_standart_scaling = apply_standart_scaling
        self.dropx = dropx
        self.callbacks = callbacks
        self.nb_neuronx = nb_neuronx
        self.nb_epoch = nb_epoch
        self.validation_split = validation_split
        self.verbose = verbose
        self.loss = loss
        self.batch_size = batch_size

    def fit(self, X, y):
        nb_features = X.shape[1]
        if len(y.shape)==1:
            nb_out = 1
        else:
            nb_out = y.shape[1]
        self.standart_scaling = StandardScaler() if self.apply_standart_scaling else None

        if self.standart_scaling:
            X = self.standart_scaling.fit_transform(X)

        model = Sequential()
        model.add(Dropout(self.dropx[0]))

        model.add(Dense(nb_features, self.nb_neuronx[0], init='glorot_uniform'))
        model.add(PReLU((self.nb_neuronx[0],)))
        model.add(BatchNormalization((self.nb_neuronx[0],)))
        model.add(Dropout(self.dropx[1]))

        model.add(Dense(self.nb_neuronx[0], self.nb_neuronx[1], init='glorot_uniform'))
        model.add(PReLU((self.nb_neuronx[1],)))
        model.add(BatchNormalization((self.nb_neuronx[1],)))
        model.add(Dropout(self.dropx[2]))

        if len(self.nb_neuronx)>2:
            model.add(Dense(self.nb_neuronx[1], self.nb_neuronx[2], init='glorot_uniform'))
            model.add(PReLU((self.nb_neuronx[2],)))
            model.add(BatchNormalization((self.nb_neuronx[2],)))
            model.add(Dropout(self.dropx[3]))


        model.add(Dense(self.nb_neuronx[-1], nb_out, init='glorot_uniform'))

        nn_verbose = 1 if self.verbose>0 else 0
        optz = keras.optimizers.Adam()
        model.compile(loss=self.loss, optimizer=optz)
        model.fit(X, y, batch_size=self.batch_size, callbacks=self.callbacks,
                  nb_epoch=self.nb_epoch, validation_split=self.validation_split, verbose=nn_verbose)

        self.model = model

    def predict(self, X):
        if self.standart_scaling:
            X = self.standart_scaling.transform(X)
        return self.model.predict_proba(X, verbose=0)
