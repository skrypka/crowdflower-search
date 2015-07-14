import numpy as np

from sklearn.base import TransformerMixin

class FunctionTransformer(TransformerMixin):
    """
        Transformer that takes arbitrary function for transformation as input

        Parameters
        ----------

        func : Function
            arbitrary function that gets applied to each element of the input.
    """
    def __init__(self, func):
        self.func=func

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.func(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

class LogTransformer(FunctionTransformer):
    """
        Logarithm Transformer that applies log(1+x) to the input
    """
    def __init__(self):
        super(self.__class__, self).__init__(lambda x: np.log1p(x.astype(np.float32)))

class PowerTransformer(FunctionTransformer):
    """
        Power Transformer that applies x^n to the input
    """
    def __init__(self, power):
        super(self.__class__, self).__init__(lambda x:np.power(x, power))

if __name__ == '__main__':
    pwr = PowerTransformer(2)
    arr = np.array([1,2,3,4])
    print pwr.transform(arr)
