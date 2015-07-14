from __future__ import print_function
from itertools import product

import numpy as np

def cv_all_features(fx, y, callback):
    fx_count = len(fx)
    print('FX shape:', ', '.join([str(f.shape) for f in fx]))

    cx = [c for c in list(product([True, False], repeat=fx_count)) if any(c)]
    for c in cx:
        print('C', c)
        X = np.hstack([b for b,f in zip(c, fx) if b])
        callback(X, y)
