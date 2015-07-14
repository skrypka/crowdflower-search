from __future__ import division

from math import sqrt, log10

def kaggle_points(place, teammates, users):
    return (100000/sqrt(teammates))*(place**(-0.75))*(log10(1+log10(users)))
