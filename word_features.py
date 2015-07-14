import numpy as np
import pandas as pd

from myml.files import dump
from utils import process_str

train = pd.read_csv("input/train.csv").fillna("")
test = pd.read_csv("input/test.csv").fillna("")

womanx = set(['woman', 'women', 'ladi', 'girl'])
manx = set(['man', 'boy', 'men'])

def is_man(s):
    wx = set(process_str(s))
    return len(manx & wx)>0

def is_woman(s):
    wx = set(process_str(s))
    return len(womanx & wx)>0

def wm_opposite(row):
    m_q = is_man(row['query'])
    w_q = is_woman(row['query'])
    m_t = is_man(row['product_title'])
    w_t = is_woman(row['product_title'])
    if m_q==True and w_t==True and m_t==False:
        return 1
    elif w_q==True and m_t==True and w_t==False:
        return 1
    else:
        return 0

train_wm = train.apply(wm_opposite, axis=1).reshape(-1,1)
test_wm = test.apply(wm_opposite, axis=1).reshape(-1,1)
train_wm.sum(), test_wm.sum(), train_wm.shape, test_wm.shape

dump('data/wm_features', (train_wm, test_wm))

np.savetxt('data/train_wm.csv', train_wm, delimiter=',')
np.savetxt('data/test_wm.csv', test_wm, delimiter=',')
