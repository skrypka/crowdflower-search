import re
from collections import defaultdict

import pandas as pd

from myml.files import dump
from utils_pre import stem_one, process_str_replace

train = pd.read_csv("input/train.csv").fillna("")
test = pd.read_csv("input/test.csv").fillna("")
bdf = pd.concat((train, test))
bdf.shape

wx = defaultdict(int)
for lx in [bdf['query'].unique(),
          bdf['product_title'].unique(),
          bdf['product_description'].unique()]:
    for l in lx:
        for w in process_str_replace(l):
            wx[w] += 1
wx = dict(wx)

rwx = {}
for w,v in wx.iteritems():
    if re.search("[0-9]+", w):
        continue
    if len(w)>7:
        for i in range(3, len(w)-2):
            pf = -1
            w1 = w[:i]
            w2 = w[i:]
            f1 = wx.get(w1, 0)
            f2 = wx.get(w2, 0)
            if f1>10 and f2>10 and (f1+f2)>pf:
                w1 = stem_one(w1)
                w2 = stem_one(w2)
                rwx[w] = (w1, w2)
                pf = f1+f2

dump('data/word_to_2_replacer', rwx)


numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
           'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
          'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
           'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
replace = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20,
          30, 40, 50, 60, 70, 80, 90]
replace = [str(n) for n in replace]

numbers = [stem_one(w) for w in numbers]

num_to_num = dict(zip(numbers, replace))
dump('data/num_to_num', num_to_num)
