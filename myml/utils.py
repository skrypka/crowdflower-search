from __future__ import print_function

import os
import sys
import csv
import time
import subprocess

import numpy as np

from files import create_directory

def beep():
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.2, 1000))

def clr_print(*args):
    from IPython.display import clear_output
    clear_output(wait=True)
    print(*args)
    sys.stdout.flush()

def print_experiment(name):
    os.system('git commit -a -m "auto"')
    p = subprocess.Popen(['git', '--no-pager', 'log', '-n', '1', '--pretty=format:"%H"'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.communicate()[0]
    if out:
        print("Experiment: %s => %s" % (name, out))
        return out
    else:
        print("Experiment: %s => No Git" % name)

def do_experiment(name, clf, X, X_test, y, out_fold_idx, scorer):
    git = print_experiment(name)
    d = 'experiment/'+name+'/'
    create_directory(d)
    out_fold_p = np.zeros(y.shape)
    scores = []
    for i, (train_idx, val_idx) in enumerate(out_fold_idx):
        print("Out Fold:", i)
        Xt = X[train_idx]
        Xv = X[val_idx]
        yt = y[train_idx]
        yv = y[val_idx]
        clf.fit(Xt, yt)
        px = clf.predict(Xv)
        out_fold_p[val_idx] = px
        s = scorer(yv, px)
        scores.append(s)
        print("Score:", s)
    np.savetxt(d + 'out_fold.csv', out_fold_p, delimiter=',')

    print("Test pred")
    clf.fit(X, y)
    px = clf.predict(X_test)
    np.savetxt(d + 'test.csv', px, delimiter=',')

    lb_score = ''
    scores = np.array(scores)

    print([time.time(), name, git, scores.mean(), scores.std(), lb_score])
    with open("experiments.txt", "a") as f:
        swriter = csv.writer(f, delimiter=',')
        swriter.writerow([name, git, scores.mean(), scores.std(), lb_score])

    return px
