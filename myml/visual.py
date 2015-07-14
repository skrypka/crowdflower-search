import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

def plot_feature_importance(clf):
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(importances)), indices)
    plt.xlim([-1, len(importances)])
    plt.show()

def plot_tsne(X, Y, init='random'):
    """init {random, pca}"""
    tsne = manifold.TSNE(init=init)
    Ytsne = tsne.fit_transform(X)
    plt.figure(figsize=(12, 6))
    plt.scatter(Ytsne[:, 0], Ytsne[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.title("t-SNE")
    plt.axis('tight')
    plt.show()

def plot_roc(y_test, px):
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, px)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")


def plotConfMap(y_test, y_pred, classes=[], relative=False):
    """
    # Plot a confusion matrix
    """
    print(classification_report(y_test, y_pred))
    confMat = confusion_matrix(y_test, y_pred)
    print(confMat)

    width = len(confMat)
    height = len(confMat[0])

    oldParams = rcParams['figure.figsize']
    rcParams['figure.figsize'] = width, height

    fig = plt.figure()
    plt.clf()
    plt.grid(False)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    if not relative:
        res = ax.imshow(confMat, cmap='coolwarm', interpolation='nearest')
    else:
        res = ax.imshow(confMat, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=100)

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(np.round(confMat[x][y], 1)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    fig.colorbar(res)

    if len(classes) > 0:
        plt.xticks(range(width), classes)
        plt.yticks(range(height), classes)

    rcParams['figure.figsize'] = oldParams

    #return fig

# Plot CV scores of a 2D grid search
def plotGridResults2D(x, y, x_label, y_label, grid_scores):
    """
    # Coarse grid
    C_range = np.r_[np.logspace(-2, 20, 13)]
    gamma_range = np.r_[np.logspace(-9, 5, 15)]
    grid = GridSearchCV(sklearn.svm.SVC(C=1.0, kernel='rbf', class_weight='auto', verbose=False, max_iter=60000),
    {'C' : C_range, 'gamma': gamma_range},
    scoring='roc_auc', cv=10, n_jobs=8)
    grid.fit(X_learn, y_learn)

    plotGridResults2D(C_range, gamma_range, 'C', 'gamma', grid.grid_scores_)
"""

    scores = [s[1] for s in grid_scores]
    scores = np.array(scores).reshape(len(x), len(y))

    plt.figure()
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.RdYlGn)
    plt.xlabel(y_label)
    plt.ylabel(x_label)
    plt.colorbar()
    plt.xticks(np.arange(len(y)), y, rotation=45)
    plt.yticks(np.arange(len(x)), x)
    plt.title('Validation accuracy')

# Plot CV scores of a 1D "grid" search (a very narrow "grid")
def plotGridResults1D(x, x_label, grid_scores):
    """plotGridResults1D(C_range2, 'C', gridFine.grid_scores_)"""

    scores = np.array([s[1] for s in grid_scores])

    plt.figure()
    plt.plot(scores)
    plt.xlabel(x_label)
    plt.ylabel('Score')
    plt.xticks(np.arange(len(x)), x, rotation=45)
    plt.title('Validation accuracy')
