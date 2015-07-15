
"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.utils import shuffle


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

if __name__ == '__main__':

# Load the training file
    STOPWORDS = nltk.corpus.stopwords.words('english')
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

# we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

# create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

# do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

# the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=5,  max_features=None,
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = STOPWORDS)

# Fit TFIDF

    tfv.fit(traindata)
    X =  tfv.transform(traindata)
    X_test = tfv.transform(testdata)

# Initialize SVD
    svd = TruncatedSVD(n_components=225)

    # n_components = 225 CV SCORE 0.683
    # n_components = 235 CV SCORE 0.68
    # n_components = 210 CV SCORE 0.681
    # n_components = 179 CV SCORE 0.678
    # n_components = 225 CV SCORE 0.684, C= 7, gamma = 0.0025, coef0, ngram range (1,4)
    # n_components = 225 CV SCORE 0.680, C=8, gamma = 0.0025, coef0=0, min df = 4
    # n_components = 225 CV SCORE 0.685 C=7, gamma = 0.0025, coef0 = 0, min df = 5, ngram range (1,5), tol=0.001

    # amazon features

    # n_components = 225 CV SCORE 0.687, C=6, gamma = 0.0025, coef0=0, kernel = rbf, degree = 3, ngram range (1,5), tol =0

    X =  svd.fit_transform(X)
    X_test = svd.transform(X_test)

    train_ngrams = pd.read_csv("data/ngramMatch_07.csv").fillna("").values
    test_ngrams  = pd.read_csv("data/ngramMatch_test_07.csv").fillna("").values

    train_1234 = pd.read_csv("data/train1234.csv").fillna("").values
    test_1234 = pd.read_csv("data/test1234.csv").fillna("").values

    train_1234_2 = pd.read_csv("data/train1234_2.csv").fillna("").values
    test_1234_2 = pd.read_csv("data/test1234_2.csv").fillna("").values

    train_1234_3 = pd.read_csv("data/train1234_3.csv").fillna("").values
    test_1234_3 = pd.read_csv("data/test1234_3.csv").fillna("").values

    train_alt_query = pd.read_csv('data/alt_query_features_train.csv').fillna("").values
    test_alt_query = pd.read_csv('data/alt_query_features_test.csv').fillna("").values

    train_feature_noun = pd.read_csv('data/product_simscore_train.csv').fillna("").values
    test_feature_noun = pd.read_csv('data/product_simscore_test.csv').fillna("").values

    print('train ngrams shape',train_ngrams.shape)

    X=np.hstack((X,train_ngrams))
    X=np.hstack((X,train_1234))
    X=np.hstack((X,train_1234_2))
    X=np.hstack((X,train_1234_3))
#    X=np.hstack((X,train_alt_query))
#    X=np.hstack((X,train_feature_noun))

    X_test=np.hstack((X_test,test_ngrams))
    X_test=np.hstack((X_test,test_1234))
    X_test=np.hstack((X_test,test_1234_2))
    X_test=np.hstack((X_test,test_1234_3))
#    X_test=np.hstack((X_test,test_alt_query))
#    X_test=np.hstack((X_test,test_feature_noun))

    scl = StandardScaler()
# Initialize the standard scaler

    X = scl.fit_transform(X)
    X_test = scl.fit_transform(X_test)
# We will use SVM here..
    clf =  SVC(C=7, kernel='rbf',degree=3, gamma=0.0025, coef0=0, shrinking=True, probability=False,
               cache_size=225, class_weight=None,tol=0.001)

# Create the pipeline

    # n_components = 225 CV SCORE 0.685 C=7, gamma = 0.0025, coef0 = 0, min df = 5, ngram range (1,5), tol=0.001

# Create a parameter grid to search for best parameters for everything in the pipeline

# Kappa Scorer:


# Initialize Grid Search Model:
    #y = y.reshape(len(y),1)
    #X_y = np.hstack((X,y))

    #X_y = shuffle(X_y)

    #y = X_y[:,X_y.shape[1]-1]

    #X = X_y[:,:X_y.shape[1]-2]

    indices = pd.read_csv('data/cv_5fold_keys.csv')

    kappa_list = []
    for i in range (1,6):
        X_new = X[indices[indices['fold']!=i].values-1,:][:,1]
        y_new = y[indices[indices['fold']!=i].values-1][:,1]
        X_cv = X[indices[indices['fold']==i].values-1,:][:,1]
        y_cv = y[indices[indices['fold']==i].values-1][:,1]
# Fit Grid Search Model
        clf.fit(X_new, y_new)

        y_pred = clf.predict(X_cv)
        print (i, quadratic_weighted_kappa(y_cv,y_pred))
        kappa_list.append(quadratic_weighted_kappa(y_cv,y_pred))

        cv_sub = pd.DataFrame({ "prediction": y_pred})
        cv_sub.to_csv('data/sub_d_cv_'+str(i)+'.csv', index=False)

    print('kappa mean', np.array(kappa_list).mean())




# Fit model with best parameters optimized for quadratic_weighted_kappa
    clf.fit(X,y)
    preds = clf.predict(X_test)

# Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("data/submission_e.csv", index=False)
