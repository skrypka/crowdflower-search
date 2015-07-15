import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
from sklearn.preprocessing import Normalizer
from nltk.stem.porter import *
from utils import round_with_x, make_submission, cv_generate
from sklearn.ensemble import BaggingRegressor
from sklearn import cross_validation





def logscaler(input):
        return np.log(input)

logscaler.transform=logscaler.__call__
logscaler.fit= lambda x: logscaler

def extract_features(data): #the features are the number of times that query tokens appear in the title or description, which is cool
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        data["query_tokens_in_title"] = 0.0
        data["query_tokens_in_description"] = 0.0
        data["title_length"] = 0.0
        data["description_length"] = 0.0
        for i, row in data.iterrows():
                query = set(x.lower() for x in token_pattern.findall(row["query"]))
                title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
                description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
                data.set_value(i, "title_length", len(title))
                data.set_value(i, "description_length", len(description))
                data.set_value(i, "query_length", float(len(query)))
                data.set_value(i, "title_length", float(len(title)))
                data.set_value(i, "description_length", float(len(description)))
                if len(title) > 0:
                        data.set_value(i, "query_tokens_in_title", float(len(query.intersection(title)))/float(len(title)))
                else:
                        data.set_value(i, "query_tokens_in_title", 0)
                if len(description) > 0:
                        data.set_value(i, "query_tokens_in_description", len(query.intersection(description))/len(description))
                else:
                        data.set_value(i, "query_tokens_in_description",0)


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
        Calculates the quadratic weighted kappa
        axquadratic_weighted_kappa calculates the quadratic weighted kappa
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



# Load the training file
train = pd.read_csv('input/train.csv').fillna("")
test = pd.read_csv('input/test.csv').fillna("")

print 'there are '+str(train["query"].unique().shape)+' unique queries'

alt_query=pd.read_csv("data/alt_query_features_train_and_test_v01.csv",header=None,sep=';').fillna("").values
alt_query_train=alt_query[:10158]
alt_query_test=alt_query[10158:]
print alt_query_train.shape
#print(alt_query.head())


#print test.describe()





# we dont need ID columns
idx = test.id.values.astype(int)


# create labels. drop useless columns
y = train.median_relevance.values

#remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
stemmer = PorterStemmer()
## Stemming functionality

class stemmerUtility(object):
        """Stemming functionality"""
        @staticmethod
        def stemPorter(review_text):
                porter = PorterStemmer()
                preprocessed_docs = []
                for doc in review_text:
                        final_doc = []
                        for word in doc:
                                final_doc.append(porter.stem(word))
                                #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
                        preprocessed_docs.append(final_doc)
                return preprocessed_docs

if __name__ == '__main__':


        # array declarations
        sw=[]
        s_data = []
        s_labels = []
        t_data = []
        t_labels = []


        ##Working only with query and title
        for i in range(len(train.id)):
                #s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
                s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")])
                s=re.sub("[^a-zA-Z0-9]"," ", s)
                s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
                s_data.append(s)
                s_labels.append(str(train["median_relevance"][i]))

        for i in range(len(test.id)):
                #s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
                s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")])
                s=re.sub("[^a-zA-Z0-9]"," ", s)
                s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
                t_data.append(s)

        train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
        train = train.drop('id', axis=1)
        test = test.drop('id', axis=1)

        extract_features(train)
        extract_features(test)

        traindata=s_data
        testdata=t_data

        #print s_data.shape

#min_df_list=[3,5,7]
        #ngrams=[(1,4)]
        n_components_list=[250]

##best performance so far with ngram 1,4

        for n_components in n_components_list:
                print ('Training for components= ' + str(n_components))
                tfv = TfidfVectorizer(min_df=5,  max_features=None,
                                                  strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                                                  ngram_range=(1,4), use_idf=1,smooth_idf=1,sublinear_tf=1,
                                                  stop_words = 'english')
                tfv.fit(traindata)
                X =  tfv.transform(traindata)
                X_test = tfv.transform(testdata)
                svd = TruncatedSVD(n_components)
                vec_text=svd.fit_transform(X)
                vec_text_test = svd.transform(X_test)

                ##Add some magical features##
                #query_tokens_in_title=train['query_tokens_in_title'].values
                #query_tokens_in_title=np.reshape(query_tokens_in_title,(query_tokens_in_title.shape[0],1))
                #X=np.hstack((X,query_tokens_in_title))
                #X=query_tokens_in_title
                #query_tokens_in_title=test['query_tokens_in_title'].values
                #query_tokens_in_title=np.reshape(query_tokens_in_title,(query_tokens_in_title.shape[0],1))
                #print X_test.shape
                #print query_tokens_in_title.shape
                #X_test=np.hstack((X_test,query_tokens_in_title))
                #X_test=query_tokens_in_title
                ##Add some even more magical features##
                train_ngrams = pd.read_csv("data/ngramMatch_07.csv").fillna("").values
                test_ngrams  = pd.read_csv("data/ngramMatch_test_07.csv").fillna("").values
                train_1234 = pd.read_csv("data/train1234.csv",header=None).fillna("").values #header=None! important! or it crashes like a b-tard!
                test_1234 = pd.read_csv("data/test1234.csv",header=None).fillna("").values
                train_1234_2 = pd.read_csv("data/train1234_2.csv",header=None).fillna("").values
                test_1234_2 = pd.read_csv("data/test1234_2.csv",header=None).fillna("").values
                train_1234_3 = pd.read_csv("data/train1234_3.csv",header=None).fillna("").values
                test_1234_3 = pd.read_csv("data/test1234_3.csv",header=None).fillna("").values
                train_anti_1234=pd.read_csv("data/train_anti_1234.csv",header=None).fillna("").values
                test_anti_1234=pd.read_csv("data/test_anti_1234.csv",header=None).fillna("").values
                train_okapi=pd.read_csv("data/Okapi_train.csv",header=None,sep=';').fillna("").values
                test_okapi=pd.read_csv("data/Okapi_test.csv",header=None,sep=';').fillna("").values
                kaggle_dist_train=pd.read_csv("data/kaggle_dist_train.csv",header=None,sep=';').fillna("").values
                kaggle_dist_test=pd.read_csv("data/kaggle_dist_test.csv",header=None,sep=';').fillna("").values
                train_w2vec=pd.read_csv("data/w2vec_train.csv",header=None).fillna("").values
                test_w2vec=pd.read_csv("data/w2vec_test.csv",header=None).fillna("").values
                train_wm=pd.read_csv("data/train_wm.csv",header=None).fillna("").values
                test_wm=pd.read_csv("data/test_wm.csv",header=None).fillna("").values
                #Okapi_test


                #train_anti_1234
                #alt_query_train
                X=np.hstack((vec_text,train_ngrams,train_1234,train_1234_2,train_1234_3,train_anti_1234))
                #alt_query_test
                #test_anti_1234
                X_test=np.hstack((vec_text_test,test_ngrams,test_1234,test_1234_2,test_1234_3,test_anti_1234))


                #X=np.log(1+X)
                #X_test=np.log(1+X_test)

                scl = StandardScaler()

                # We will use SVM here..
                learner = SVC()

                # Create the pipeline
                clf = pipeline.Pipeline([('scl', scl),('learner', learner)])

                # Create a parameter grid to search for best parameters for everything in the pipeline
                param_grid = {#'svd__n_components' : [200,250,300], #240 was the best for SVC
                'learner__C': [9],'learner__gamma':[0.002],'learner__class_weight':[None,'auto']} #For SVC
                #'learner__C': [15,10],'learner__gamma':[0,0.001]}
                #C=6.0 and ngram=1,4 gives the best performance for the data we have right now



                kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)


                model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                                                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)
                model.fit(X, y)


                X_train, X_test_, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)


                svd_=clf.steps[1][1]
                #print svd_.dual_coef_

                print("Best score: %0.3f" % model.best_score_)
                print("Best parameters set:")
                best_parameters = model.best_estimator_.get_params()
                for param_name in sorted(param_grid.keys()):
                        print("\t%s: %r" % (param_name, best_parameters[param_name]))

                best_model = model.best_estimator_
                #best_model.fit(X,y)

                b = BaggingRegressor(best_model,5, bootstrap=False, verbose=10)
                cv_generate(b, "SVM5b_final", X, y, X_test)
