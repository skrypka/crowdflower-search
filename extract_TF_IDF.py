##Scores 0.61817
##With cv=0.630

##TODO: custom stopwords

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
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier

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

#min_df_list=[3,5,7]
        ngrams=[(1,4)]

##best performance so far with ngram 1,4

        for ngram_val in ngrams:
                print ('Training for ngram= ' + str(ngram_val))
                tfv = TfidfVectorizer(min_df=5,  max_features=None,
                                                  strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                                                  ngram_range=ngram_val, use_idf=1,smooth_idf=1,sublinear_tf=1,
                                                  stop_words = 'english')
                tfv.fit(traindata)
                X =  tfv.transform(traindata)
                X_test = tfv.transform(testdata)
                svd = TruncatedSVD(n_components=245)
                X=svd.fit_transform(X)
                X_test = svd.transform(X_test)

                np.savetxt("data/Tf_idf_train.csv", X, delimiter=",")
                np.savetxt("data/Tf_idf_test.csv", X_test, delimiter=",")
