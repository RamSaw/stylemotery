from operator import itemgetter
from pprint import pprint
from time import time
import logging

import itertools
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


###############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
]
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

data = fetch_20newsgroups(subset='train', categories=categories)
print("%d documents" % len(data.filenames))
print("%d categories" % len(data.target_names))
print()

def fit_pair(X):
    cx = X.tocoo()
    X_pair = [[] for i in range(X.shape[0])]
    for i,j,v in zip(cx.row, cx.col, cx.data):
        X_pair[i].append((str(j),v))
    return X_pair
###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.75,ngram_range=(1,2))),
    ('tfidf', TfidfTransformer(use_idf=True,norm="l2")),
    ('clf', SGDClassifier(alpha=0.00001,penalty="l2")),
])

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier

    print("pipeline:", pipeline)
    t0 = time()
    vect = CountVectorizer(max_df=0.75,ngram_range=(1,2))
    tfidf = TfidfTransformer(use_idf=True,norm="l2")
    hash = FeatureHasher(n_features=5000, input_type='pair')
    clf = SGDClassifier(alpha=0.00001,penalty="l2")

    X_data = vect.fit_transform(data.data,data.target)
    X_data = tfidf.fit_transform(X_data,data.target)

    X_data_pair = fit_pair(X_data)
    X_data = hash.fit_transform(X_data_pair,data.target)
    clf.fit(X_data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    X_data = vect.transform(data.data)
    X_data = tfidf.transform(X_data)
    X_data_pair = fit_pair(X_data)
    X_data = hash.transform(X_data_pair,data.target).toarray()
    print(accuracy_score(clf.predict(X_data),data.target))