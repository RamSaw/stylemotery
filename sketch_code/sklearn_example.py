import time
from operator import itemgetter
from keras.layers.core import *
from keras.models import Sequential
import sklearn
import sklearn.ensemble
import sklearn.metrics
from keras.layers import LSTM, TimeDistributed
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer



def interpret_data(X, y, func, class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    times, scores = [], []
    for r_idx in range(10):
        start_time = time.time()
        exp = explainer.explain_instance(newsgroups_test.data[r_idx], func, num_features=6)
        times.append(time.time() - start_time)
        scores.append(exp.score)
        print('Document id: %d' % r_idx)
        print('Probability(christian) =', c.predict_proba([newsgroups_test.data[r_idx]])[0, 1])
        print('True class: %s' % class_names[newsgroups_test.target[r_idx]])
        print("Features: ",list(sorted(exp.as_list(),key=itemgetter(1),reverse=True))[:5])
        print('...')

    return times, scores

    return  model
if __name__ == '__main__':
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    class_names = ['atheism', 'christian']

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    test_vectors = vectorizer.transform(newsgroups_test.data)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train_vectors, newsgroups_train.target)
    pred = rf.predict(test_vectors)
    sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')
    c = make_pipeline(vectorizer, rf)

    times, scores = interpret_data(train_vectors, newsgroups_train.target, c.predict_proba, class_names)