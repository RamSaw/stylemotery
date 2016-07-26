# import numpy as np
#
# np.random.seed(10)
#
# import matplotlib.pyplot as plt
# from sklearn.decomposition import TruncatedSVD
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
#                               GradientBoostingClassifier)
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import roc_curve
# from sklearn.pipeline import make_pipeline
#
# n_estimator = 10
# X, y = make_classification(n_samples=80000)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# # It is important to train the ensemble of trees on a different subset
# # of the training data than the linear regression model to avoid
# # overfitting, in particular if the total number of leaves is
# # similar to the number of training samples
# X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
#                                                             y_train,
#                                                             test_size=0.5)
#
# svd = TruncatedSVD(2)
# # Supervised transformation based on random forests
# rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
# rf_enc = OneHotEncoder()
# rf_lm = LogisticRegression()
# rf.fit(X_train, y_train)
# rf_enc.fit(rf.apply(X_train))
# rf_lm.fit(svd.fit_transform(rf_enc.transform(rf.apply(X_train_lr))), y_train_lr)
#
# y_pred_rf_lm = rf_lm.predict_proba(svd.transform(rf_enc.transform(rf.apply(X_test))))[:, 1]
# fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
#
#
# from sklearn.metrics import auc
# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rf_lm, tpr_rf_lm, label='GBT + LR %f' % auc(fpr_rf_lm, tpr_rf_lm,reorder=True))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.show()
#

class ngramiterator(object):
    def __init__(self, s, n=1):
        self.n = n
        self.s = s
        self.pos = 0

    def __iter__(self):
        return self

    def next(self):
        if self.pos + self.n > len(self.s):
            raise StopIteration
        else:
            self.pos += 1
            return self.s[self.pos - 1:self.pos + self.n - 1]

    def __next__(self):
        return self.next()


def ngrams(string, n):
    return [x for x in ngramiterator(string, n)]


print(ngrams("text",2))