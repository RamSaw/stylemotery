import numpy as np


def softmax(X):
    X = X.copy()
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

from sklearn.linear_model import LogisticRegression

def softmax_cross_entropy2(t, y, classes):
    X_prob = softmax(t)
    predict_prob_indics = X_prob.argmax(axis=1)
    match = classes[predict_prob_indics] == y
    right = np.where(match == True)[0]
    wrong = np.where(match == False)[0]
    return right, wrong


X = np.random.uniform(0, 1, (10, 5))
y = np.random.randint(0, 5, 10)

print(X)
print(y)
classes = np.array([0, 1, 2, 3, 4])
print(softmax_cross_entropy2(X, y, classes))
