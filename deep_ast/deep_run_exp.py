#!/usr/bin/env python
"""Sample script of recursive neural networks for sentiment analysis.
This is Socher's simple recursive model, not RTNN:
  R. Socher, C. Lin, A. Y. Ng, and C.D. Manning.
  Parsing Natural Scenes and Natural Language with Recursive Neural Networks.
  in ICML2011.
"""

import argparse
import codecs
import collections
import random
import re
import time
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from chainer import variable
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from sklearn.metrics import accuracy_score

from ast_example.ASTVectorizater import TreeFeatures
from ast_parser import children
from deep_ast.tree_lstm.treelstm import TreeLSTM
from prog_bar import Progbar
from utils import get_basefolder, parse_src_files

xp = np  # cuda.cupy  #

MAX_BRANCH = 4


def convert_tree(vocab, exp):
    assert isinstance(exp, list) and (len(exp) == 2 or len(exp) == 3)

    if len(exp) == 2:
        label, leaf = exp
        if leaf not in vocab:
            vocab[leaf] = len(vocab)
        return {'label': int(label), 'node': vocab[leaf]}
    elif len(exp) == 3:
        label, left, right = exp
        node = (convert_tree(vocab, left), convert_tree(vocab, right))
        return {'label': int(label), 'node': node}


class RecursiveNet(chainer.Chain):
    def __init__(self, n_units, n_label, classes=None):
        super(RecursiveNet, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        self.add_link("lstm", TreeLSTM(MAX_BRANCH, n_units, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def leaf(self, x, train_mode=False):
        p = self.embed_vec(x, train_mode)
        return self.lstm(None, None, p)

    def embed_vec(self, x, train_mode=False):
        word = xp.array([self.feature_dict.astnodes.index(x)], np.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        return self.embed(w)

    def merge(self, x, children):
        c_list, h_list = zip(*children)
        return self.lstm(c_list, h_list, x)

    def label(self, v):
        return self.w(v)

    def predict(self, x):
        t = self.label(x)
        X_prob = F.softmax(t)
        indics_ = cuda.to_cpu(X_prob.data.argmax(axis=1))
        return self.classes_[indics_]

    def loss(self, x, y, train_mode=False):
        w = self.label(x)
        label = xp.array([y], np.int32)
        t = chainer.Variable(label, volatile=not train_mode)
        return F.softmax_cross_entropy(w, t)


class RecursiveLSTMNet(chainer.Chain):
    def __init__(self, n_units, n_label, classes=None):
        super(RecursiveLSTMNet, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        self.add_link("batch", L.BatchNormalization(n_units))
        self.add_link("lstm", L.LSTM(n_units, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def leaf(self, x, train_mode=False):
        return self.embed_vec(x, train_mode)

    def embed_vec(self, x, train_mode=False):
        word = xp.array([self.feature_dict.astnodes.index(x)], np.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        return self.embed(w)

    def merge(self, x, children):
        # c_list,h_list = zip(*children)
        h = self.lstm(x)  # self.batch(
        self.lstm.reset_state()
        return h

    def label(self, v):
        return self.w(v)

    def predict(self, x):
        t = self.label(x)
        X_prob = F.softmax(t)
        indics_ = cuda.to_cpu(X_prob.data.argmax(axis=1))
        return self.classes_[indics_]

    def predict_proba(self, x):
        t = self.label(x)
        X_prob = F.softmax(t)
        return cuda.to_cpu(X_prob.data)

    def loss(self, x, y, train_mode=False):
        w = self.label(x)
        label = xp.array(np.where(self.classes_ ==y)[0], np.int32)
        t = chainer.Variable(label, volatile=not train_mode)
        return F.softmax_cross_entropy(w, t)


def traverse_tree(model, node, train_mode=True):
    children_ast = list(children(node))
    if len(children_ast) == 0:
        # leaf node
        return model.leaf(node, train_mode=train_mode)
    else:
        # internal node
        children_nodes = []
        for child in children_ast:
            child_node = traverse_tree(model, child, train_mode=train_mode)
            children_nodes.append(child_node)
        if len(children_nodes) < MAX_BRANCH:
            children_nodes.extend(
                [model.leaf(None, train_mode=train_mode) for i in range(MAX_BRANCH - len(children_nodes))])
        children_nodes = children_nodes[:MAX_BRANCH]
        x = model.embed_vec(node, train_mode=train_mode)
        return model.merge(x, children_nodes)


def train(model, train_trees, train_labels, optimizer, batch_size=5, shuffle=True):
    progbar = Progbar(len(train_labels))
    batch_loss = 0
    total_loss = []
    if shuffle:
        random.shuffle(train_trees)
    for idx, tree in enumerate(train_trees):
        # tree.body = tree.body[0]
        root_vec = traverse_tree(model, tree, train_mode=True)
        batch_loss += model.loss(root_vec, train_labels[idx], train_mode=True)
        progbar.update(idx+1,values=[("training loss",batch_loss.data)])
        if idx % batch_size == 0:
            model.zerograds()
            batch_loss.backward()
            optimizer.update()
            total_loss.append(float(batch_loss.data) / batch_size)
            batch_loss = 0
    return np.mean(total_loss)


def evaluate(model, test_trees, test_labels, batch_size=1):
    m = model.copy()
    m.volatile = True
    progbar = Progbar(len(test_labels))
    batch_loss = 0
    total_loss = []
    predict_proba = []
    predict = []
    for idx, tree in enumerate(test_trees):
        root_vec = traverse_tree(m, tree, train_mode=False)
        batch_loss += m.loss(root_vec, test_labels[idx], train_mode=False)
        progbar.update(idx + 1, values=[("test loss", batch_loss.data)])
        predict.extend(m.predict(root_vec))
        predict_proba.append(m.predict_proba(root_vec))
        if idx % batch_size == 0:
            total_loss.append(float(batch_loss.data) / batch_size)
            batch_loss = 0
    predict = np.array(predict)
    accuracy = accuracy_score(predict, test_labels)
    mean_loss = np.mean(total_loss)
    print("\tAccuracy: %0.2f " % (accuracy))
    print("\tLoss: %0.2f " % mean_loss)
    return mean_loss


def split_trees(trees, tree_labels, validation=0.1, test=0.1, shuffle=True):
    classes_, y = np.unique(tree_labels, return_inverse=False)
    tree_labels = y
    indices = np.arange(trees.shape[0])
    if shuffle:
        random.shuffle(indices)
    train_samples = int((1 - validation - test) * indices.shape[0])
    valid_samples = int(validation * indices.shape[0])
    test_samples = int(test * indices.shape[0])

    train_indices = indices[:train_samples]
    train_trees, train_lables = trees[train_indices], tree_labels[train_indices]

    if validation > 0:
        validate_indices = indices[train_samples:train_samples + valid_samples]
        validate_trees, validate_lables = trees[validate_indices], tree_labels[validate_indices]

    test_indices = indices[:-test_samples]
    test_trees, test_lables = trees[test_indices], tree_labels[test_indices]

    if validation > 0:
        return train_trees, train_lables, validate_trees, validate_lables, test_trees, test_lables, classes_
    else:
        return train_trees, train_lables, test_trees, test_lables, classes_


def split_trees2(trees, tree_labels, shuffle=True):
    classes_ = np.unique(tree_labels, return_inverse=False)
    # tree_labels = y
    cv = StratifiedKFold(tree_labels, n_folds=5, shuffle=shuffle)
    train_indices, test_indices = next(cv.__iter__())
    train_trees, train_lables = trees[train_indices], tree_labels[train_indices]
    test_trees, test_lables = trees[test_indices], tree_labels[test_indices]
    return train_trees, train_lables, test_trees, test_lables, classes_


def main():
    USE_GPU = -1

    basefolder = get_basefolder()
    trees, tree_labels, lable_problems = parse_src_files(basefolder)

    # train_trees, train_lables, validate_trees, validate_lables, test_trees, test_lables, classes = split_trees2(trees,
    #                                                                                                            tree_labels,
    #                                                                                                            validation=0.1,
    #                                                                                                            test=0.1,
    #                                                                                                            shuffle=True)

    train_trees, train_lables, test_trees, test_lables, classes = split_trees2(trees, tree_labels, shuffle=True)

    n_epoch = 100
    n_units = 50
    batch_size = 10

    model = RecursiveLSTMNet(n_units, len(classes), classes=classes)

    if USE_GPU >= 0:
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.AdaGrad(lr=0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    for epoch in range(1,n_epoch+1):
        print('Epoch: {0:d} / {0:d}'.format(epoch,n_epoch))

        cur_at = time.time()
        total_loss = train(model, train_trees, train_lables, optimizer, batch_size, shuffle=True)
        print('loss: {:.2f}'.format(total_loss))
        now = time.time()
        throughput = float(len(train_trees)) / (now - cur_at)
        print('{:.2f} iters/sec, {:.2f} sec'.format(throughput, now - cur_at))

        # if (epoch + 1) % epoch_per_eval == 0:
        #     print('Train data evaluation:')
        #     evaluate(model, train_trees)
        #     print('')

        print('Test evaluateion')
        evaluate(model, test_trees, test_lables, batch_size)
        print()


if __name__ == "__main__":
    main()
