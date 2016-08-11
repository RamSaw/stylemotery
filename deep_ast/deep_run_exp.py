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

import numpy as np
from chainer import variable
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers

from ast_example.ASTVectorizater import TreeFeatures
from ast_parser import children
from utils import get_basefolder, parse_src_files

xp = np #cuda.cupy  #


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
    def __init__(self, n_vocab, n_units, n_label, classes=None):
        super(RecursiveNet, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l=L.StatelessLSTM(n_units, n_units),
            w=L.Linear(n_units, n_label))
        self.classes_ = classes
        self.feature_dict = TreeFeatures()

    def leaf(self, x,train_mode=False):
        p = self.embed_vec(x,train_mode)
        return self.l(None,None,p)

    def embed_vec(self, x,train_mode=False):
        word = xp.array([self.feature_dict.astnodes.index(x)], np.int32)
        w = chainer.Variable(word,volatile=not train_mode)
        return self.embed(w)

    def merge(self, x):
        c_list,h_list = zip(*x)
        return self.l(c_list,h_list,x)

    def label(self, v):
        return self.w(v)

    def predict(self, x):
        t = self.label(x)
        X_prob = F.softmax(t)
        indics_ = cuda.to_cpu(X_prob.data.argmax(axis=1))
        return self.classes_[indics_]

    def loss(self, x, y):
        t = self.label(x)
        label = xp.array([y], np.int32)
        t = chainer.Variable(label, volatile=not train)
        return F.softmax_cross_entropy(y, t)


def traverse(model, node, train=True, evaluate=None):
    children_ast = list(children(node))
    if len(children_ast) == 0:
        # leaf node
        return model.leaf(node)
    else:
        # internal node
        children_nodes = []
        for child in children_ast:
            child_node = traverse(model, child, train=train, evaluate=evaluate)
            children_nodes.append(child_node)
        x = model.embed_vec(node)
        return model.merge(x,children_nodes)


def train(model, train_trees, optimizer, batch_size=1, shuffle=True):
    batch_loss = 0
    total_loss = 0
    if shuffle:
        random.shuffle(train_trees)
    for idx, tree in enumerate(train_trees):
        loss, v = traverse(model, tree, train=True)
        batch_loss += loss

        if idx % batch_size == 0:
            model.zerograds()
            batch_loss.backward()
            optimizer.update()
            total_loss += float(batch_loss.data)
            batch_loss = 0
    return total_loss


def evaluate(model, test_trees, batch_size=1):
    m = model.copy()
    m.volatile = True
    result = collections.defaultdict(lambda: 0)
    for tree in test_trees:
        traverse(m, tree, train=False, evaluate=result)

    acc_node = 100.0 * result['correct_node'] / result['total_node']
    acc_root = 100.0 * result['correct_root'] / result['total_root']
    print(' Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_node, result['correct_node'], result['total_node']))
    print(' Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_root, result['correct_root'], result['total_root']))


def split_trees(trees, tree_labels, validation=0.1, test=0.1, shuffle=True):
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
        return train_trees, train_lables, validate_trees, validate_lables, test_trees, test_lables
    else:
        return train_trees, train_lables, test_trees, test_lables

def main():
    USE_GPU = -1

    basefolder = get_basefolder()
    trees, tree_labels, lable_problems = parse_src_files(basefolder)

    train_trees, train_lables, validate_trees, validate_lables, test_trees, test_lables = split_trees(trees,
                                                                                                      tree_labels,
                                                                                                      validation=0.1,
                                                                                                      test=0.1,
                                                                                                      shuffle=True)

    n_epoch = 100
    n_units = 50
    batch_size = 1
    classes = np.unique(tree_labels)

    model = RecursiveNet(100, n_units,len(classes),classes=classes)

    if USE_GPU >= 0:
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.AdaGrad(lr=0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    for epoch in range(n_epoch):
        print('Epoch: {0:d}'.format(epoch))

        cur_at = time.time()
        total_loss = train(model, train_trees, optimizer, batch_size, shuffle=True)
        print('loss: {:.2f}'.format(total_loss))
        now = time.time()
        throughput = float(len(train_trees)) / (now - cur_at)
        print('{:.2f} iters/sec, {:.2f} sec'.format(throughput, now - cur_at))
        print()

        # if (epoch + 1) % epoch_per_eval == 0:
        #     print('Train data evaluation:')
        #     evaluate(model, train_trees)
        #     print('')

        print('Test evaluateion')
        evaluate(model, test_trees, batch_size)


if __name__ == "__main__":
    main()
