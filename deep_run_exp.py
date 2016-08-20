#!/usr/bin/env python
"""Sample script of recursive neural networks for sentiment analysis.
This is Socher's simple recursive model, not RTNN:
  R. Socher, C. Lin, A. Y. Ng, and C.D. Manning.
  Parsing Natural Scenes and Natural Language with Recursive Neural Networks.
  in ICML2011.
"""

import argparse
import collections
import random
import os
from operator import itemgetter

import numpy as np
from chainer import cuda, Serializer
from chainer import optimizers
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
<<<<<<< HEAD
from sklearn.cross_validation import StratifiedKFold
from ast_tree.ASTVectorizater import TreeFeatures
=======

>>>>>>> e83d97f289c50f7c857ec7a037363b1e0ec47637
from ast_tree.ast_parser import children
# from deep_ast.tree_lstm.treelstm import TreeLSTM
from models import RecursiveLSTMNet
from utils.prog_bar import Progbar
<<<<<<< HEAD
from utils.utils import get_basefolder, generate_trees, parse_src_files


# xp = np  # cuda.cupy  #

# MAX_BRANCH = 4

# class RecursiveNet(chainer.Chain):
#     def __init__(self, n_units, n_label, classes=None):
#         super(RecursiveNet, self).__init__()
#         self.classes_ = classes
#         self.feature_dict = TreeFeatures()
#
#         self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
#         self.add_link("lstm", TreeLSTM(MAX_BRANCH, n_units, n_units))
#         self.add_link("w", L.Linear(n_units, n_label))
#
#     def leaf(self, x, train_mode=False):
#         p = self.embed_vec(x, train_mode)
#         return self.lstm(None, None, p)
#
#     def embed_vec(self, x, train_mode=False):
#         word = self.xp.array([self.feature_dict.astnodes.index(x)], self.xp.int32)
#         w = chainer.Variable(word, volatile=not train_mode)
#         return self.embed(w)
#
#     def merge(self, x, children):
#         c_list, h_list = zip(*children)
#         return self.lstm(c_list, h_list, x)
#
#     def label(self, v):
#         return self.w(v)
#
#     def predict(self, x):
#         t = self.label(x)
#         X_prob = F.softmax(t)
#         indics_ = cuda.to_cpu(X_prob.data.argmax(axis=1))
#         return self.classes_[indics_]
#
#     def loss(self, x, y, train_mode=False):
#         w = self.label(x)
#         label = self.xp.array([y], self.xp.int32)
#         t = chainer.Variable(label, volatile=not train_mode)
#         return F.softmax_cross_entropy(w, t)


class RecursiveLSTMNet(chainer.Chain):
    def __init__(self, n_units, n_label, classes=None):
        super(RecursiveLSTMNet, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        #self.add_link("batch1", L.BatchNormalization(n_units))
        #self.add_link("batch2", L.BatchNormalization(n_units))
        #self.add_link("batch3", L.BatchNormalization(n_units))
        # self.add_link("batch1", L.BatchNormalization(n_units))
        # self.add_link("batch2", L.BatchNormalization(n_units))
        # self.add_link("batch3", L.BatchNormalization(n_units))
        self.add_link("lstm1", L.LSTM(n_units, n_units))
        # self.add_link("lstm2", L.LSTM(n_units, n_units))
        # self.add_link("lstm3", L.LSTM(n_units, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def leaf(self, x, train_mode=False):
        return self.embed_vec(x, train_mode)

    def embed_vec(self, x, train_mode=False):
        word = self.xp.array([self.feature_dict.astnodes.index(x)], self.xp.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        return self.embed(w)

    def merge(self, x, children, train_mode=False):
        # c_list,h_list = zip(*children)
        #h0 = self.lstm1(self.batch1(x))  # self.batch(
        #h1 = self.lstm2(self.batch2(h0))  # self.batch(
        #h2 = F.dropout(self.lstm3(self.batch3(h1)),train=train_mode)  # self.batch(
        #h1 = F.dropout(self.lstm2(self.batch2(h0)),train=train_mode)  # self.batch(
        #h2 = F.dropout(self.lstm3(self.batch3(h1)),train=train_mode)  # self.batch(
        # h0 = self.lstm1(self.batch1(x))  # self.batch(
        # h1 = self.lstm2(self.batch2(h0))  # self.batch(
        # h2 = F.dropout(self.lstm3(self.batch3(h1)),train=train_mode)  # self.batch(
        h0 = self.lstm1(x)  # self.batch(
        for child in children:
            h0 = self.lstm1(child)
        # h1 = F.dropout(self.lstm2(self.batch2(h0)),train=train_mode)  # self.batch(
        # h2 = F.dropout(self.lstm3(self.batch3(h1)),train=train_mode)  # self.batch(
        self.lstm1.reset_state()
        # self.lstm2.reset_state()
        # self.lstm3.reset_state()
        return h0

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
        return cuda.to_cpu(X_prob.data)[0]

    def loss(self, x, y, train_mode=False):
        w = self.label(x)
        label = self.xp.array([y], self.xp.int32)
        t = chainer.Variable(label, volatile=not train_mode)
        return F.softmax_cross_entropy(w, t)
=======
from utils.utils import get_basefolder, parse_src_files, print_model
>>>>>>> e83d97f289c50f7c857ec7a037363b1e0ec47637


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
        x = model.embed_vec(node, train_mode=train_mode)
        return model.merge(x, children_nodes, train_mode=train_mode)


def train(model, train_trees, train_labels, optimizer, batch_size=5, shuffle=True):
    progbar = Progbar(len(train_labels))
    batch_loss = 0
    total_loss = []
    if shuffle:
        indices = np.arange(len(train_labels))
        random.shuffle(indices)
        train_trees = train_trees[indices]
        train_labels = train_labels[indices]
    for idx, tree in enumerate(train_trees):
        root_vec = traverse_tree(model, tree, train_mode=True)
        batch_loss += model.loss(root_vec, train_labels[idx], train_mode=True)
        progbar.update(idx + 1, values=[("training loss", cuda.to_cpu(batch_loss.data))])
        if (idx + 1) % batch_size == 0:
            model.zerograds()
            batch_loss.backward()
            optimizer.update()
            total_loss.append(float(batch_loss.data))
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
        predict.extend(m.predict(root_vec, index=True))
        predict_proba.append(m.predict_proba(root_vec))
        if idx % batch_size == 0:
            total_loss.append(float(batch_loss.data))
            batch_loss = 0
    predict = np.array(predict)
    accuracy = accuracy_score(predict, test_labels)
    mean_loss = np.mean(total_loss)
    print("\tAccuracy: %0.2f " % (accuracy))
    print("\tLoss: %0.2f " % mean_loss)
    # print("\tPrediction Proba : ", collections.Counter([prob[prob.argmax()] for prob in predict_proba]).most_common())
    # print("\tPrediction : ", collections.Counter(predict).most_common())
    return accuracy, mean_loss


def split_trees1(trees, tree_labels, validation=0.1, test=0.1, shuffle=True):
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


def split_trees(trees, tree_labels, n_folds=10, shuffle=True):
    classes_, y = np.unique(tree_labels, return_inverse=True)
    tree_labels = y
    if shuffle:
        indices = np.arange(trees.shape[0])
        random.shuffle(indices)
        trees = trees[indices]
        tree_labels = tree_labels[indices]
<<<<<<< HEAD
    return trees, tree_labels, trees, tree_labels, classes_
    #classes_ = np.arange(len(classes_))
    #cv = StratifiedKFold(tree_labels, n_folds=10, shuffle=shuffle)
    #train_indices, test_indices = next(cv.__iter__())
    #train_trees, train_lables = trees[train_indices], tree_labels[train_indices]
    #test_trees, test_lables = trees[test_indices], tree_labels[test_indices]
    #return train_trees, train_lables, test_trees, test_lables, classes_
    #return train_trees, train_lables, test_trees, test_lables, classes_


def make_backward_graph(basefolder, filename, var):
    import chainer.computational_graph as c
    import os
    g = c.build_computational_graph(var)
    with open(os.path.join(basefolder, filename), '+w') as o:
        o.write(g.dump())
    # dot -Tps filename.dot -o outfile.ps
    from subprocess import call
    call(["dot", "-Tpdf", os.path.join(basefolder, filename), "-o", os.path.join(basefolder, filename + ".pdf")])


def main():
    n_epoch = 500
    n_units = 500
    batch_size = 1
=======
    # classes_ = np.arange(len(classes_))
    cv = StratifiedKFold(tree_labels, n_folds=n_folds, shuffle=shuffle)
    train_indices, test_indices = next(cv.__iter__())
    train_trees, train_lables = trees[train_indices], tree_labels[train_indices]
    test_trees, test_lables = trees[test_indices], tree_labels[test_indices]
    return train_trees, train_lables, test_trees, test_lables, classes_


def pick_subsets(trees, tree_labels, labels=2):
    # pick a small subsets of the classes
    labels_subset = np.arange(len(tree_labels))
    random.shuffle(labels_subset)
    labels_subset = tree_labels[labels_subset][:labels]

    selected_indices = np.where(
        np.in1d(tree_labels, labels_subset))  # (np.where(tree_labels[tree_labels == i]) for i in labels_subset)
    trees = trees[selected_indices]
    tree_labels = tree_labels[selected_indices]

    return trees, tree_labels


def main_experiment():
>>>>>>> e83d97f289c50f7c857ec7a037363b1e0ec47637
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--name', '-n', type=str, default="default_experiment",help='Experiment name')
    parser.add_argument('--folder', '-f', type=str, default="~/projects/stylometory/stylemotery/results",help='Base folder for logs and results')
    args = parser.parse_args()

    output_folder = args.folder  # R"C:\Users\bms\PycharmProjects\stylemotery_code" #
    exper_name = args.name
    output_file = open(os.path.join(output_folder, exper_name+"_results.txt"), mode="+w")
    output_file.write("Testing overfitting the model on all the datasets\n")

    n_epoch = 5
    n_units = 500
    batch_size = 1
    gpu = args.gpu

    base_folder = get_basefolder()
    trees, tree_labels, lable_problems = parse_src_files(base_folder)
    # trees, tree_labels = pick_subsets(trees, tree_labels, labels=2)
    train_trees, train_lables, test_trees, test_lables, classes = split_trees(trees, tree_labels, n_folds=5,
                                                                              shuffle=True)

    output_file.write("Class ratio %s\n" % list(
        sorted([(t, c, c / len(tree_labels)) for t, c in collections.Counter(tree_labels).items()], key=itemgetter(0),
               reverse=False)))
    output_file.write("Train labels :(%s,%s%%)\n" % (len(train_lables), (len(train_lables) / len(tree_labels)) * 100))
    output_file.write("Test  labels :(%s,%s%%)\n" % (len(test_lables), (len(test_lables) / len(tree_labels)) * 100))

    model = RecursiveLSTMNet(n_units, len(classes), classes=classes)
    output_file.write("Model: {0} \n".format(exper_name))
    print_model(model, depth=1, output=output_file)

    if gpu >= 0:
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)  # AdaGrad(lr=0.1) #
    output_file.write("Optimizer: {0} \n".format((type(optimizer).__name__, optimizer.__dict__)))
    optimizer.setup(model)

    output_file.write("Evaluation\n")
    output_file.write("epoch\ttraining loss\ttest loss\ttest accuracy\n")

    output_file.flush()
    for epoch in range(1, n_epoch + 1):
        print('Epoch: {0:d} / {1:d}'.format(epoch, n_epoch))
        print('Train')
        training_loss = train(model, train_trees, train_lables, optimizer, batch_size, shuffle=True)
        print('Test')
<<<<<<< HEAD
        evaluate(model, test_trees, test_lables, batch_size)
        # evaluate(model, test_trees[:10], test_lables[:10], batch_size)
=======
        test_accuracy, test_loss = evaluate(model, test_trees, test_lables, batch_size)
>>>>>>> e83d97f289c50f7c857ec7a037363b1e0ec47637
        print()
        output_file.write("{0}\t{1}\t{2}\t{3}\n".format(epoch, training_loss, test_loss, test_accuracy))
        output_file.flush()
    output_file.close()


if __name__ == "__main__":
    main_experiment()
