import argparse
import collections
import random
import os
from operator import itemgetter

import numpy as np
import chainer
import sys
from chainer import cuda, Serializer
from chainer import optimizers
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from ast_tree.ast_parser import children
# from deep_ast.tree_lstm.treelstm import TreeLSTM
from models import RecursiveLSTM, RecursiveTreeLSTM
from utils.prog_bar import Progbar
from utils.fun_utils import get_basefolder, parse_src_files, print_model, generate_trees, make_backward_graph


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
        root_vec = model.traverse(tree, train_mode=True)
        batch_loss += model.loss(root_vec, train_labels[idx], train_mode=True)
        progbar.update(idx + 1, values=[("training loss", batch_loss.data)])
        if (idx + 1) % batch_size == 0:
            model.zerograds()
            batch_loss.backward()

            make_backward_graph(R"C:\Users\bms\PycharmProjects\stylemotery_code","treelstm",[batch_loss])
            exit()
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
        root_vec = m.traverse(tree, train_mode=False)
        batch_loss += m.loss(root_vec, test_labels[idx], train_mode=False)
        progbar.update(idx + 1, values=[("test loss", batch_loss.data)])
        predict.extend(m.predict(root_vec, index=True))
        predict_proba.append(m.predict_proba(root_vec))
        if idx % batch_size == 0:
            total_loss.append(float(batch_loss.data) / batch_size)
            batch_loss = 0
    predict = np.array(predict)
    accuracy = accuracy_score(predict, test_labels)
    mean_loss = np.mean(total_loss)
    print("\tAccuracy: %0.2f " % (accuracy))
    print("\tLoss: %0.2f " % mean_loss)
    return accuracy, mean_loss


def split_trees(trees, tree_labels, n_folds=10, shuffle=True):
    classes_, y = np.unique(tree_labels, return_inverse=True)
    tree_labels = y
    if shuffle:
        indices = np.arange(trees.shape[0])
        random.shuffle(indices)
        trees = trees[indices]
        tree_labels = tree_labels[indices]
    # classes_ = np.arange(len(classes_))
    cv = StratifiedKFold(tree_labels, n_folds=n_folds, shuffle=shuffle)
    train_indices, test_indices = next(cv.__iter__())
    train_trees, train_lables = trees[train_indices], tree_labels[train_indices]
    test_trees, test_lables = trees[test_indices], tree_labels[test_indices]
    return train_trees, train_lables, test_trees, test_lables, classes_


def main_experiment():
    output_file = sys.stdout
    output_file.write("Testing overfitting the model on all the datasets\n")

    n_epoch = 10
    n_units = 500
    batch_size = 1

    base_folder = get_basefolder()
    trees, tree_labels, lable_problems = generate_trees(base_folder,labels=2,children=4,examples_per_label=10)
    train_trees, train_lables, test_trees, test_lables, classes = split_trees(trees, tree_labels, n_folds=5,
                                                                              shuffle=False)

    output_file.write("Class ratio %s\n" % list(sorted([(t, c, c / len(tree_labels)) for t, c in collections.Counter(tree_labels).items()], key=itemgetter(0),reverse=False)))
    output_file.write("Train labels :(%s,%s%%)\n" % (len(train_lables), (len(train_lables) / len(tree_labels)) * 100))
    output_file.write("Test  labels :(%s,%s%%)\n" % (len(test_lables), (len(test_lables) / len(tree_labels)) * 100))

    model = RecursiveTreeLSTM(4,n_units, len(classes), classes=classes)
    output_file.write("Model: {0} \n".format(type(model).__name__))
    print_model(model, depth=1, output=output_file)

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
        test_accuracy, test_loss = evaluate(model, test_trees, test_lables, batch_size)
        print()
        output_file.write("{0}\t{1}\t{2}\t{3}\n".format(epoch, training_loss, test_loss, test_accuracy))
        output_file.flush()

        if test_loss < 0.0001:
            output_file.write("Early Stopping\n")
            print("Early Stopping")
            break

    output_file.close()


if __name__ == "__main__":
    main_experiment()
