import random

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
# from deep_ast.tree_lstm.treelstm import TreeLSTM
from utils.prog_bar import Progbar


def train(model, train_trees, train_labels, optimizer, batch_size=5, shuffle=True):
    progbar = Progbar(len(train_labels))
    batch_loss = 0
    total_loss = []
    predict = []
    if shuffle:
        indices = np.arange(len(train_labels))
        random.shuffle(indices)
        train_trees = train_trees[indices]
        train_labels = train_labels[indices]
    for idx, tree in enumerate(train_trees):
        root_vec = model.traverse(tree, train_mode=True)
        batch_loss += model.loss(root_vec, train_labels[idx], train_mode=True)
        predict.extend(model.predict(root_vec, index=True))
        progbar.update(idx + 1, values=[("training loss", batch_loss.data)])
        if (idx + 1) % batch_size == 0:
            model.zerograds()
            batch_loss.backward()
            optimizer.update()
            total_loss.append(float(batch_loss.data) / batch_size)
            batch_loss = 0
    predict = np.array(predict)
    accuracy = accuracy_score(predict, train_labels)
    print("\tAccuracy: %0.2f " % (accuracy))
    return accuracy, np.mean(total_loss)


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
        # predict_proba.append(m.predict_proba(root_vec))
        if idx % batch_size == 0:
            total_loss.append(float(batch_loss.data) / batch_size)
            batch_loss = 0
    predict = np.array(predict)
    accuracy = accuracy_score(predict, test_labels)
    mean_loss = np.mean(total_loss)
    print("\tAccuracy: %0.2f " % (accuracy))
    # print("\tLoss: %0.2f " % mean_loss)
    return accuracy, mean_loss


def validation_split_trees(trees, tree_labels, validation=0.1, test=0.1, shuffle=True):
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


def split_trees(trees, tree_labels, n_folds=10, shuffle=True,seed=None,iterations=0):
    classes_, y = np.unique(tree_labels, return_inverse=True)
    tree_labels = y
    # if shuffle:
    #     indices = np.arange(trees.shape[0])
    #     random.shuffle(indices)
    #     trees = trees[indices]
    #     tree_labels = tree_labels[indices]
    # classes_ = np.arange(len(classes_))
    # seed = random.randint(0, 4294967295)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=seed)
    for idx,(train_indices, test_indices) in enumerate(cv.split(trees,tree_labels)):
        if idx >= iterations:
            break
    train_trees, train_lables = trees[train_indices], tree_labels[train_indices]
    test_trees, test_lables = trees[test_indices], tree_labels[test_indices]
    return train_trees, train_lables, test_trees, test_lables, classes_, cv


def pick_subsets(trees, tree_labels, labels=2,classes=[],seed=None):
    # pick a small subsets of the classes
    if  classes is not None and len(classes) > 0:
        labels_subset = np.array(classes)
    else:
        labels_subset = np.unique(tree_labels)
        np.random.seed(seed)
        random.shuffle(labels_subset)
        labels_subset = labels_subset[:labels]

    selected_indices = np.where(np.in1d(tree_labels, labels_subset))
    trees = trees[selected_indices]
    tree_labels = tree_labels[selected_indices]

    return trees, tree_labels

def read_config(filename):
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if line.startswith("Seed"):
                args_line = line.replace(":-",":").split(":",1)
                seed = int(args_line[1].strip())
            elif line.startswith("Classes"):
                classes = [v for v in eval(line.split(":")[1])]
        return seed,classes
