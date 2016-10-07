




import os
import random

import sys
from operator import itemgetter

from chainer import serializers
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from ast_tree.ast_parser import AstNodes
from deep_run_exp import split_trees, pick_subsets
from models.lstm_models import RecursiveLSTM
from utils.extract_loss_progress import parse_result_file
from utils.fun_utils import print_model, parse_src_files
from utils.graph import plot_all, plot_each
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA,KernelPCA
from utils.prog_bar import Progbar


def cluster_plot(estimator,X,y,true_labels):
    reducer = KernelPCA(kernel="rbf",n_components=2)
    # X_r = X
    X_r = reducer.fit_transform(X)
    estimator.fit(X)
    core_samples_mask = np.ones_like(estimator.labels_, dtype=bool)
    # core_samples_mask[estimator.core_sample_indices_] = True
    true_labels = np.array(true_labels)
    labels = estimator.labels_
    # Percentage of variance explained for each components
    unique_labels = set(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X_r[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=10)
        for y,x in zip(true_labels[class_member_mask],xy):
            plt.annotate(y, (x[0], x[1]))

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def neighbors_table(estimator,X,y,labels):
    # estimator = PCA(n_components=2)
    # X_r = X
    estimator.fit(X)
    # core_samples_mask[estimator.core_sample_indices_] = True
    # Percentage of variance explained for each components
    near_points = {}
    for idx,x in enumerate(X):
        distances, indics = estimator.kneighbors([x], 7, return_distance=True)
        indics = indics.astype(np.int32)
        close_neighbors = list(sorted((labels[p] for p,d in zip(indics[0],distances[0])),key=itemgetter(1)))
        near_points.update({labels[idx]: close_neighbors})

    for k,v in near_points.items():
        print(k, " => ",v)

def cluster_table(estimator,X,y,true_labels):
    # estimator = PCA(n_components=2)
    # X_r = X
    estimator.fit(X)
    # core_samples_mask[estimator.core_sample_indices_] = True
    labels = estimator.labels_
    # Percentage of variance explained for each components
    unique_labels = set(labels)
    for k in unique_labels:
        class_member_mask = (labels == k)
        # xy = X[class_member_mask & core_samples_mask]
        xy = true_labels[class_member_mask]
        print("Cluster {0}: ".format(k),xy)

def data_plot(X,y,labels):
    estimator = PCA(n_components=2)
    X_r = estimator.fit_transform(X)
    # print('explained variance ratio (first two components): %s'% str(estimator))

    if estimator.n_components == 2:
        plt.figure()
        for idx,x in enumerate(X_r):
            plt.scatter(x[0], x[1])
            plt.annotate(labels[idx], (x[0], x[1]))
        plt.legend()
    elif estimator.n_components == 3:
        fig = plt.figure(0, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        plt.cla()

        for idx,x in enumerate(X_r):
            plt.scatter(x[0], x[1],int(x[2]))
            # plt.annotate(labels[idx], (x[0], x[1]))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
    plt.show()

def evaluate(model, test_trees, test_labels):
    m = model.copy()
    m.volatile = True
    predict = []
    progbar = Progbar(len(test_labels))
    for idx, tree in enumerate(test_trees):
        root_vec = m.traverse(tree, train_mode=False)
        progbar.update(idx + 1)
        predict.extend(m.label(root_vec).data)
    predict = np.array(predict)
    return predict, test_labels

def show_embeding(model):
    # Word Embedding Analysis
    X = scale(model.embed.W.data)
    y = np.arange(X.shape[0])
    ast_nodes = AstNodes()
    true_labels = np.array(ast_nodes.nodetypes + [ast_nodes.NONE])

    # estimator  =  KernelPCA(n_components=2,kernel="rbf")#PCA(n_components=2)#PCA(n_components=2) #TSNE(n_components=2, random_state=None)#
    # PCA_plot(estimator,X,y,true_labels)

    # estimator = DBSCAN(eps=0.3, min_samples=10)
    estimator = KMeans(n_clusters=10,init='k-means++')
    cluster_plot(estimator, X, y, true_labels)
    print("*"*10," Cluster AST Node types:","*"*10)
    cluster_table(estimator, X, y, true_labels)

    estimator = NearestNeighbors(n_neighbors=5)
    print("*" * 10, " Neighbors of AST Node types:", "*" * 10)
    neighbors_table(estimator, X, y, true_labels)

from sklearn.preprocessing import scale
def show_authors(model):
    dataset_folder = "dataset700"
    trees, tree_labels, lable_problems = parse_src_files(dataset_folder)
    # trees, tree_labels = pick_subsets(trees, tree_labels, labels=2)
    classes_, y = np.unique(tree_labels, return_inverse=True)
    trees_X,_ = evaluate(model,trees,y)

    # estimator = DBSCAN(eps=0.3, min_samples=10)
    print("\n")
    print("*" * 10, " Cluster Authors:", "*" * 10)
    estimator = KMeans(n_clusters=5,init='k-means++')

    X_avg = []
    y_avg = []
    label_avg = []
    for label in np.unique(tree_labels):
        X_avg.append(trees_X[tree_labels == label].mean(axis=0))
        y_avg.append(label)
        label_avg.append(classes_[np.argwhere(classes_ == label)[0][0]])

    X_avg = np.vstack((X_avg))
    y_avg = np.array(y_avg)
    label_avg = np.array(label_avg)

    X_scale = scale(X_avg)
    cluster_table(estimator,  trees_X, y, tree_labels)
    cluster_plot(estimator,  X_scale, y_avg, label_avg)

    estimator = NearestNeighbors(n_neighbors=5)
    print("*" * 10, " Neighbors of Authors:", "*" * 10)
    neighbors_table(estimator, trees_X, y, tree_labels)

if __name__ == "__main__":
    path = R"C:\Users\bms\Files\study\DC\Experiments\results\best results\saved_models\1_lstm_1_250_70_labels10_epoch_135.my"
    model = RecursiveLSTM(250, 70,1,0.2, classes=None)
    serializers.load_npz(path,model)
    print_model(model, depth=1, output=sys.stdout)

    #show_embeding(model)
    show_authors(model)






