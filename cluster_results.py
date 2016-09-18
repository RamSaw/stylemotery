




import os

import sys
from chainer import serializers
import numpy as np
from sklearn.manifold import TSNE

from ast_tree.ast_parser import AstNodes
from models import RecursiveLSTM
from utils.extract_loss_progress import parse_result_file
from utils.fun_utils import print_model
from utils.graph import plot_all, plot_each
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA,KernelPCA


def cluster_plot(estimator,X,y,labels):
    # estimator = PCA(n_components=2)
    # X_r = X
    X_r = estimator.fit_transform(X)
    core_samples_mask = np.zeros_like(estimator.labels_, dtype=bool)
    # core_samples_mask[estimator.core_sample_indices_] = True
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

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def PCA_plot(estimator,X,y,labels):
    # estimator = PCA(n_components=2)
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
if __name__ == "__main__":
    path = R"C:\Users\bms\Files\study\DC\Experiments\results\Sept 2nd Week\saved_models\1_lstm_1_dropout_1000_70_labels10_epoch_39.my"
    model = RecursiveLSTM(1000, 70, classes=None)
    serializers.load_npz(path,model)
    print_model(model, depth=1, output=sys.stdout)

    # X = model.w.W.data
    # y = np.arange(X.shape[0])
    # true_labels = [v for k,v in ([(0, 'AMV0'), (1, 'Argaen0'), (2, 'Bastiandantilus0'), (3, 'Binnie0'), (4, 'BlackEagle0'), (5, 'ChevalierMalFet0'), (6, 'Coconut Big0'), (7, 'Eko0'), (8, 'EnTerr0'), (9, 'Entropy0'), (10, 'Fizu0'), (11, 'GauravRai0'), (12, 'Greatlemer0'), (13, 'IdahoJacket0'), (14, 'IdoLivneh0'), (15, 'J3ffreySmith0'), (16, 'Michael0'), (17, 'NaN0'), (18, 'Nooodles0'), (19, 'Phayr0'), (20, 'RalfKistner0'), (21, 'SickMath0'), (22, 'ToR0'), (23, 'YOBA0'), (24, 'addie90000'), (25, 'alexamici0'), (26, 'ana valeije0'), (27, 'anb0'), (28, 'aosuka0'), (29, 'bigOnion0'), (30, 'caethan0'), (31, 'cathco0'), (32, 'cheilman0'), (33, 'fractal0'), (34, 'gepa0'), (35, 'gizzywump0'), (36, 'graygrass0'), (37, 'hannanaha0'), (38, 'imakaramegane0'), (39, 'int n0'), (40, 'j4b0'), (41, 'jakab9220'), (42, 'jgaten0'), (43, 'joegunrok0'), (44, 'kawasaki0'), (45, 'kmod0'), (46, 'lookingfor0'), (47, 'max bublis0'), (48, 'mth0'), (49, 'netsuso0'), (50, 'nlse0'), (51, 'nwin0'), (52, 'oonishi0'), (53, 'pavlovic0'), (54, 'pawko0'), (55, 'pek0'), (56, 'pyronimous0'), (57, 'radkokotev0'), (58, 'rainmayecho0'), (59, 'raja baz0'), (60, 'rmmh0'), (61, 'ronnodas0'), (62, 'royf0'), (63, 'serialk0'), (64, 'shishkander0'), (65, 'taichino0'), (66, 'tama eguchi0'), (67, 'xoxie0'), (68, 'yordanmiladinov0'), (69, 'ziyan0')])]

    # estimator  =  TSNE(n_components=2, random_state=None)#KernelPCA(n_components=2,kernel="rbf")#PCA(n_components=2)#TSNE(n_components=2, random_state=None)#KernelPCA(n_components=2,kernel="rbf")#PCA(n_components=2)
    # PCA_plot(estimator,X,y,true_labels)

    X = model.embed.W.data
    y = np.arange(X.shape[0])

    ast_nodes = AstNodes()
    true_labels = ast_nodes.nodetypes + [ast_nodes.NONE]

    estimator  =  PCA(n_components=2) #TSNE(n_components=2, random_state=None)#KernelPCA(n_components=2,kernel="rbf")#PCA(n_components=2)
    PCA_plot(estimator,X,y,true_labels)



    # estimator = DBSCAN(eps=0.3, min_samples=10)
    # estimator = KMeans(n_clusters=5,init='k-means++')
    # cluster_plot(estimator, X, y, true_labels)



