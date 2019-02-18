import matplotlib
#matplotlib.use('Agg')
import json
from itertools import cycle
from operator import itemgetter
import os
from random import Random
import traceback
from chainer.functions import dropout
import itertools
from matplotlib import gridspec
import matplotlib.pyplot as plt
from numpy.core.multiarray import interp
from sklearn.metrics import auc
import numpy as np


def draw_roc_curve(ax, tpr, fpr, xlabel,zoom=False,thresholds=None, lw=1,markevery=3, line_style='-',last_time=False):#,color=None,marker='*', ):
    if last_time:
        ax.plot([0, 1], [0, 1], 'r--',label="Straight", color=(0.6, 0.6, 0.6))
        if zoom:
            ax.set_xlim([0.0, 0.2])
            ax.set_ylim([0.9, 1.02])
            ax.set_xticks(np.linspace(0, 0.2, 5))
            ax.set_yticks(np.linspace(0.9, 1.0, 3))
        else:
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            ax.set_xticks(np.linspace(0, 1, 11))
            ax.set_yticks(np.linspace(0, 1, 11))
        ax.set_xlabel('False Positive Rate')
        ax.legend(loc="lower right", prop={'size': 8},numpoints=1)
        ax.set_ylabel('True Positive Rate')
        # ax.set_title('Receiver operating characteristic example')
    else:
        ax.plot(fpr, tpr, line_style, label=xlabel, lw=lw , markevery=markevery)#,color=color,markers=marker)


def draw_recall_precision_curve(ax3, precision_curve, recall_curve, average_precision):
    ax3.plot(recall_curve, precision_curve, label='Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_xlim([-0.05, 1.05])
    ax3.set_title('Precision-Recall curve')
    ax3.legend(loc="lower left", prop={'size': 4})

def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                          local_path=None):
    fig = plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if local_path:
        fig.savefig(os.path.join(local_path,"cm"), dpi=900)
    else:
        plt.show()

def draw_confusion_matrix(figure, cmx, labels, cm):
    cax = cmx.matshow(cm)
    cmx.set_title('Confusion matrix')
    cmx.set_ylabel('True label')
    cmx.set_xlabel('Predicted label')
    norm_conf = []
    for i in cm:
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if float(a) != 0:
                tmp_arr.append(float(j) / float(a))
            else:
                tmp_arr.append(0.0)
        norm_conf.append(tmp_arr)
    # res = cmx.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')
    width = len(cm)
    height = len(cm[0])
    for x in range(width):
        for y in range(height):
            cmx.annotate(str(cm[x, y]), xy=(y, x), horizontalalignment='center', verticalalignment='center',)
    figure.colorbar(cax)
    plt.xticks(range(width), labels[:width])
    plt.yticks(range(height), labels[:height])




class GraphicPresenter():
    def __init__(self,roc=True,zoom_roc=True,timeline=False,adversial=False,incremental=False,cm=False,pca=False,density=False,local_path=None):
        self.roc = roc
        self.zoom_roc = zoom_roc
        self.cm = cm
        self.timeline = timeline
        self.incremental = incremental
        self.adversial = adversial
        self.pca = pca
        self.density = density
        self.local_path = local_path
        self.clf_styls = None

    def present(self,results):
        # linestyles = ['--']
        # markers = ['o','s','*','+','x']
        markers = itertools.cycle(('s', 'p', 'o','^','8','+','x','*', 'd'))
        linestyles = itertools.cycle(('-', '--', '-.', ':'))
        colors = itertools.cycle(('b', 'g', 'r', 'k','m'))
        # self.clf_styls = json.loads("{\"Raw 4-Grams\": \"b-s\", \"Binary 4-Grams\": \"g--p\", \"Binary 3-Grams\": \"m-8\", \"Raw 2-Grams\": \"k:^\", \"Binary 2-Grams\": \"r-.o\", \"Raw 3-Grams\": \"b--+\"}")
        if not self.clf_styls:
            self.clf_styls = {}
            for dataset in results:
                for clf_name in results[dataset]:
                    if clf_name not in self.clf_styls:
                        self.clf_styls[clf_name] = next(colors) + next(linestyles) + next(markers)
            print(json.dumps(self.clf_styls))

        for dataset in results:
            print("present "+dataset+" ...")
            clf_rand = Random().randint(1, 100)
            if self.roc or self.zoom_roc:
                self.present_roc(dataset,results[dataset],clf_rand)
            if self.pca:
                self.present_pca(dataset,results[dataset],clf_rand)
            if self.density:
                self.present_density(dataset, results[dataset],clf_rand)
            if self.timeline:
                self.present_timeline(dataset, results[dataset],clf_rand)
            if self.incremental:
                self.present_incremental(dataset, results[dataset], clf_rand)
            if self.adversial:
                self.present_adversial(dataset, results[dataset], clf_rand)
                self.present_adversial_zoom(dataset, results[dataset], clf_rand)
        if self.pca:
            self.present_pca_all(results,clf_rand)
        plt.close('all')

    def present_roc(self,dataset,results,clf_rand):
        print("\tgenerate roc")
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        roc_result = self.accumlate_roc(results)
        for idx,(clf,auc_val,mean_tpr,mean_fpr) in enumerate(sorted(roc_result,key= itemgetter(1),reverse=True)):
            line_style = self.clf_styls[clf]
            draw_roc_curve(ax1, mean_tpr,mean_fpr,xlabel=clf + ' ROC (area = %0.4f)' % auc_val,lw=1, line_style=line_style)
        draw_roc_curve(ax1, None, None, None,zoom=False,last_time=True)
        ax1.grid('on')
        figure.savefig(os.path.join(self.local_path, dataset+"_roc_%s" % clf_rand), dpi=900)
        figure.clear()

        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        ax1.grid('on')
        roc_result = self.accumlate_roc(results)
        for idx,(clf,auc_val,mean_tpr,mean_fpr) in enumerate(sorted(roc_result,key= itemgetter(1),reverse=True)):
            line_style = self.clf_styls[clf]
            draw_roc_curve(ax1, mean_tpr,mean_fpr, xlabel=clf + ' ROC (area = %0.4f)' % auc_val,lw=1, line_style=line_style)
        draw_roc_curve(ax1, None, None, None,zoom=True,last_time=True)
        # draw_roc_curve(ax1, None, None, None,last_time=True)
        figure.savefig(os.path.join(self.local_path, dataset+"_roc_%s_zoom" % clf_rand), dpi=900)
        figure.clear()

    def present_pca2(self,dataset,results,clf_rand):
        # pca ratio
        print("\tgenerate pca")
        for clf in results:
            result = results[clf]
            ratios_ = result['pca']
            max_len = max([len(ratio) for ratio in ratios_])
            accu_ratio = np.zeros(max_len)
            for ratio in ratios_:
                accu_ratio[:len(ratio)] += ratio
            pca_ratio = accu_ratio/ result['kfold']
            figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])
            ax1 = plt.subplot(gs[0])
            accum = np.cumsum(pca_ratio)
            ax1.plot(accum)
            max_ind = np.argmax(accum)
            ax1.axvline(max_ind, color='k',linestyle='--')
            ax1.set_xlabel('Number of singular variables')
            ax1.set_ylabel('cumulative explained variance')
            ax1.set_xlim([0, 1000])

            ax2 = plt.subplot(gs[2])
            ax2.bar(np.arange(pca_ratio.shape[0]),pca_ratio)
            ax2.set_xlabel('Singular Variable')
            ax2.set_ylabel('Explained Variance')
            figure.savefig(os.path.join(self.local_path, dataset+"_"+clf+"_pca_%s" % clf_rand), dpi=900)

    def present_pca(self, dataset, results, clf_rand):
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            pca_components, TPR, FPR = results[clf_name]['pca_tpr_fpr']
            line_style = self.clf_styls[clf_name]
            ax1.plot(pca_components, TPR, line_style, label=clf_name+" TPR", markevery=5)  # linestyle=)
            ax1.plot(pca_components, FPR, line_style, label=clf_name+" FPR", markevery=5)  # linestyle=)
            max_ind = np.argmax(TPR-FPR)
            ax1.axvline(pca_components[max_ind], color=line_style[0],linestyle='--')
        ax1.set_xlabel('Number of singular variables')
        # ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        # ax1.set_ylabel('Rate')
        ax1.grid('on')
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax2 = plt.subplot(gs[2])
        for clf in results:
            result = results[clf]
            ratios_ = result['pca']
            max_len = max([len(ratio) for ratio in ratios_])
            accu_ratio = np.zeros(max_len)
            for ratio in ratios_:
                accu_ratio[:len(ratio)] += ratio
            pca_ratio = accu_ratio / result['kfold']
            ax2.bar(np.arange(pca_ratio.shape[0]), pca_ratio)
            ax2.set_xlabel('Singular Variable')
            ax2.set_ylabel('Explained Variance')
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        figure.savefig(os.path.join(self.local_path, dataset + "_" + "_pca_%s" % clf_rand), dpi=900)
        figure.clear()

        # separate figure
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            pca_components, TPR, FPR = results[clf_name]['pca_tpr_fpr']
            line_style = self.clf_styls[clf_name]
            ax1.plot(pca_components, TPR, "b"+line_style[1:], label=clf_name+"_TPR", markevery=10)  # linestyle=)
            ax1.plot(pca_components, FPR, "r"+line_style[1:], label=clf_name+"_FPR", markevery=10)  # linestyle=)
            max_ind = np.argmax(TPR-FPR)
            ax1.axvline(pca_components[max_ind], color='k',linestyle='--')
        ax1.set_xlabel('Number of singular variables')
        # ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        # ax1.set_ylabel('Rate')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        figure.savefig(os.path.join(self.local_path, dataset + "_pca_rate_%s" % clf_rand),bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=900)
        figure.clear()

    def present_pca_all(self, db_results, clf_rand):
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(len(db_results), len(db_results), width_ratios=[len(db_results), len(db_results)])
        explained1 = False
        explained2 = False
        for idx,dataset in enumerate(db_results):
            results = db_results[dataset]
            ax2 = plt.subplot(gs[idx])
            for clf in results:
                result = results[clf]
                ratios_ = result['pca']
                max_len = max([len(ratio) for ratio in ratios_])
                accu_ratio = np.zeros(max_len)
                for ratio in ratios_:
                    accu_ratio[:len(ratio)] += ratio
                pca_ratio = accu_ratio / result['kfold']
                ax2.bar(np.arange(pca_ratio.shape[0]), pca_ratio)
                ax2.set_xlabel('Number of singular variables')
                if not explained1:
                    ax2.set_ylabel('Explained Variance')
                    explained1 = True

            ax1 = plt.subplot(gs[len(db_results)+idx])
            for idx, clf_name in enumerate(results):
                pca_components, TPR, FPR = results[clf_name]['pca_tpr_fpr']
                line_style = self.clf_styls[clf_name]
                ax1.plot(pca_components, TPR, "b" + line_style[1:], label=clf_name + "_TPR", markevery=10)
                ax1.plot(pca_components, FPR, "r" + line_style[1:], label=clf_name + "_FPR", markevery=10)
                max_ind = np.argmax(TPR - FPR)
                ax1.axvline(pca_components[max_ind], color='k', linestyle='--')
            ax1.set_xlabel('Number of singular variables')
            ax1.set_xticks(pca_components[::10])#np.linspace(0, max(pca_components), len(pca_components)))
            if not explained2:
                ax1.set_ylabel('Rate')
                explained2 = True
            ax1.grid('on')
        # ax1.legend(loc="upper center", prop={'size': 8}, numpoints=1)
        handles, labels = ax1.get_legend_handles_labels()
        lgd = figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        figure.savefig(os.path.join(self.local_path, "all_" + "_pca_%s" % clf_rand),bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=900)
        # figure.savefig(os.path.join(self.local_path, "all_" + "_pca_%s" % clf_rand), dpi=900)
        figure.clear()


    def present_density(self,dataset,results,clf_rand):
        print("\tgenerate density")
        for clf in results[dataset]:
            result = results[dataset][clf]
            scores = result['scores']
            pos_scores = []
            neg_scores = []
            for (y_scores,y_test) in scores:
                pos_scores.extend(y_scores[y_test == 1].ravel())
                neg_scores.extend(y_scores[y_test == -1].ravel())

            # max_len = max([len(score) for score in pos_scores])
            # avg_pos = np.zeros(max_len)
            # for score in pos_scores:
            #     avg_pos[:len(score)] += score
            # avg_pos = avg_pos/ result['kfold']
            #
            # max_len = max([len(score) for score in neg_scores])
            # avg_neg = np.zeros(max_len)
            # for score in neg_scores:
            #     avg_neg[:len(score)] += score
            # avg_neg = avg_neg/ result['kfold']

            figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
            gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
            # weights = np.ones_like(pos_scores)/float(len(pos_scores))
            # ax1.hist(pos_scores, bins=10 , weights=weights,normed=0,label='positive label')

            # weights = np.ones_like(neg_scores)/float(len(neg_scores))
            # ax1.hist(neg_scores, bins=10 , weights=weights,normed=0,label='negative label')
            ax1 = plt.subplot(gs[0])
            bins = np.linspace(min(neg_scores), max(pos_scores), 100)
            ax1.hist(pos_scores,bins, label='positive label',color='blue')
            ax1.hist(neg_scores,bins, label='negative label',color='red')
            ax1.legend(loc='upper right')

            # ax2 = plt.subplot(gs[1])
            # bins = np.linspace(min(neg_scores), max(neg_scores), 100)
            # ax2.hist(neg_scores,bins, label='negative label',color='red')
            # ax2.legend(loc='upper right')

            figure.savefig(os.path.join(self.local_path, dataset+"_"+clf+"_density_%s" % clf_rand), dpi=900)

    def present_incremental(self,dataset,results,clf_rand):
        # separate figure
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        self.clf_styls = eval("{\"Retrained LR 2-Grams\": \"b-s\", \"LR 2-Grams\": \"g--p\", \"LR 3-Grams\": \"k:^\", \"Retrained LR 3-Grams\": \"r-.o\"}")
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            epochs, TPR, FPR = results[clf_name]['incremental']
            line_style = self.clf_styls[clf_name]
            ax1.plot(epochs, TPR, line_style, label=clf_name, markevery=5)  # linestyle=)
            # ax1.plot(epochs, FPR, line_style, label=clf_name, markevery=1)  # linestyle=)
            # ax1.plot(epochs, TPR, "b"+line_style[1:], label=clf_name+"_TPR", markevery=1)  # linestyle=)
            # ax1.plot(epochs, FPR, "r"+line_style[1:], label=clf_name+"_FPR", markevery=1)  # linestyle=)
            # max_ind = np.argmax(TPR-FPR)
            # ax1.axvline(epochs[max_ind], color='k',linestyle='--')
        # ax1.set_xlim([0.0, 0.2])
        ax1.set_ylim([0.9, 1.02])
        # ax1.set_xticks(np.linspace(0, 0.3, 5))
        ax1.set_yticks(np.linspace(0.9, 1.0, 5))
        ax1.grid('on')

        ax1.set_xlabel('Epoch')
        # ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        ax1.set_ylabel('TPR')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        figure.savefig(os.path.join(self.local_path, dataset + "_incremental_%s" % clf_rand),bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=900)
        figure.clear()

    def present_adversial_zoom(self,dataset,results,clf_rand):
        # separate figure
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        # self.clf_styls = eval("{\"Retrained LR 2-Grams\": \"b-s\", \"LR 2-Grams\": \"g--p\", \"LR 3-Grams\": \"k:^\", \"Retrained LR 3-Grams\": \"r-.o\"}")
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            trace_len, TPR,F1 = results[clf_name]['adversial']
            line_style = self.clf_styls[clf_name]
            ax1.plot(trace_len, TPR, line_style, label=clf_name, markevery=5)  # linestyle=)
            # ax1.plot(epochs, FPR, line_style, label=clf_name, markevery=1)  # linestyle=)
            # ax1.plot(epochs, TPR, "b"+line_style[1:], label=clf_name+"_TPR", markevery=1)  # linestyle=)
            # ax1.plot(epochs, FPR, "r"+line_style[1:], label=clf_name+"_FPR", markevery=1)  # linestyle=)
            # max_ind = np.argmax(TPR-FPR)
            # ax1.axvline(epochs[max_ind], color='k',linestyle='--')
        # ax1.set_xlim([0.0, 0.2])
        ax1.set_ylim([0.9, 1.01])
        # ax1.set_xticks(np.linspace(0, 0.3, 5))
        # ax1.set_yticks(np.linspace(0.9, 1.0, 5))
        ax1.grid('on')

        ax1.set_xlabel('Trace length')
        # ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        ax1.set_ylabel('TPR')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        figure.savefig(os.path.join(self.local_path, dataset + "_adversial_zoom_tpr_%s" % clf_rand),bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=900)
        figure.clear()

        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        # self.clf_styls = eval("{\"Retrained LR 2-Grams\": \"b-s\", \"LR 2-Grams\": \"g--p\", \"LR 3-Grams\": \"k:^\", \"Retrained LR 3-Grams\": \"r-.o\"}")
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            trace_len, _,F1 = results[clf_name]['adversial']
            line_style = self.clf_styls[clf_name]
            ax1.plot(trace_len, F1, line_style, label=clf_name, markevery=5)  # linestyle=)
            # ax1.plot(epochs, FPR, line_style, label=clf_name, markevery=1)  # linestyle=)
            # ax1.plot(epochs, TPR, "b"+line_style[1:], label=clf_name+"_TPR", markevery=1)  # linestyle=)
            # ax1.plot(epochs, FPR, "r"+line_style[1:], label=clf_name+"_FPR", markevery=1)  # linestyle=)
            # max_ind = np.argmax(TPR-FPR)
            # ax1.axvline(epochs[max_ind], color='k',linestyle='--')
        # ax1.set_xlim([0.0, 0.2])
        ax1.set_ylim([0.9, 1.02])
        # ax1.set_xticks(np.linspace(0, 0.3, 5))
        # ax1.set_yticks(np.linspace(0.9, 1.0, 5))
        ax1.grid('on')

        ax1.set_xlabel('Trace length')
        # ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        ax1.set_ylabel('F1')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        figure.savefig(os.path.join(self.local_path, dataset + "_adversial_zoom_f1_%s" % clf_rand),bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=900)
        figure.clear()

    def present_adversial(self,dataset,results,clf_rand):
        # separate figure
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        # self.clf_styls = eval("{\"Retrained LR 2-Grams\": \"b-s\", \"LR 2-Grams\": \"g--p\", \"LR 3-Grams\": \"k:^\", \"Retrained LR 3-Grams\": \"r-.o\"}")
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            trace_len, TPR,F1 = results[clf_name]['adversial']
            line_style = self.clf_styls[clf_name]
            ax1.plot(trace_len, TPR, line_style, label=clf_name, markevery=5)  # linestyle=)
            # ax1.plot(epochs, FPR, line_style, label=clf_name, markevery=1)  # linestyle=)
            # ax1.plot(epochs, TPR, "b"+line_style[1:], label=clf_name+"_TPR", markevery=1)  # linestyle=)
            # ax1.plot(epochs, FPR, "r"+line_style[1:], label=clf_name+"_FPR", markevery=1)  # linestyle=)
            # max_ind = np.argmax(TPR-FPR)
            # ax1.axvline(epochs[max_ind], color='k',linestyle='--')
        # ax1.set_xlim([0.0, 0.2])
        ax1.set_ylim([0.5, 1.02])
        # ax1.set_xticks(np.linspace(0, 0.3, 5))
        # ax1.set_yticks(np.linspace(0.9, 1.0, 5))
        ax1.grid('on')

        ax1.set_xlabel('Trace length')
        # ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        ax1.set_ylabel('TPR')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        figure.savefig(os.path.join(self.local_path, dataset + "_adversial_tpr_%s" % clf_rand),bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=900)
        figure.clear()

        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        # self.clf_styls = eval("{\"Retrained LR 2-Grams\": \"b-s\", \"LR 2-Grams\": \"g--p\", \"LR 3-Grams\": \"k:^\", \"Retrained LR 3-Grams\": \"r-.o\"}")
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            trace_len, _,F1 = results[clf_name]['adversial']
            line_style = self.clf_styls[clf_name]
            ax1.plot(trace_len, F1, line_style, label=clf_name, markevery=5)  # linestyle=)
            # ax1.plot(epochs, FPR, line_style, label=clf_name, markevery=1)  # linestyle=)
            # ax1.plot(epochs, TPR, "b"+line_style[1:], label=clf_name+"_TPR", markevery=1)  # linestyle=)
            # ax1.plot(epochs, FPR, "r"+line_style[1:], label=clf_name+"_FPR", markevery=1)  # linestyle=)
            # max_ind = np.argmax(TPR-FPR)
            # ax1.axvline(epochs[max_ind], color='k',linestyle='--')
        # ax1.set_xlim([0.0, 0.2])
        ax1.set_ylim([0.5, 1.02])
        # ax1.set_xticks(np.linspace(0, 0.3, 5))
        # ax1.set_yticks(np.linspace(0.9, 1.0, 5))
        ax1.grid('on')

        ax1.set_xlabel('Trace length')
        # ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        ax1.set_ylabel('F1')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        figure.savefig(os.path.join(self.local_path, dataset + "_adversial_f1_%s" % clf_rand),bbox_extra_artists=(lgd,),bbox_inches='tight', dpi=900)
        figure.clear()


    def present_timeline(self,dataset,results,clf_rand):
        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        for idx,clf_name in enumerate(results):
            indices,train_time, _ = results[clf_name]['time']
            line_style = self.clf_styls[clf_name]
            ax1.plot(indices,train_time, line_style,label=clf_name,markevery=3)#linestyle=)
        ax1.set_xlabel('Number of traces')
        ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        ax1.set_ylabel('Time (Second)')
        ax1.grid('on')
        figure.savefig(os.path.join(self.local_path, dataset+"_train_timeline_%s" % clf_rand), dpi=900)
        figure.clear()

        figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
        gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        for idx, clf_name in enumerate(results):
            indices, _, eval_time = results[clf_name]['time']
            line_style = self.clf_styls[clf_name]
            ax1.plot(indices, eval_time, line_style, label=clf_name, markevery=3)  # linestyle=)
        ax1.set_xlabel('Number of traces')
        ax1.legend(loc="upper left", prop={'size': 8}, numpoints=1)
        ax1.set_ylabel('Time (Second)')
        ax1.grid('on')
        figure.savefig(os.path.join(self.local_path, dataset + "_eval_timeline_%s" % clf_rand), dpi=900)
        figure.clear()

    def accumlate_roc(self,dataset):
        curves = []
        for clf in dataset:
            result = dataset[clf]
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)
            mean_thr = []
            for tpr, fpr,threshold in result['all_tpr']:
                mean_tpr += interp(mean_fpr, fpr, tpr)
                mean_tpr[0] = 0.0
                mean_thr.extend(threshold)
            mean_tpr /= result['kfold']
            mean_tpr[-1] = 1.0
            curves.append((clf,auc(mean_fpr, mean_tpr),mean_tpr,mean_fpr))
        return curves


if __name__ == "__main__":
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)


    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    # fig = plt.figure()
    # gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    # ax1 = plt.subplot(gs[0])
    # draw_confusion_matrix(fig,cmx=ax1,labels=y_test,cm=cnf_matrix)

    # plt.show()