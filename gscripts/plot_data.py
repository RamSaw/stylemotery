import matplotlib
# matplotlib.use('Agg')
import json
from itertools import cycle
from operator import itemgetter
import os
from random import Random
import traceback
from chainer.functions import Dropout
import itertools
from matplotlib import gridspec
import matplotlib.pyplot as plt
from numpy.core.multiarray import interp
from sklearn.metrics import auc
import numpy as np

basefolder = ""


def per(l):
    return ["{0:<2.2f}".format(s*100) for s in l]

def plot_relax():
    lstm1l =  per([0.857142857,0.942857143,0.985714286,0.992857143])
    blstm1l = per([0.9,0.957142857,0.964285714,0.971428571])
    rf = per([0.742857143,0.884285714,0.941428571,0.98])

    print(lstm1l)
    markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))
    figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
    gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    # ax1.set_xticklabels([1,5,10,15])

    # ax1.set_xlim([70.00, 100.00])
    ax1.set_ylim([60.00, 100.02])
    ax1.set_xticks([1,5,10,15])#np.linspace(1, 10 , 2))
    ax1.set_yticks([60,65,70,75,80,85,90,95,100])#np.linspace(60, 100, 8))
    ax1.plot([1,5,10,15],lstm1l, (next(colors) + next(linestyles) + next(markers)), label="LSTM", lw=1, markevery=1)
    ax1.plot([1,5,10,15],blstm1l, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM", lw=1, markevery=1)
    ax1.plot([1,5,10,15],rf, (next(colors) + next(linestyles) + next(markers)), label="RandomForest", lw=1, markevery=1)
    for tick in ax1.get_xaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()
    ax1.set_xlabel('Top Relaxed Classification')
    ax1.legend(loc="lower right", prop={'size': 8}, numpoints=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid('on')
    plt.show()

    figure.savefig(os.path.join(R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\relaxed",
                                "relax1"), dpi=900)
    figure.clear()

def plot_complex():
    lstm1l =  per([0.8,0.857142857,0.8357,0.8357])
    lstm12 =  per([0.678571429,0.728571429,0.7571,0.7714])
    blstml1 = per([0.792857143,0.9,0.8571,0.85])
    blstm12 = per([0.678571429,0.728571429,0.764285714,0.757142857])
    x_values = [50,100,250,500]

    markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))
    figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
    gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    # ax1.set_xticklabels([1,5,10,15])

    # ax1.set_xlim([70.00, 100.00])
    ax1.set_ylim([60.00, 100.02])
    ax1.set_xticks(x_values)#np.linspace(1, 10 , 2))
    ax1.set_yticks([60,65,70,75,80,85,90,95,100])#np.linspace(60, 100, 8))

    ax1.plot(x_values,lstm1l, (next(colors) + next(linestyles) + next(markers)), label="LSTM-1 Layer", lw=1, markevery=1)
    ax1.plot(x_values,blstml1, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM-1 Layer", lw=1, markevery=1)
    ax1.plot(x_values,lstm12, (next(colors) + next(linestyles) + next(markers)), label="LSTM-2 Layers", lw=1, markevery=1)
    ax1.plot(x_values,blstm12, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM-2 Layers", lw=1, markevery=1)
    for tick in ax1.get_xaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()
    ax1.set_xlabel('Number of Hidden Units')
    ax1.legend(loc="lower right", prop={'size': 8}, numpoints=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid('on')
    plt.show()

    figure.savefig(os.path.join(R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\scale",
                                "scale"), dpi=900)
    figure.clear()


def plot_scale():
    lstm1l =  per(reversed([0.857142857,0.942857143,0.985714286,0.992857143]))
    blstm1l = per(reversed([0.9,0.957142857,0.964285714,0.971428571]))
    rf = per(reversed([0.742857143,0.884285714,0.941428571,0.98]))
    x_values = [5,25,55,70]

    markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))
    figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
    gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    # ax1.set_xticklabels([1,5,10,15])

    # ax1.set_xlim([70.00, 100.00])
    ax1.set_ylim([60.00, 100.02])
    ax1.set_xticks(x_values)#np.linspace(1, 10 , 2))
    ax1.set_yticks([60,65,70,75,80,85,90,95,100])#np.linspace(60, 100, 8))
    ax1.plot(x_values,lstm1l, (next(colors) + next(linestyles) + next(markers)), label="LSTM", lw=1, markevery=1)
    ax1.plot(x_values,blstm1l, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM", lw=1, markevery=1)
    ax1.plot(x_values,rf, (next(colors) + next(linestyles) + next(markers)), label="RandomForest", lw=1, markevery=1)
    for tick in ax1.get_xaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()
    ax1.set_xlabel('Number of Programmers')
    ax1.legend(loc="lower left", prop={'size': 12}, numpoints=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid('on')
    plt.show()

    figure.savefig(os.path.join(R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\scale",
                                "scale"), dpi=900)
    figure.clear()

if __name__ == "__main__":
    # plot_relax()
    # plot_complex()
    plot_scale()