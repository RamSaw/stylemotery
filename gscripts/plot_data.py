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
markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
linestyles = itertools.cycle(('-', '--', '-.', ':'))
colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))

lstm_style = (next(colors) + next(linestyles) + next(markers))
bilstm_style = (next(colors) + next(linestyles) + next(markers))
rf_style = (next(colors) + next(linestyles) + next(markers))

def per(l):
    return ["{0:<2.2f}".format(s) for s in l]

def plot_relax():
    lstm1l =  per([86.36, 94.29, 98.57, 99.29])
    blstm1l = per([88.86, 95.71, 96.43, 97.14])
    rf = per([72.90, 89.57, 95.00, 97.00])

    figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
    gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    # ax1.set_xticklabels([1,5,10,15])

    # ax1.set_xlim([70.00, 100.00])
    ax1.set_ylim([60.00, 100.02])
    ax1.set_xticks([1,5,10,15])#np.linspace(1, 10 , 2))
    ax1.set_yticks([60,65,70,75,80,85,90,95,100])#np.linspace(60, 100, 8))

    ax1.plot([1,5,10,15],lstm1l, lstm_style, label="LSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot([1,5,10,15],blstm1l, bilstm_style, label="BiLSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot([1,5,10,15],rf, rf_style, label="Random Forest", lw=1, markevery=1)
    for tick in ax1.get_xaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()
    ax1.set_xlabel('Top Prediction')
    ax1.legend(loc="lower right", prop={'size': 9.5}, numpoints=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid('on')
    # plt.show()
    #
    figure.savefig(os.path.join(R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\relaxed","relax1"), dpi=900)
    # figure.clear()

def plot_complex():
    lstm1l =  per([0.8,0.857142857,0.8357,0.8357])
    lstm12 =  per([0.678571429,0.728571429,0.7571,0.7714])
    blstml1 = per([0.792857143,0.9,0.8571,0.85])
    blstm12 = per([0.678571429,0.728571429,0.764285714,0.757142857])
    x_values = [50,100,250,500]

    figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
    gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    # ax1.set_xticklabels([1,5,10,15])

    # ax1.set_xlim([70.00, 100.00])
    ax1.set_ylim([60.00, 100.02])
    ax1.set_xticks(x_values)#np.linspace(1, 10 , 2))
    ax1.set_yticks([60,65,70,75,80,85,90,95,100])#np.linspace(60, 100, 8))

    ax1.plot(x_values,lstm1l, (next(colors) + next(linestyles) + next(markers)), label="LSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot(x_values,blstml1, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot(x_values,lstm12, (next(colors) + next(linestyles) + next(markers)), label="LSTM (2-Layers, 250-Units)", lw=1, markevery=1)
    ax1.plot(x_values,blstm12, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM (2-Layers, 250-Unites)", lw=1, markevery=1)

    for tick in ax1.get_xaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()
    ax1.set_xlabel('Hidden Units')
    ax1.legend(loc="lower right", prop={'size': 9.5}, numpoints=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid('on')
    # plt.show()

    figure.savefig(os.path.join(R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\scale",
                                "scale"), dpi=900)
    figure.clear()

def plot_classifiers():
    lstm1l =  per([0.8,0.857142857,0.8357,0.8357])
    lstm12 =  per([0.678571429,0.728571429,0.7571,0.7714])
    blstml1 = per([0.792857143,0.9,0.8571,0.85])
    blstm12 = per([0.678571429,0.728571429,0.764285714,0.757142857])
    x_values = [50,100,250,500]

    figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
    gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    # ax1.set_xticklabels([1,5,10,15])

    # ax1.set_xlim([70.00, 100.00])
    ax1.set_ylim([60.00, 100.02])
    ax1.set_xticks(x_values)#np.linspace(1, 10 , 2))
    ax1.set_yticks([60,65,70,75,80,85,90,95,100])#np.linspace(60, 100, 8))

    ax1.plot(x_values,lstm1l, (next(colors) + next(linestyles) + next(markers)), label="LSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot(x_values,blstml1, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot(x_values,lstm12, (next(colors) + next(linestyles) + next(markers)), label="LSTM (2-Layers, 250-Units)", lw=1, markevery=1)
    ax1.plot(x_values,blstm12, (next(colors) + next(linestyles) + next(markers)), label="BiLSTM (2-Layers, 250-Unites)", lw=1, markevery=1)

    for tick in ax1.get_xaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()
    ax1.set_xlabel('Hidden Units')
    ax1.legend(loc="lower right", prop={'size': 9.5}, numpoints=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid('on')
    # plt.show()

    figure.savefig(os.path.join(R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\scale",
                                "scale"), dpi=900)
    figure.clear()


def plot_scale():
    lstm1l =  per([100.00, 92.00, 86.43, 86.36])
    blstm1l = per([100.00, 96.00, 89.09, 88.86])
    rf = per([90.00, 80.00, 77.45, 72.90])
    x_values = [5,25,55,70]

    figure = plt.figure(figsize=(8, 6))  # num=2,figsize=(5,5),dpi=200
    gs = gridspec.GridSpec(1, 1, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    # ax1.set_xticklabels([1,5,10,15])

    # ax1.set_xlim([70.00, 100.00])
    ax1.set_ylim([60.00, 100.02])
    ax1.set_xticks(x_values)#np.linspace(1, 10 , 2))
    ax1.set_yticks([60,65,70,75,80,85,90,95,100])#np.linspace(60, 100, 8))
    ax1.plot(x_values,lstm1l, lstm_style, label="LSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot(x_values,blstm1l,bilstm_style, label="BiLSTM (1-Layer, 100-Units)", lw=1, markevery=1)
    ax1.plot(x_values,rf,rf_style, label="Random Forest", lw=1, markevery=1)
    for tick in ax1.get_xaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()
    ax1.set_xlabel('Source Code Authors')
    ax1.legend(loc="lower left", prop={'size': 9.5}, numpoints=1)
    ax1.set_ylabel('Accuracy (%)')
    ax1.grid('on')
    # plt.show()

    figure.savefig(os.path.join(R"C:\Users\bms\Files\current\research\stylemotry\stylometry papers\best results\RNN\scale",
                                "scale"), dpi=900)
    # figure.clear()

if __name__ == "__main__":
    # plot_relax()
    # plot_complex()
    # plot_scale()
    plot_classifiers()