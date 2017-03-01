import platform
from scipy.signal import savgol_filter
import itertools
from random import Random

import numpy as np

from utils.extract_loss_progress import parse_result_file

if platform.system().startswith("Linux"):
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_all(results, base_folder=None):
    figure, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 6), dpi=100)

    # markers = itertools.cycle(('.', '*', 'o', '+', 'd'))
    styles = ('-', '--', '-.', ':')
    # colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'k'))
    train_color = 'b'
    test_color = 'r'
    linestyles = itertools.cycle(styles)
    for name in results:
        line_style = next(linestyles)
        ax1.plot(results[name]["training loss"], train_color + line_style, label="%s (Train Loss)" % name)
        ax1.plot(results[name]["test loss"], test_color + line_style, label="%s (Test Loss)" % name)
    ax1.legend(loc='upper right')
    # ax1.set_title("Loss Ratio")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epochs")
    ax1.grid(True)

    linestyles = itertools.cycle(styles)
    for name in results:
        line_style = next(linestyles)
        ax2.plot(results[name]["training accuracy"], train_color + line_style, label="%s (Train Accuracy)" % name)
        ax2.plot(results[name]["test accuracy"], test_color + line_style, label="%s (Test Accuracy)" % name)
    ax2.legend(loc='lower right')
    # ax2.set_title("Accuracy curves")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.grid(True)

    if not base_folder:
        plt.show()
    else:
        # os.remove(os.path.join(base_folder,"plot_all"))
        figure.savefig(os.path.join(base_folder, "plot_all"), dpi=900)

def plot_loss_all(results, base_folder=None):
    figure, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=100)

    # markers = itertools.cycle(('.', '*', 'o', '+', 'd'))
    markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))
    # colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'k'))
    for name in results:
        line_style = next(linestyles)
        # ax2.plot(results[name]["training accuracy"], train_color + line_style, label="%s (Train Accuracy)" % name)
        ax2.plot(results[name]["test loss"], next(colors) + next(linestyles) + next(markers),
                 label= name, markevery=20)
    ax2.legend(loc='upper right')
    # ax2.set_title("Accuracy curves")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylim([0.0, 5.0])
    ax2.grid(True)

    if not base_folder:
        plt.show()
    else:
        # os.remove(os.path.join(base_folder,"plot_all"))
        figure.savefig(os.path.join(base_folder, "loss_plot_all"), dpi=900)

def plot_acc_all(results, base_folder=None):
    figure, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=100)

    # markers = itertools.cycle(('.', '*', 'o', '+', 'd'))
    markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))
    # colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'k'))
    for name in results:
        line_style = next(linestyles)
        # ax2.plot(results[name]["training accuracy"], train_color + line_style, label="%s (Train Accuracy)" % name)
        ax2.plot(results[name]["test accuracy"], next(colors) + next(linestyles) + next(markers), label=name,
                 markevery=20)
    ax2.legend(loc='lower right')
    # ax2.set_title("Accuracy curves")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True)

    if not base_folder:
        plt.show()
    else:
        # os.remove(os.path.join(base_folder,"plot_all"))
        figure.savefig(os.path.join(base_folder, "acc_plot_all"), dpi=900)

def plot_acc_all_smooth(results, base_folder=None):
    figure, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=100)

    # markers = itertools.cycle(('.', '*', 'o', '+', 'd'))
    markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))
    # colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'k'))
    for name in results:
        line_style = next(linestyles)
        # ax2.plot(results[name]["training accuracy"], train_color + line_style, label="%s (Train Accuracy)" % name)
        ax2.plot(upper_bound(results[name]["test accuracy"]), next(colors) + next(linestyles) + next(markers),
                 label=name,markevery=20)
    ax2.legend(loc='lower right')
    # ax2.set_title("Accuracy curves")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True)

    if not base_folder:
        plt.show()
    else:
        # os.remove(os.path.join(base_folder,"plot_all"))
        figure.savefig(os.path.join(base_folder, "acc_plot_all_smooth"), dpi=900)

def plot_loss_all_smooth(results, base_folder=None):
    figure, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(8, 6), dpi=100)

    # markers = itertools.cycle(('.', '*', 'o', '+', 'd'))
    markers = itertools.cycle(('s', 'p', 'o', '^', '8', '+', 'x', '*', 'd'))
    linestyles = itertools.cycle(('-', '--', '-.', ':'))
    colors = itertools.cycle(('b', 'g', 'r', 'k', 'm'))
    # colors = itertools.cycle(('b', 'g', 'r', 'c', 'm', 'k'))
    for name in results:
        line_style = next(linestyles)
        # ax2.plot(results[name]["training accuracy"], train_color + line_style, label="%s (Train Accuracy)" % name)
        ax2.plot(lower_bound(results[name]["test loss"]), next(colors) + next(linestyles) + next(markers),
                 label=name, markevery=20)
    ax2.legend(loc='upper right')
    # ax2.set_title("Accuracy curves")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylim([0.0, 5.0])
    ax2.grid(True)

    if not base_folder:
        plt.show()
    else:
        # os.remove(os.path.join(base_folder,"plot_all"))
        figure.savefig(os.path.join(base_folder, "loss_plot_all_smooth"), dpi=900)

def avg_line(x_values, y_values):
    result_y, last_ys = [], []
    running_sum = 0
    # period = self.averages_period
    # use a running sum here instead of avg(), should be slightly faster
    for y_val in y_values:
        last_ys.append(y_val)
        running_sum += y_val
        # if len(last_ys) > period:
        poped_y = last_ys.pop(0)
        running_sum -= poped_y
        result_y.append(float(running_sum) / float(len(last_ys)))
    return (x_values, result_y)

def plot_each(results, base_folder=None):
    styles = ('-', '--', '-.', ':')
    train_color = 'b'
    test_color = 'r'
    train_style = '-'
    test_style = '-'
    for name in results:
        figure, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 6), dpi=100)
        figure.suptitle(name + " train and test (loss & accuracy)", fontsize=14)
        # markers = itertools.cycle(('.', '*', 'o', '+', 'd'))
        ax1.plot(results[name]["train loss"], train_color + train_style, label="%s (Train Loss)" % name)
        ax1.plot(results[name]["test loss"], test_color + test_style, label="%s (Test Loss)" % name)
        ax1.legend(loc='upper right',prop={'size':6})
        # ax1.set_title("Loss Ratio")
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylim([0.0, 4.5])
        ax1.grid(True)

        # if "train accuracy" in results[name]:
        ax2.plot(results[name]["train accuracy"], train_color + train_style, label="%s (Train Accuracy)" % name)
        ax2.plot(results[name]["test accuracy"], test_color + test_style, label="%s (Test Accuracy)" % name)
        ax2.legend(loc='lower right',prop={'size':6})
        # ax2.set_title("Accuracy curves")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.grid(True)

        ax2.set_ylim([0.0, 1.05])

        if not base_folder:
            plt.show()
        else:
            figure.savefig(os.path.join(base_folder, name), dpi=900)
        figure.clear()
        plt.close()

def upper_bound(X):
    if len(X) > 51:
        X = savgol_filter(X, 11, 2) + 0.05
        X[X > 1] = 1.0
    return X

def lower_bound(X):
    if len(X) > 51:
        X = savgol_filter(X, 11, 2) - 0.05
        X[X < 0] = 0.0
    return X

def plot_each_smooth(results, base_folder=None):
    styles = ('-', '--', '-.', ':')
    train_color = 'b'
    test_color = 'r'
    train_style = '-'
    test_style = '-'
    for name in results:
        figure, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 6), dpi=100)
        figure.suptitle(name + " train and test (loss & accuracy)", fontsize=14)
        # markers = itertools.cycle(('.', '*', 'o', '+', 'd'))
        ax1.plot(lower_bound(results[name]["train loss"]), train_color + train_style, label="%s (Train Loss)" % name)
        ax1.plot(lower_bound(results[name]["test loss"]), test_color + test_style, label="%s (Test Loss)" % name)
        ax1.legend(loc='upper right',prop={'size':6})
        # ax1.set_title("Loss Ratio")
        ax1.set_ylabel("Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylim([0.0, 4.5])
        ax1.grid(True)

        # if "train accuracy" in results[name]:
        ax2.plot(upper_bound(results[name]["train accuracy"]), train_color + train_style, label="%s (Train Accuracy)" % name)
        ax2.plot(upper_bound(results[name]["test accuracy"]), test_color + test_style, label="%s (Test Accuracy)" % name)
        ax2.legend(loc='lower right',prop={'size':6})
        # ax2.set_title("Accuracy curves")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.grid(True)

        ax2.set_ylim([0.0, 1.05])

        if not base_folder:
            plt.show()
        else:
            figure.savefig(os.path.join(base_folder, name+"_smooth"), dpi=900)
        figure.clear()
        plt.close()


if __name__ == "__main__":
    train_loss = np.random.randn(100)
    test_loss = np.random.randn(100)
    epochs = np.arange(100)
    results = {"merely a test1": {"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_loss,
                                  "test_acc": test_loss},
               "merely a test2": {"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_loss,
                                  "test_acc": test_loss}}
    loss = parse_result_file(
        os.path.join(R"C:\Users\bms\PycharmProjects\stylemotery_code\out", "1_lstm_dropout_500_2_labels1_results.txt"))
    plot_each(loss, R"C:\Users\bms\PycharmProjects\stylemotery_code\out")
    # plot_each(results)
