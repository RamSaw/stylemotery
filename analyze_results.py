import os

import sys
from chainer import serializers
import numpy as np
from models import RecursiveLSTM
from utils.extract_loss_progress import parse_result_file
from utils.fun_utils import print_model
from utils.graph import plot_all, plot_each
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, KernelPCA


def standarize(values):
    return {k.replace("_", " ").replace("training", "train"): v for k, v in values.items()}

def plot_results(base_folder):
    # print(base_folder)
    results = {}
    for filename in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, filename)):
            plot_results(os.path.join(base_folder, filename))
        elif filename.endswith("_results.txt"):
            loss = parse_result_file(os.path.join(base_folder, filename))
            if len(next(iter(loss.values()))) > 0:
                results.update(loss)
    # print(results.keys())
    new_results = {name: standarize(values) for name, values in results.items()}
    for name, values in new_results.items():
        print(name, " ==> ", values.keys())

    plot_each(new_results, base_folder)


if __name__ == "__main__":
    base_folder = R"C:\Users\bms\Files\study\DC\Experiments\results\Sept 2nd Week"
    plot_results(base_folder)
