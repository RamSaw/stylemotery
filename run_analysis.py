import matplotlib.pyplot as plt
import numpy as np

from ast_tree.traverse import bfs, children, split_trees2
from utils.dataset_utils import get_basefolder, parse_src_files, unified_ast_trees, make_binary_tree
import os

def max_depth(ast_tree):
    def max_depth_lambda(x, d, o):
        if len(o) == 0:
            o.append(d)
        elif d > o[0]:
            o[0] = d

    out = bfs(ast_tree, callback=max_depth_lambda, mode="leaves", out=[])
    return out[0]


def max_branch(ast_tree):
    def max_branch_lambda(x, d, o):
        count = len(list(children(x)))
        if len(o) == 0:
            o.append(count)
        elif count > o[0]:
            o[0] = count

    out = bfs(ast_tree, callback=max_branch_lambda, mode="all", out=[])
    return out[0]


def plot_dist(name, dist, max_len=1000):
    dist_np = np.array(dist)
    dist = dist_np[dist_np < max_len]
    plt.hist(dist)
    plt.title(name + " Histogram")
    plt.xlabel(name)
    plt.ylabel("Frequency")

    plt.show()


def plot_dists(name, depths, branches, max_len=1000,base_folder=None):
    figure, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 6), dpi=100)
    figure.suptitle(name + " Depth and Branch Distrubtions", fontsize=14)
    dist_np = np.array(depths)
    if max_len != None:
        dist = dist_np[dist_np < max_len]
    else:
        dist = dist_np
    ax1.hist(dist)
    ax1.set_title("Depth Histogram")
    ax1.set_xlabel("Depth")
    ax1.set_ylabel("Frequency")

    dist_np = np.array(branches)
    if max_len != None:
        dist = dist_np[dist_np < max_len]
    else:
        dist = dist_np
    ax2.hist(dist)
    ax2.set_title("Branch Histogram")
    ax2.set_xlabel("Branch")
    ax2.set_ylabel("Frequency")

    if not base_folder:
        plt.show()
    else:
        figure.savefig(os.path.join(base_folder, name), dpi=900)
    figure.clear()
    plt.close()

    plt.show()


if __name__ == "__main__":
    basefolder = get_basefolder()
    X, y, tags = parse_src_files(basefolder)

    # X = make_binary_tree(unified_ast_trees(X), 9)
    depths = [max_depth(x) for x in X]
    branches = [max_branch(x) for x in X]
    plot_dists("Single Tree", depths, branches, max_len=100)

    # X,y,tags = split_trees2(X,y,tags)
    # depths = np.array([max_depth(x) for x in X])
    # branches = [max_branch(x) for x in X]
    # plot_dists("Multiple Trees", depths, branches)
