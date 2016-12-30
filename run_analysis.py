from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import collections
from ast_tree.traverse import bfs, children
from deep_run_exp import read_config
from utils.analysis_utils import max_depth, max_branch,avg_branch,avg_depth
from utils.dataset_utils import parse_src_files, unified_ast_trees, make_binary_tree
import os

# def max_depth(ast_tree):
#     def max_depth_lambda(x, d, o):
#         if len(o) == 0:
#             o.append(d)
#         elif d > o[0]:
#             o[0] = d
#
#     out = bfs(ast_tree, callback=max_depth_lambda, mode="leaves", out=[])
#     return out[0]
#
#
# def max_branch(ast_tree):
#     def max_branch_lambda(x, d, o):
#         count = len(list(children(x)))
#         if len(o) == 0:
#             o.append(count)
#         elif count > o[0]:
#             o[0] = count
#
#     out = bfs(ast_tree, callback=max_branch_lambda, mode="all", out=[])
#     return out[0]
from utils.exp_utlis import pick_subsets


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

def labels_ratio(X,y,min_depth=5,min_branch=5):
    print("Before:")
    print("Class ratio :- %s\n" % list(
        sorted([(t, c, "%.2f" % (c / len(y))) for t, c in collections.Counter(y).items()], key=itemgetter(0),
               reverse=False)))
    depths = np.array([max_depth(x) for x in X])
    branches = np.array([max_branch(x) for x in X])
    bool_vec = (depths >= min_depth) & (branches >= min_branch)
    X = X[bool_vec]
    y = y[bool_vec]
    print("After:")
    print("Class ratio :- %s\n" % list(
        sorted([(t, c, "%.2f" % (c / len(y))) for t, c in collections.Counter(y).items()], key=itemgetter(0),
               reverse=False)))
    depths = np.array([max_depth(x) for x in X])
    branches = np.array([max_branch(x) for x in X])
    plot_dists("Single Tree", depths, branches, max_len=10)


if __name__ == "__main__":
    X, y, tags,features = parse_src_files(os.path.join("dataset","python"),seperate_trees=False,verbose=0)
    # for file in os.listdir(os.path.join("train","cpp")):
    #     print(file)
    #     rand_seed, classes = read_config(os.path.join("train","cpp",file))
    #     X, y = pick_subsets(X_e, y_e, classes=classes)
    #     print("Class ratio :- %s" % list(sorted([(t, c, c / len(y)) for t, c in collections.Counter(y).items()], key=itemgetter(0),reverse=False)))
    #     print()
    # X = make_binary_tree(unified_ast_trees(X), 9)
    depths = [max_depth(x) for x in X]
    branches = [max_branch(x) for x in X]
    plot_dists("CPP SEP MAX Single Tree", depths, branches, max_len=None)#,base_folder=R"C:\Users\bms\Files\current\research\stylemotry\stylemotery_code\dataset\analysis")
    plot_dists("CPP SEP MAX Single Tree (Less 100)", depths, branches, max_len=100)#,base_folder=R"C:\Users\bms\Files\current\research\stylemotry\stylemotery_code\dataset\analysis")

    # labels_ratio(X,y,min_depth=5,min_branch=5)
