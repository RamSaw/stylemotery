
from ast_tree.traverse import bfs, children
import numpy as np

def max_depth(ast_tree):
    def max_depth_lambda(x, d, o):
        if len(o) == 0:
            o.append(d)
        elif d > o[0]:
            o[0] = d
    out = bfs(ast_tree, callback=max_depth_lambda, mode="all", out=[])
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


def avg_depth(ast_tree):
    def avg_depth_lambda(x, d, o):
        o.append(d)
    out = bfs(ast_tree, callback=avg_depth_lambda, mode="all", out=[])
    return int(np.mean(out))


def avg_branch(ast_tree):
    def avg_branch_lambda(x, d, o):
        count = len(list(children(x)))
        if count > 0:
            o.append(count)
    out = bfs(ast_tree, callback=avg_branch_lambda, mode="all", out=[])
    return int(np.mean(out))

