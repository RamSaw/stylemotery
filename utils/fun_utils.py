import ast
import os
import platform
import sys
import numpy as np

import copy
from ast_tree.ast_parser import children, ast_print, bfs


def get_basefolder():
    if platform.system().startswith("Linux"):
        return R"dataset700"
    elif platform.system().startswith("Win"):
        return R"dataset/dataset700"


def ast_parse_file(filename):
    try:
        with open(filename, 'r', encoding="utf-8") as file:
            tree = ast.parse(file.read())
            return tree
    except Exception as e:
        print("ERROR: ", e, " filename", filename)




def get_src_files(basefolder):
    files = os.listdir(basefolder)
    files_noext = ['.'.join(s.split('.')[:-1]) for s in files]

    problems = [p.split('.')[0] for p in files_noext]
    users = [' '.join(p.split('.')[1:]) for p in files_noext]

    return np.array([os.path.join(basefolder, file) for file in files]), np.array(users), np.array(problems)


def parse_src_files(basefolder, seperate_trees=False):
    X_names, y, problems = get_src_files(basefolder)
    return np.array([ast_parse_file(name) for name in X_names]), np.array(y), problems
    # return np.array(make_binary_tree(unified_ast_trees([ast_parse_file(name) for name in X_names]))), np.array(y), problems


def generate_tree(node, children):
    root = ast.Module()
    root._fields = ("body",)

    ast_nodes = [ast.Add, ast.Assign,ast.And,ast.arguments,ast.AugAssign]

    child = node()
    child._fields = ("body",)
    child.body = [ast_nodes[i]() for i in range(children)]

    root.body = [child for i in range(children)]
    return root


def generate_trees(basefolder, labels=2, children=10, examples_per_label=10):
    X = []
    y = []
    ast_nodes = [ast.Add, ast.Assign]
    for label in range(labels):
        tree = generate_tree(ast_nodes[label], children)
        X.extend([tree for i in range(examples_per_label)])
        y.extend([label for i in range(examples_per_label)])
    return np.array(X), np.array(y), np.array([])


def make_backward_graph(basefolder, filename, var):
    import chainer.computational_graph as c
    import os
    g = c.build_computational_graph(var)
    with open(os.path.join(basefolder, filename), '+w') as o:
        o.write(g.dump())
    # dot -Tps filename.dot -o outfile.ps
    from subprocess import call
    call(["dot", "-Tpdf", os.path.join(basefolder, filename), "-o", os.path.join(basefolder, filename + ".pdf")])


def print_model(model,depth=0,output=sys.stdout):
    if len(list(model.children())) == 0:
        output.write("\t"*depth+"{0} {1} {2} \n".format(model.name,type(model).__name__,model._params))
        for p in model._params:
            output.write("\t"*depth+"{0} {1} {2} {3}\n".format("\t"*depth, p, "=>", getattr(model, p).data.shape))
    else:
        if model.name is not None:
            output.write("\t"*depth+"{0} \n".format(model.name))
        for child in model.children():
            print_model(child,depth=depth+1,output=output)


class Node:
    def __init__(self,child):
        self.children = child
        self._fields = ('children',)


def unified_ast_trees(trees):
    def convert_tree(src_tree):
        childern = []
        for child in children(src_tree):
            childern.append(convert_tree(child))
        src_tree.children = childern
        return src_tree
    utrees = []
    for tree in trees:
        utrees.append(convert_tree(tree))
    return utrees
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
def make_binary_tree(trees,max_branches=10):
    def make_binary_tree(src_tree,dst_tree,max_branches = 2):
        childs = list(children(src_tree))
        if len(childs) > 0:
            dst_tree.children.extend(childs[:max_branches])
            if len(childs)-max_branches > 0:
                if len(childs) - max_branches == 1:
                    dst_tree.children.append(childs[-1])
                else:
                    dst_node = copy.copy(dst_tree)
                    dst_node.children = []
                    src_node = copy.copy(dst_tree)
                    src_node.children = childs[max_branches:]
                    dst_tree.children.append(make_binary_tree(src_node,dst_node))
            for idx,child in enumerate(children(dst_tree)):
                dst_child = copy.copy(child)
                dst_child.children = []
                dst_tree.children[idx] = make_binary_tree(child,dst_child,max_branches)
        return dst_tree
    import sys
    sys.setrecursionlimit(100000)
    dst_trees = []
    for idx,tree in enumerate(trees):
        # print(idx, ", max branches=",max_branch(tree)," max depth =", max_depth(tree))
        binary_tree = copy.copy(tree)
        binary_tree.children = []
        dst_trees.append(make_binary_tree(tree,binary_tree,max_branches))
    return np.array(dst_trees)

if __name__ == "__main__":
    trees,labels,problems = generate_trees("",2,5,20)
    trees = unified_ast_trees(trees[:1])
    ast_print(trees[0])
    ast_print(make_binary_tree(trees,max_branches=2)[0])





