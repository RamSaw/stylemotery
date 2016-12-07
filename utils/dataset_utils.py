import ast
import os
import platform
import sys
import numpy as np

import copy

from tqdm import tqdm

from ast_tree.traverse import children, bfs
from ast_tree.tree_nodes import DotNodes, AstNodes
from ast_tree.tree_parser import parse_dot, ast_parse_file, fast_parse_dot, parse_tree


def get_ast_src_files(basefolder):
    files = os.listdir(basefolder)
    files_noext = ['.'.join(s.split('.')[:-1]) for s in files]

    problems = [p.split('.')[0] for p in files_noext]
    users = [' '.join(p.split('.')[1:]) for p in files_noext]

    return np.array([os.path.join(basefolder, file) for file in files]), np.array(users), np.array(problems)

def get_dot_files(basefolder):
    trees = []
    users = []
    problems = []
    for folder in [f for f in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder,f))]:
        for number in os.listdir(os.path.join(basefolder, folder)):
            file = [filename for filename in os.listdir(os.path.join(basefolder, folder, number)) if
                    filename.endswith(".tree")][0]
            trees.append(parse_tree(os.path.join(basefolder, folder, number, file)))
            users.append(folder)
    return np.array(trees),np.array(users),np.array(problems)

def get_dot_files2(basefolder,seperate_trees=False):
    trees = []
    users = []
    problems = []
    for file in tqdm(os.listdir(basefolder)):
        program_trees = parse_tree(os.path.join(basefolder, file),seperate_trees)
        trees.extend(program_trees)
        users.extend([file.split('.')[0]]*len(program_trees))
    return np.array(trees),np.array(users),np.array(problems)

def parse_src_files(basefolder, seperate_trees=False):
    if basefolder.endswith("python"):
        X_names, y, problems = get_ast_src_files(basefolder)
        return np.array([ast_parse_file(name) for name in tqdm(X_names)]), np.array(y), problems,AstNodes()
    else:
        X, y, problems = get_dot_files2(basefolder,seperate_trees)
        return np.array(X), np.array(y), problems, DotNodes()

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





