

import ast
import os
import pprint
import uuid
from copy import deepcopy
from operator import itemgetter

import jsonpickle
import json
from collections import defaultdict, Counter

from tqdm import tqdm

from ast_tree.tree_nodes import Node, print_dot_node, DotNodes, AstNodes, print_ast_node
from ast_tree.traverse import tree_print, bfs, children
import sys
import numpy as np

from utils.analysis_utils import max_depth, max_branch


def parse_ast_tree(filename):
    nodes = {}
    links = {}
    nodetypes = {}
    for x in dir(ast):
        try:
            if isinstance(ast.__getattribute__(x)(), ast.AST):
                nodetypes[x.lower()] = ast.__getattribute__(x)
        except TypeError:
            pass
    for line in open(filename):
        if line.startswith("<"):
            parts = line[1:].strip("\n").split("=")
            links[parts[0]] = parts[1].split(",")
        elif line.startswith(">"):
            parts = line[1:].strip("\n").split("\t")
            nodes[parts[0]] = nodetypes[parts[1].lower()]()
            nodes[parts[0]].children = []
    root_nodes = []
    for id, value in sorted(links.items()):
        for link in value:
            nodes[id].children.append(nodes[link])
            root_nodes.append(link)

    dot = AstNodes()
    for node in nodes.values():
        try:
            dot.index(node)
        except:
            print(filename)
    root_nodes = set(nodes.keys()) - set(root_nodes)
    root_nodes = [nodes[id] for id in list(sorted(root_nodes))]

    return nodes['0']

def unify_children(node):
    if not hasattr(node,"uuid"):
        node.uuid = str(uuid.uuid4())
    if not hasattr(node,"children"):
        children = []
        fields = []
        for field, value in ast.iter_fields(node):
            fields.append(field)
            if isinstance(value, ast.AST):
                children.append(unify_children(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        children.append(unify_children(item))
        node.children = children
        # for field in fields:
        #     delattr(node,field)
        # if hasattr(node, "_fields"):
        #     delattr(node, "_fields")
    return node

def ast_parse_file(filename):
    try:
        with open(filename, 'r', encoding="utf-8") as file:
            tree = ast.parse(file.read())

            # tree = unify_children(tree)
            #export trees
            # js = jsonpickle.encode(tree)
            # tree = json.loads(js)
            return tree
    except Exception as e:
        print("ERROR: ", e, " filename", filename)


def get_ast_src_files(basefolder):
    files = os.listdir(basefolder)
    files_noext = ['.'.join(s.split('.')[:-1]) for s in files]
    problems = [p.split('.')[0] for p in files_noext]
    users = [' '.join(p.split('.')[1:]) for p in files_noext]

    return np.array([os.path.join(basefolder, file) for file in files]), np.array(users), np.array(problems)

def parse_src_files(basefolder, seperate_trees=False,verbose=0):
    if basefolder.endswith("python"):
        X_names, y, problems = get_ast_src_files(basefolder)
        X ,y,tags = np.array([ast_parse_file(name) for name in tqdm(X_names)]), np.array(y), problems
        return X ,y,tags,AstNodes()
    else:
        X_names, y, problems = get_ast_src_files(basefolder)
        X ,y,tags = np.array([parse_ast_tree(name) for name in tqdm(X_names)]), np.array(y), problems
        return X ,y,tags,AstNodes()


def count_nodes(x,d,o):
    o.append(1)



def convert_src_files(basefolder):
    X_names, y, problems = get_ast_src_files(basefolder)
    X ,y,tags = np.array([ast_parse_file(name) for name in tqdm(X_names)]), np.array(y), problems
    for name,tree in zip(X_names,X):
        name = os.path.basename(name)
        tree = unify_children(tree)
        # print(1)
        with open(os.path.join("..","dataset","python_trees",os.path.splitext(name)[0]+".tree"),"+w") as file:
            cc = []
            coun = len(bfs(tree,callback=count_nodes,out=cc))
            node_ids = nodes_ids(tree)
            node_links = nodes_links(node_ids, tree)
            for k,(v,l) in node_ids.items():
                file.write(">{0}\t{1}\n".format(l,type(v).__name__))
            for k,v in node_links.items():
                file.write("<{0}={1}\n".format(str(k),",".join([str(i) for i in v])))

c = 0
def nodes_ids(tree):
    def nodes_ids_lambda(x, d, o):
        global c
        o[x.uuid] = (x, c)
        c += 1
        return o
    global c
    c = 0
    ids = {}
    bfs(tree, callback=nodes_ids_lambda, mode="all", out=ids)
    # for x,id in node_ids:
    #     links[id].append()
    return ids

def nodes_links(node_ids,tree):
    def nodes_links_lambda(x, d, o):
        for child in children(x):
            o[node_ids[x.uuid][1]].append(node_ids[child.uuid][1])
        return o

    links = defaultdict(list)
    bfs(tree, callback=nodes_links_lambda, mode="all", out=links)
    # for x,id in node_ids:
    #     links[id].append()
    return links


def test_main():
    dataset_folder = os.path.join("..", "dataset", "python_trees")
    trees, tree_labels, lable_problems, tree_nodes = parse_src_files(dataset_folder, False)
    depths = np.array([max_depth(x) for x in trees])
    branches = np.array([max_branch(x) for x in trees])
    print(list(depths))
    print(np.mean(depths))
    print(np.mean(branches))
    print("Class ratio :- %s" % list(sorted([(t, c, c / len(tree_labels)) for t, c in Counter(tree_labels).items()], key=itemgetter(0),reverse=False)))

    tree_6 = trees[10]
    dataset_folder = os.path.join("..", "dataset", "python")
    trees, tree_labels, lable_problems, tree_nodes = parse_src_files(dataset_folder, False)
    depths = np.array([max_depth(x) for x in trees])
    branches = np.array([max_branch(x) for x in trees])
    print(list(depths))
    print(np.mean(depths))
    print(np.mean(branches))
    print("Class ratio :- %s" % list(sorted([(t, c, c / len(tree_labels)) for t, c in Counter(tree_labels).items()], key=itemgetter(0),reverse=False)))
    tree_62 = trees[10]
    print()
    tree_print(tree_6,print_ast_node)
    print()
    tree_print(tree_62,print_ast_node)

if __name__ == "__main__":
    # dataset_folder = os.path.join("..", "dataset", "python")
    # convert_src_files(basefolder=dataset_folder)
    test_main()