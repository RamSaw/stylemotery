import ast
import copy
import os
import re
from collections import defaultdict, Counter
from operator import itemgetter
import copy
import numpy as np
import codegen as cg

from utils import ast_parse_file, get_basefolder, parse_src_files


class AstNodes:
    NONE = "NONE"

    def __init__(self):
        self.nodetypes = []
        for x in dir(ast):
            try:
                if isinstance(ast.__getattribute__(x)(), ast.AST):
                    self.nodetypes.append(x)
            except TypeError:
                pass
        self.nodetypes.append(self.NONE)
        self.nodetypes_indices = {v.upper(): i for i, v in enumerate(self.nodetypes)}

    def index(self, node):
        if node is None:
            return self.nodetypes_indices[self.NONE]
        if isinstance(node, ast.AST):
            return self.nodetypes_indices[type(node).__name__.upper()]
        elif isinstance(node, tuple):
            return sum([(self.size() ** i) * s for i, s in enumerate(reversed(node))])

    def get(self, index):
        return self.nodetypes[index]

    def size(self):
        return len(self.nodetypes)


class PythonKeywords:
    def __init__(self):
        self.py_keywords = ["and", "del", "from", "not", "while",
                            "as", "elif", "global", "or", "with",
                            "assert", "else", "if", "pass", "yield",
                            "break", "except", "import", "print",
                            "class", "exec", "in", "raise",
                            "continue", "finally", "is", "return",
                            "def", "for", "lambda", "try",
                            "None"]
        self.py_keywords_indices = {v: i for i, v in enumerate(self.py_keywords)}

    def index(self, keyword):
        return self.py_keywords_indices[keyword]

    def try_index(self, keyword):
        try:
            return self.py_keywords_indices[keyword]
        except Exception:
            return -1

    def get(self, index):
        return self.py_keywords[index]

    def size(self):
        return len(self.py_keywords)


def printcb(node, depth, out=None):
    '''print indented node names'''
    nodename = node.__class__.__name__
    nodename += "("
    for name, value in ast.iter_fields(node):
        field = getattr(node, name)
        if type(field) in [int, str, float]:
            nodename += str(getattr(node, name))
    nodename += ")"
    print(' ' * depth * 2 + nodename)


def set_condition(mode):
    if mode == "parents":
        return lambda node: True if len(list(children(node))) > 0 else False
    elif mode == "leaves":
        return lambda node: True if len(list(children(node))) == 0 else False
    elif mode == "all":
        return lambda node: True


def bfs(node, callback, mode="all", out=None):
    condition = set_condition(mode)
    depth = 0

    def bfs_rec(node, depth, out):
        if condition(node):
            callback(node, depth, out)
        for child in children(node):
            bfs_rec(child, depth + 1, out)

    bfs_rec(node, depth, out)
    return out


def dfs(node, callback, mode="all", out=None):
    condition = set_condition(mode)
    depth = 0

    def dfs_rec(node, depth, out):
        for child in children(node):
            dfs_rec(child, depth + 1, out)
        if condition(node):
            callback(node, depth, out)

    dfs_rec(node, depth, out)
    return out


def children(node):
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield item


def ast_print(tree):
    bfs(tree, callback=printcb, mode="all")


def split_trees(X, y, problems):
    subX = []
    subY = []
    subProblem = []
    for i, tree in enumerate(X):
        ast_children = children(tree)
        for child in ast_children:
            subX.append(child)
            subY.append(y[i])
            subProblem.append(problems[i])
    return np.array(subX), np.array(subY), np.array(subProblem)


def split_trees2(X, y, problems):
    subX = []
    subY = []
    subProblem = []

    for i, tree in enumerate(X):
        functions = []
        classes = []
        imports = []
        global_code = []
        ast_children = children(tree)
        for child in ast_children:
            child_name = type(child).__name__
            if child_name == "FunctionDef":
                functions.append(child)
            elif child_name == "ClassDef":
                classes.append(child)
            elif child_name == "ImportFrom" or child_name == "Import":
                imports.append(child)
            else:
                global_code.append(child)
        # add functions
        subX.extend(functions)
        subY.extend([y[i]] * len(functions))
        # add classes
        subX.extend(classes)
        subY.extend([y[i]] *len(classes))
        # add imports
        tree.body = []
        import_module = copy.deepcopy(tree)
        import_module.body = imports
        subX.append(import_module)
        subY.append(y[i])
        # add the rest of the code ( global instructions)
        tree.body = []
        global_module = copy.deepcopy(tree)
        global_module.body = global_code
        subX.append(global_module)
        subY.append(y[i])

        subProblem.append(problems[i])
    return np.array(subX), np.array(subY), np.array(subProblem)


if __name__ == "__main__":
    # filename = os.path.join(os.getcwd(), 'dump_program.py')
    # traverse = bfs(ast_parse_file(filename), callback=printcb, mode="all")
    # astnodes = AstNodes()
    basefolder = get_basefolder()
    X, y, problems = parse_src_files(basefolder)
    subX, subY, subProblems = split_tees(X, y, problems)

    print("\t\t%s Unique problems, %s Unique users :" % (len(set(problems)), len(set(y))))
    print("\t\t%s All problems, %s All users :" % (len(problems), len(y)))
    print("\t\t%s Sub problems, %s sub users :" % (len(subProblems), len(subY)))
    ratio = sorted([(i, Counter(subY)[i],
                     "%{0}".format(round((Counter(subY)[i] / float(len(subY)) * 100.0), 2))) for i in Counter(subY)],
                   key=itemgetter(1), reverse=True)
    print("\t\t all users ratio ", ratio)
    first_layer = []
    for x in subX:
        first_layer.append(type(x).__name__)

    ratio = sorted([(i, Counter(first_layer)[i],
                     "%{0}".format(round((Counter(first_layer)[i] / float(len(first_layer)) * 100.0), 2))) for i in
                    Counter(first_layer)], key=itemgetter(1), reverse=True)
    print("\t\t all users ratio ", ratio)
