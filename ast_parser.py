import ast
import copy
import os
import re
from collections import defaultdict, Counter
import numpy as np
import codegen as cg

from utils import ast_parse_file, get_basefolder, parse_src_files


class AstNodes:
    def __init__(self):
        self.nodetypes = []
        for x in dir(ast):
            try:
                if isinstance(ast.__getattribute__(x)(), ast.AST):
                    self.nodetypes.append(x)
            except TypeError:
                pass
        self.nodetypes_indices = {v.upper(): i for i, v in enumerate(self.nodetypes)}

    def index(self, node):
        if isinstance(node,ast.AST):
            return self.nodetypes_indices[type(node).__name__.upper()]
        elif isinstance(node,tuple):
            return sum([(self.size() ** i) * s for i,s in enumerate(reversed(node))])

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

def breakup_tees(X, y, problems):
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

if __name__ == "__main__":
    # filename = os.path.join(os.getcwd(), 'dump_program.py')
    # traverse = bfs(ast_parse_file(filename), callback=printcb, mode="all")
    # astnodes = AstNodes()
    basefolder = get_basefolder()
    X, y, problems = parse_src_files(basefolder)
    subX, subY, subProblems = breakup_tees(X,y,problems)

    print("\t\t%s Unique problems, %s Unique users :" % (len(set(problems)), len(set(y))))
    print("\t\t%s All problems, %s All users :" % (len(problems), len(y)))
    print("\t\t%s Sub problems, %s sub users :" % (len(subProblems), len(subY)))

