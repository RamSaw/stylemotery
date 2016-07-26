import ast
import copy
import os
import re
from collections import defaultdict, Counter

import codegen as cg


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
        return lambda node: True if len(list(children(node))) else False
    elif mode == "leaves":
        return lambda node: False if len(list(children(node))) else True
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


def ast_parse_file(filename):
    try:
        with open(filename, 'r',encoding="utf-8") as file:
            return ast.parse(file.read())
    except Exception as e:
        print("ERROR: ERROR",e)
        print("ERROR: filename",filename)


def ast_print(tree):
    bfs(tree, callback=printcb, mode="all")



if __name__ == "__main__":
    filename = os.path.join(os.getcwd(), 'dump_program.py')
    traverse = bfs(ast_parse_file(filename), callback=printcb, mode="all")
    astnodes = AstNodes()

