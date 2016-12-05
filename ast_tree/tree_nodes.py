import ast
import copy
import os
import re
from collections import defaultdict, Counter
from operator import itemgetter
import copy
import numpy as np
import codegen as cg

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


def print_ast_node(node, depth, out=None):
    '''print indented node names'''
    nodename = node.__class__.__name__
    nodename += "("
    # if hasattr(node,"_fields"):
    for name, value in ast.iter_fields(node):
        field = getattr(node, name)
        if type(field) in [int, str, float]:
            nodename += str(getattr(node, name))
    # elif hasattr(node,"_content"):
    #     for name, value in node.content():
    #         field = getattr(node, name)
    #         if type(field) in [int, str, float]:
    #             nodename += str(getattr(node, name))
    nodename += ")"
    print(' ' * depth * 2 + nodename)


class Node:
    def __init__(self, type, code, child):
        self.type = type
        self.code = (code if len(code) > 0 else " ")
        self.children = child
        self._fields = ('children',)

class DotNodes:
    NONE = "NONE"

    def __init__(self):
        self.nodetypes = [s.upper() for s in ['AndExpression', 'ExclusiveOrExpression', 'IncDec', 'ParameterType', 'Parameter',
                    'IdentifierDecl',
                    'ContinueStatement', 'CompoundStatement', 'PrimaryExpression', 'Expression', 'AdditiveExpression',
                    'ExpressionStatement', 'DoStatement', 'SwitchStatement', 'CastExpression', 'InclusiveOrExpression',
                    'Label', 'IncDecOp', 'ClassDefStatement', 'Sizeof', 'MemberAccess', 'EqualityExpression',
                    'UnaryOperator', 'WhileStatement', 'ConditionalExpression', 'ParameterList', 'CastTarget',
                    'InitializerList', 'IfStatement', 'ElseStatement', 'RelationalExpression', 'BlockStarter',
                    'ReturnStatement', 'GotoStatement', 'UnaryExpression', 'ArrayIndexing', 'ArgumentList',
                    'ReturnType',
                    'Statement', 'AssignmentExpr', 'OrExpression', 'FunctionDef', 'CallExpression',
                    'IdentifierDeclStatement', 'PtrMemberAccess', 'UnaryOp', 'MultiplicativeExpression', 'Argument',
                    'BitAndExpression', 'ShiftExpression', 'Identifier', 'Condition', 'ForStatement', 'Callee',
                    'IdentifierDeclType', 'SizeofExpr', 'BreakStatement', 'ForInit', 'SizeofOperand',"Program"]]
        self.nodetypes.append(self.NONE)
        self.nodetypes_indices = {v.upper(): i for i, v in enumerate(self.nodetypes)}

    def index(self, node):
        if node is None:
            return self.nodetypes_indices[self.NONE]
        if isinstance(node, Node):
            # if not node.type:
            #     return self.nodetypes_indices[self.NONE]
            return self.nodetypes_indices[node.type.upper()]
        elif isinstance(node, tuple):
            return sum([(self.size() ** i) * s for i, s in enumerate(reversed(node))])

    def get(self, index):
        return self.nodetypes[index]

    def size(self):
        return len(self.nodetypes)

def print_dot_node(node, depth, out=None):
    print(' ' * depth * 2 + node.type + "()")  # node.code

