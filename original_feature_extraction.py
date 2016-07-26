# Builtins
from __future__ import print_function

import os
from functools import reduce
from collections import Counter
import ast, sys, re, copy, json
from ast import iter_fields, iter_child_nodes, AST

# from https://github.com/andreif/codegen
import codegen as cg


#########################
# Generic n-gram utility functons
#########################
class ngramiterator(object):
    def __init__(self, s, n=1):
        self.n = n
        self.s = s
        self.pos = 0

    def __iter__(self):
        return self

    def next(self):
        if self.pos + self.n > len(self.s):
            raise StopIteration
        else:
            self.pos += 1
            return self.s[self.pos - 1:self.pos + self.n - 1]

    def __next__(self):
        return self.next()


def ngrams(string, n):
    return [x for x in ngramiterator(string, n)]


#########################
# get some basic data about node types and keywords extracted
#########################

ast_node_types = []
for x in dir(ast):
    try:
        if isinstance(ast.__getattribute__(x)(), AST):
            ast_node_types.append(x)
    except TypeError:
        pass
ast_node_types.sort()

python_kws = ["and", "del", "from", "not", "while",
              "as", "elif", "global", "or", "with",
              "assert", "else", "if", "pass", "yield",
              "break", "except", "import", "print",
              "class", "exec", "in", "raise",
              "continue", "finally", "is", "return",
              "def", "for", "lambda", "try",
              "None"]
python_kws.sort()


#########################
# Generator to yield paths through an AST one at a time
# This gets used to produce AST bigrams
#########################

def ast_paths(here, path_to_here=()):
    if isinstance(here, AST):
        # The child is a 'simple' AST node
        path_to_here += (here.__class__.__name__,)
        for k, v in iter_fields(here):
            # Iterate over all key-value pairs in the node...
            for p in ast_paths(v, path_to_here + (k,)):
                # produce all paths rooted at that subtree; adding k to the
                # current path produces a set of alternating key-value sets
                yield p
    elif isinstance(here, list):
        # This is e.g. a function body where the statements (AST nodes) are
        # listed sequentially
        if len(here) == 0: yield path_to_here + (here,)
        # ^^^ special case for length-0 lists, often found as
        # unsupplied/optional parameters
        for v in here:
            for p in ast_paths(v, path_to_here):
                yield p
    else:
        # This is a 'simple' value, so just add it to the list
        yield path_to_here + (here,)


#########################
# Feature extraction code starts here
#########################

# This records the depth for each node in the AST
def fv_dump(mod, level=0, ov=None, pf=None):
    if ov == None: ov = []
    if pf == None: pf = ''

    def is_ast(x):
        if isinstance(x, AST):
            return 1
        if isinstance(x, list):
            if x == []: return 0
            if isinstance(x[0], AST): return 1
        return 0

    for child in iter_child_nodes(mod):
        ov.append((level + 1, child.__class__.__name__, [(k, v) for k, v in iter_fields(child) if not is_ast(v)]))
        # Note that we abuse the way python modifies lists in place to get the correct result here;
        # because of the inplace modification the list is updated as a side-effect, so we don't need
        # to capture and manipulate a return value, even if that's arguably a safer way to do it.
        fv_dump(child, level + 1, ov)
    return ov


# Strip out all string content by replacing any strings with empty ones; this
# is needed to avoid accidentally counting string contents as keywords
def nuke_all_strings(mod):
    if isinstance(mod, ast.Str):
        mod.s = ''
    for child in iter_child_nodes(mod):
        nuke_all_strings(child)


def count_keywords(mod):
    if isinstance(mod, str):
        mod = ast.parse(mod)
    mod2 = copy.deepcopy(mod)
    nuke_all_strings(mod2)
    codestr = cg.to_source(mod2)
    ws = re.compile(("\_*[a-zA-Z]+[a-zA-Z0-9\_]*"))
    tokens = ws.findall(codestr)
    return [tokens.count(x) for x in python_kws]


# extract sequence of keywords from source file string
def keyword_tokenize(mod):
    if isinstance(mod, str):
        mod = ast.parse(mod)
    mod2 = copy.deepcopy(mod)
    nuke_all_strings(mod2)
    codestr = cg.to_source(mod2)
    ws = re.compile(("\_*[a-zA-Z]+[a-zA-Z0-9\_]*"))
    return [x for x in ws.findall(codestr) if x in python_kws]


def AST_tf(mod):
    if isinstance(mod, str):
        mod = ast.parse(mod)
    ### Takes an ast and builds the tf vector for each node type
    fv = [0] * (len(ast_node_types))
    dump = fv_dump(mod, ov=[])
    for i in dump:
        fv[ast_node_types.index(i[1])] += 1
    return fv


def AST_avgdepth(mod):
    if isinstance(mod, str):
        mod = ast.parse(mod)
    ### takes an ast and returns the average depth for each node type
    fv = [0] * (len(ast_node_types))
    fv_ct = AST_tf(mod)
    dump = fv_dump(mod, ov=[])
    for i in dump:
        fv[ast_node_types.index(i[1])] += i[0]
    for index, (i, j) in enumerate(zip(fv, fv_ct)):
        if i == 0: continue
        fv[index] = float(i) / j
    return fv


def AST_maxdepth(mod):
    if isinstance(mod, str):
        mod = ast.parse(mod)
    dump = fv_dump(mod)
    if dump == []: return 0
    return max([x[0] for x in dump])


def ngrammify_keywords(mod, n, keyxform=lambda x: x):
    rv = Counter(map(tuple, ngrams(keyword_tokenize(mod), n)))
    return Counter(dict((keyxform(x), y) for x, y in rv.items()))


def ngrammify_ast(tree, n, keyxform=lambda x: x):
    paths = ast_paths(tree)
    rv = Counter(
        reduce(lambda x, y: x + y, map(lambda x: ngrams(x, n), [tuple(map(str, p)) for p in paths])))
    return Counter(dict((keyxform(x), y) for x, y in rv.items()))


def counterify_avg_depth(tree):
    return Counter(dict(zip(['avd_' + x for x in ast_node_types], AST_avgdepth(tree))))


def counterify_max_depth(tree):
    return Counter(dict([('maxdepth', AST_maxdepth(tree))]))


def normalize_counter(ctr):
    ctr = copy.deepcopy(ctr)
    norm = float(sum(ctr.values()))
    if norm == 0: return ctr
    for k, v in ctr.items():
        ctr[k] = v / norm
    return ctr


def make_mega_fv(tree):
    # Takes an AST as input, returns the (term-frequency) normalized features
    if AST_maxdepth(tree) == 0:
        return counterify_max_depth(tree)
    return \
        normalize_counter(ngrammify_keywords(tree, 1, lambda x: "tf_kw_" + ":".join(map(str, x)))) + \
        normalize_counter(ngrammify_keywords(tree, 2, lambda x: "tf_kw_2gram_" + ":".join(map(str, x)))) + \
        normalize_counter(ngrammify_ast(tree, 1, lambda x: "tf_ast_" + ":".join(map(str, x)))) + \
        normalize_counter(ngrammify_ast(tree, 2, lambda x: "tf_ast_2gram_" + ":".join(map(str, x)))) + \
        counterify_avg_depth(tree) + \
        counterify_max_depth(tree)


def make_mega_fv_unnormed(tree):
    # Return the 'raw' features from a single AST
    if AST_maxdepth(tree) == 0:
        return counterify_max_depth(tree)
    return \
        ngrammify_keywords(tree, 1, lambda x: "tf_kw_" + ":".join(map(str, x))) + \
        ngrammify_keywords(tree, 2, lambda x: "tf_kw_2gram_" + ":".join(map(str, x))) + \
        ngrammify_ast(tree, 1, lambda x: "tf_ast_" + ":".join(map(str, x))) + \
        ngrammify_ast(tree, 2, lambda x: "tf_ast_2gram_" + ":".join(map(str, x))) + \
        counterify_avg_depth(tree) + \
        counterify_max_depth(tree)


if __name__ == '__main__':
    try:
        thefile = os.path.join(os.getcwd(), 'ast_example', 'dump_program.py') #sys.argv[1]
    except IndexError:
        print(",".join(['ct_' + x for x in python_kws] + ['tf_' + x for x in ast_node_types] + ['avd_' + x for x in
                                                                                                ast_node_types] + [
                           'maxdepth']))
        sys.exit(0)
    codestr = open(thefile, 'rb').read()
    try:
        tree = ast.parse(codestr)
    except:
        print("#", thefile, "-- couldn't parse (python 3?)")
        sys.exit(0)
    print(thefile, ":", json.dumps(make_mega_fv(tree)))
