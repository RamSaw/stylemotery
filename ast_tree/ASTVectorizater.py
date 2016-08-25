import ast
import copy
import re
from collections import Counter
from collections import defaultdict

import codegen as cg
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from ast_tree.ast_parser import bfs, children, AstNodes, PythonKeywords, ast_print
from utils.fun_utils import ast_parse_file


def ast_name(node):
    return (node.__class__.__name__,)

def ast_paths(here, path_to_here=()):
    path_to_here += ast_name(here)
    for v in children(here):
        # Iterate over all key-value pairs in the node...
        for p in ast_paths(v, path_to_here ):
            # produce all paths rooted at that subtree; adding k to the
            # current path produces a set of alternating key-value sets
            yield p
    if len(list(children(here))) == 0:
        yield path_to_here

class TreeFeatures:
    def __init__(self):
        self.astnodes = AstNodes()
        self.keywords = PythonKeywords()

    def tf_ngrams_node(self, ast_tree, ngram=2):
        # out = []
        out = defaultdict(int)

        def ngrams_nodes(x, d, o, ngram=ngram):
            successors = list(children(x))
            if len(successors) > 0:
                father = x
                for grams in zip(*[successors[i:] for i in range(0, ngram - 1)]):
                    grams = (self.astnodes.index(father),) + tuple(self.astnodes.index(gram) for gram in grams)
                    o[self.astnodes.index(grams)] += 1

        return bfs(ast_tree, callback=ngrams_nodes, mode="all", out=out)

    def tf_ngrams_node_fast(self, ast_tree, ngram=2):
        out = defaultdict(int)

        def ngrams_nodes(x, d, o, ngram=ngram, predecessor=tuple()):

            if len(predecessor) < ngram:
                predecessor = predecessor + (x,)  # (type(x).__name__,)
                for child in children(x):
                    grams = ngrams_nodes(child, d, o, ngram=ngram, predecessor=predecessor)
                    if len(grams) == ngram:
                        grams_idx = tuple(self.astnodes.index(gram) for gram in grams)
                        o[self.astnodes.index(grams_idx)] += 1
            return predecessor

        return bfs(ast_tree, callback=ngrams_nodes, mode="all", out=out)

    def tf_skip_grams_node_fast(self, ast_tree, ngram=2,v_skip=0):
        out = defaultdict(int)

        def ngrams_nodes(x, d, o, ngram=ngram,v_skip=v_skip,predecessor=tuple()):

            if len(predecessor) < ngram+v_skip:
                predecessor = predecessor + (x,)  # (type(x).__name__,)
                for child in children(x):
                    grams = ngrams_nodes(child, d, o, ngram=ngram,v_skip=v_skip,predecessor=predecessor)
                    grams = grams[::v_skip+1]
                    if len(grams) == ngram:
                        grams_idx = tuple(self.astnodes.index(gram) for gram in grams)
                        # grams_idx = tuple(type(gram).__name__ for gram in grams)
                        o[self.astnodes.index(grams_idx)] += 1
                        # o[grams_idx] += 1
            return predecessor

        return bfs(ast_tree, callback=ngrams_nodes, mode="all", out=out)

    def ngrams_node_fast(self, ast_tree, ngram=2):
        out = []

        def ngrams_nodes(x, d, o):
            # grams_idx = tuple(self.astnodes.index(gram) for gram in grams)
            # grams_idx = tuple(type(gram).__name__ for gram in grams)
            o.append(self.astnodes.index(x))

        return bfs(ast_tree, callback=ngrams_nodes, mode="leaves", out=out)

    def max_depth(self, ast_tree):
        def max_depth_lambda(x, d, o):
            if len(o) == 0:
                o.append(d)
            elif d > o[0]:
                o[0] = d

        out = bfs(ast_tree, callback=max_depth_lambda, mode="leaves", out=[])
        return out[0]

    def tf_keywords(self, ast_tree):
        def nuke_all_strings(mod):
            if isinstance(mod, ast.Str):
                mod.s = ''
            for child in children(mod):
                nuke_all_strings(child)

        ast_tree1 = copy.deepcopy(ast_tree)
        nuke_all_strings(ast_tree1)
        codestr = cg.to_source(ast_tree1)
        ws = re.compile(("\_*[a-zA-Z]+[a-zA-Z0-9\_]*"))
        out = defaultdict(int)
        for x in ws.findall(codestr):
            idx = self.keywords.try_index(x)
            if idx != -1:
                out[idx] += 1
        return out

    def tf_node_types(self, ast_tree):
        out = defaultdict(int)

        def tf_nodes(x, d, o):
            out[self.astnodes.index(x)] += 1

        out = bfs(ast_tree, callback=tf_nodes, mode="parents", out=out)
        return out

    def tf_node_leaves(self, ast_tree):
        out = defaultdict(int)

        def tf_nodes(x, d, o):
            o[self.astnodes.index(x)] += 1

        out = bfs(ast_tree, callback=tf_nodes, mode="leaves", out=out)
        return out

    def avg_node_types(self, ast_tree):
        out = defaultdict(list)

        def avg_nodes(x, d, o):
            info = o[self.astnodes.index(x)]
            if len(info) == 0:
                info.extend([0, 0])
            info[0] += d
            info[1] += 1

        out = bfs(ast_tree, callback=avg_nodes, mode="parents", out=out)
        out_avg = {k: v[0] / v[1] for k, v in out.items()}
        return out_avg

    def avg_node_leaves(self, ast_tree):
        out = defaultdict(list)

        def avg_nodes(x, d, o):
            info = o[self.astnodes.index(x)]
            if len(info) == 0:
                info.extend([0, 0])
            info[0] += d
            info[1] += 1

        out = bfs(ast_tree, callback=avg_nodes, mode="leaves", out=out)
        out_avg = {k: v[0] / v[1] for k, v in out.items()}
        return out_avg


class ASTVectorizer(BaseEstimator):
    def __init__(self, ngram=2,v_skip=0, normalize=True, idf=False, norm="l2", binary=False, dtype=np.float32):
        self.ngram = ngram
        self.v_skip = v_skip
        self.idf = idf
        self.normalize = normalize
        self.norm = norm
        self.tree_features = TreeFeatures()
        self.dtype = dtype
        self.binary = binary

    def fit(self, X, y=None, verbose=False):
        self.features_categories = []
        X_list = self._fit(X, y)
        # Extract TFIDF
        if self.idf:
            smooth_idf = True
            sublinear_tf = False
            self.idf_ngrams_node = TfidfTransformer(norm=self.norm, use_idf=self.idf, smooth_idf=smooth_idf,
                                                    sublinear_tf=sublinear_tf)
            self.idf_node_types = TfidfTransformer(norm=self.norm, use_idf=self.idf, smooth_idf=smooth_idf,
                                                   sublinear_tf=sublinear_tf)
            self.idf_node_leaves = TfidfTransformer(norm=self.norm, use_idf=self.idf, smooth_idf=smooth_idf,
                                                    sublinear_tf=sublinear_tf)
            self.idf_ngrams_node.fit(X_list[0])
            self.idf_node_types.fit(X_list[1])
            self.idf_node_leaves.fit(X_list[2])

        self.features_categories.extend(["tf_ngrams_node_sp" for s in range(X_list[0].shape[1])])
        self.features_categories.extend(["tf_node_types_sp" for s in range(X_list[1].shape[1])])
        self.features_categories.extend(["tf_node_leaves_sp" for s in range(X_list[2].shape[1])])
        self.features_categories.extend(["tf_node_keywords_sp" for s in range(X_list[3].shape[1])])
        # self.features_categories.extend(["max_node_depth_sp" for s in range(X_list[3].shape[1])])
        self.features_categories.extend(["avg_node_types_depth_sp" for s in range(X_list[4].shape[1])])
        self.features_categories.extend(["avg_node_leaves_depth_sp" for s in range(X_list[5].shape[1])])
        self.features_categories.extend(["idf_ngrams_node" for s in range(X_list[0].shape[1])])
        self.features_categories.extend(["idf_node_types" for s in range(X_list[1].shape[1])])
        self.features_categories.extend(["idf_node_leaves" for s in range(X_list[2].shape[1])])
        # self.features_categories.extend(["idf_ngrams_node2" for s in range(X_list[5].shape[1])])

        return self  # sp.hstack(X_list, dtype=self.dtype)

    def transform(self, X, copy=True):
        X_list = self._fit(X, None)
        # Extract TFIDF
        if self.idf:
            X_list.extend([self.idf_ngrams_node.transform(X_list[0]),
                           self.idf_node_types.transform(X_list[1]),
                           self.idf_node_leaves.transform(X_list[2])])
                           # self.idf_ngrams_node2.transform(X_list[5])])

        return sp.csr_matrix(sp.hstack(X_list, dtype=self.dtype))

    def _fit(self, X, y):
        max_node_depth = []
        tf_ngrams_node = []
        tf_node_types = []
        tf_node_leaves = []
        tf_node_keywords = []
        avg_node_types_depth = []
        avg_node_leaves_depth = []

        for ast_tree in X:
            try:
                # ast_tree = ast_parse_file(x)
                # Extract N-grams
                tf_ngrams_node_local = Counter(defaultdict(int))
                for i in range(self.v_skip+1):
                    tf_ngrams_node_local += Counter(self.tree_features.tf_skip_grams_node_fast(ast_tree, ngram=self.ngram,v_skip=i))
                tf_ngrams_node.append(tf_ngrams_node_local)

                # Extract TF
                tf_node_types.append(self.tree_features.tf_node_types(ast_tree))
                tf_node_leaves.append(self.tree_features.tf_node_leaves(ast_tree))
                tf_node_keywords.append(self.tree_features.tf_keywords(ast_tree))

                # Extract AVG Depth
                avg_node_types_depth.append(self.tree_features.avg_node_types(ast_tree))
                avg_node_leaves_depth.append(self.tree_features.avg_node_leaves(ast_tree))

                # Extract Max depth
                max_node_depth.append([self.tree_features.max_depth(ast_tree)])
            except Exception as e:
                print("ERROR: ERROR", e)
                print("ERROR: filename", x)
                raise

        # transform features into sparse representation
        tf_ngrams_node_sp = self._normalize(
            self._to_sparse(tf_ngrams_node, self.tree_features.astnodes.size() ** self.ngram))
        tf_node_types_sp = self._normalize(self._to_sparse(tf_node_types, self.tree_features.astnodes.size()))
        tf_node_leaves_sp = self._normalize(self._to_sparse(tf_node_leaves, self.tree_features.astnodes.size()))
        tf_node_keywords_sp = self._normalize(self._to_sparse(tf_node_keywords, self.tree_features.keywords.size()))
        avg_node_types_depth_sp = self._normalize(
            self._to_sparse(avg_node_types_depth, self.tree_features.astnodes.size()))
        avg_node_leaves_depth_sp = self._normalize(
            self._to_sparse(avg_node_leaves_depth, self.tree_features.astnodes.size()))
        # max_node_depth_sp = self._normalize(sp.csr_matrix(max_node_depth, dtype=self.dtype))

        X_list = [tf_ngrams_node_sp,
                  tf_node_types_sp,
                  tf_node_leaves_sp,
                  # max_node_depth_sp,
                  tf_node_keywords_sp,
                  avg_node_types_depth_sp,
                  avg_node_leaves_depth_sp]
                  # tf_ngrams_node2_sp]

        return X_list

    def _to_sparse(self, X, features_size):
        indice_c = []
        indice_r = []
        data = []
        for idx, x in enumerate(X):
            indice_c.extend(list(x.keys()))
            indice_r.extend(idx * np.ones(len(x)))
            if self.binary:
                data.extend(np.ones(len(x)))
            else:
                data.extend(list(x.values()))
        return sp.csr_matrix((data, (indice_r, indice_c)), shape=(len(X), features_size), dtype=self.dtype)

    def _normalize(self, X):
        if self.normalize:
            X_out = normalize(X.astype(np.float64), norm=self.norm)
            # np.log(X_out.data,X_out.data)
            return X_out
        else:
            return X

import os
if __name__ == "__main__":
    filename = os.path.join(os.getcwd(), 'dump_program.py')
    ast_tree = ast_parse_file(filename)
    print(list(ast_paths(ast_tree)))
    ast_print(ast_tree)
    features = TreeFeatures()
    bigrams = sorted(features.tf_skip_grams_node_fast(ast_tree,ngram=2,v_skip=5).items())
    for k, v in bigrams:
        print(k, " => ", v)
