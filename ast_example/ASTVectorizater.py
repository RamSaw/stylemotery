import ast
import copy
import re
from collections import defaultdict
import codegen as cg
from tqdm import tqdm
from sklearn.base import BaseEstimator
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from ast_example.ast_parser import bfs, children, AstNodes, PythonKeywords, ast_parse_file


class TreeFeatures:
    def __init__(self):
        self.astnodes = AstNodes()
        self.keywords = PythonKeywords()

    def tf_ngrams_node(self, ast_tree):
        # out = []
        out = defaultdict(int)

        def ngrams_nodes(x, d, o, ngram=2):
            successors = list(children(x))
            if len(successors) > 0:
                father = x
                for grams in zip(*[successors[i:] for i in range(0, ngram - 1)]):
                    grams = (self.astnodes.index(father),) + tuple(self.astnodes.index(gram) for gram in grams)
                    o[self.astnodes.index(grams)] += 1

        return bfs(ast_tree, callback=ngrams_nodes, mode="all", out=out)

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
    def __init__(self, normalize=True, idf=False, norm="l2", binary=False, dtype=np.float32):
        self.idf = idf
        self.normalize = normalize
        self.norm = norm
        self.tree_features = TreeFeatures()
        self.dtype = dtype
        self.binary = binary

    def fit(self, X, y=None, verbose=False):
        X_list = self._fit(X,y)
        # Extract TFIDF
        if self.idf:
            smooth_idf = True
            sublinear_tf = False
            self.idf_ngrams_node = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=smooth_idf,
                                                    sublinear_tf=sublinear_tf)
            self.idf_node_types = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=smooth_idf,
                                                   sublinear_tf=sublinear_tf)
            self.idf_node_leaves = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=smooth_idf,
                                                    sublinear_tf=sublinear_tf)
            self.idf_ngrams_node.fit(X_list[0])
            self.idf_node_types.fit(X_list[1])
            self.idf_node_leaves.fit(X_list[2])

        return self  # sp.hstack(X_list, dtype=self.dtype)

    def transform(self, X, copy=True):
        X_list = self._fit(X,None)
        # Extract TFIDF
        if self.idf:
            #self.idf_ngrams_node.transform(X_list[0]),
            X_list.extend([self.idf_node_types.transform(X_list[1]),
                           self.idf_node_leaves.transform(X_list[2])])
        return sp.csr_matrix(sp.hstack(X_list, dtype=self.dtype))

    def _fit(self, X, y):
        max_node_depth = []
        tf_ngrams_node = []
        tf_node_types = []
        tf_node_leaves = []
        tf_node_keywords = []
        avg_node_types_depth = []
        avg_node_leaves_depth = []

        for x in X:
            try:
                ast_tree = ast_parse_file(x)

                # Extract N-grams
                tf_ngrams_node.append(self.tree_features.tf_ngrams_node(ast_tree))

                # Extract TF
                tf_node_types.append(self.tree_features.tf_node_types(ast_tree))
                tf_node_leaves.append(self.tree_features.tf_node_leaves(ast_tree))
                # tf_node_keywords.append(self.tree_features.tf_keywords(ast_tree))

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
            self._to_sparse(tf_ngrams_node, self.tree_features.astnodes.size() ** 2))
        tf_node_types_sp = self._normalize(self._to_sparse(tf_node_types, self.tree_features.astnodes.size()))
        tf_node_leaves_sp = self._normalize(self._to_sparse(tf_node_leaves, self.tree_features.astnodes.size()))
        # tf_node_keywords_sp = self._normalize(self._to_sparse_matrix(tf_node_keywords, self.tree_features.keywords.size()))
        avg_node_types_depth_sp = self._normalize(
            self._to_sparse(avg_node_types_depth, self.tree_features.astnodes.size()))
        avg_node_leaves_depth_sp = self._normalize(
            self._to_sparse(avg_node_leaves_depth, self.tree_features.astnodes.size()))
        max_node_depth_sp = self._normalize(sp.csr_matrix(max_node_depth, dtype=self.dtype))

        X_list = [tf_ngrams_node_sp,
                  tf_node_types_sp,
                  tf_node_leaves_sp,
                  max_node_depth_sp,
                  # tf_node_keywords_sp
                  avg_node_types_depth_sp,
                  avg_node_leaves_depth_sp]

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

    # def _ngrams_to_sparse_matrix(self, X, voca):
    #     indice_c = []
    #     indice_r = []
    #     data = []
    #     for idx, x in enumerate(X):
    #         for gram in x:
    #             indice_c.append(voca.index(gram))
    #             indice_r.append(idx)
    #             if self.binary:
    #                 data.append(1)
    #             else:
    #                 data.append()
    #     return sp.csr_matrix((data, (indice_r, indice_c)), shape=(len(X), features_size), dtype=self.dtype)

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
    # ast_vec = ASTVectorizer(normalize=True,idf=True, dtype=np.float32)
    # X_transform = ast_vec.fit([filename])
    # print(X_transform.shape)
    # print(X_transform)
    features = TreeFeatures().tf_ngrams_node(ast_tree).items()
    for l in features:
        print(l)
