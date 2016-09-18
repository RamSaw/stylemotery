import operator
from functools import reduce

import chainer
import chainer.functions as F
import chainer.variable as variable
import chainer.links as L
from chainer import cuda

from ast_tree.ASTVectorizater import TreeFeatures
from ast_tree.ast_parser import children
from memory_cell.treelstm import FastTreeLSTM


class RecursiveTreeLSTM(chainer.Chain):
    def __init__(self,n_children, n_units, n_label, classes=None):
        super(RecursiveTreeLSTM, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()
        self.n_children = n_children

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        self.add_link("lstm", FastTreeLSTM(self.n_children, n_units, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def leaf(self, x, train_mode=False):
        p = self.embed_vec(x, train_mode)
        return self.lstm(None, None, p)

    def embed_vec(self, x, train_mode=False):
        word = self.xp.array([self.feature_dict.astnodes.index(x)], self.xp.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        return self.embed(w)

    def merge(self, x, children,train_mode):
        c_list, h_list = zip(*children)
        return self.lstm(F.concat(c_list,axis=0), F.concat(h_list,axis=0), x)

    def traverse(self, node, train_mode):
        c,h = self.traverse_rec(node, train_mode= train_mode)
        # self.lstm.reset_state()
        return F.dropout(h,train=train_mode)

    def traverse_rec(self, node, train_mode):
        children_ast = list(children(node))
        if len(children_ast) == 0:
            # leaf node
            lf = self.leaf(node, train_mode=train_mode)
            return lf
        else:
            # internal node
            children_nodes = []
            for child in children_ast:
                child_node = self.traverse_rec(child, train_mode=train_mode)
                children_nodes.append(child_node)
            if len(children_nodes) < self.n_children:
                c, h = self.leaf(None, train_mode)
                children_nodes.extend([(c,h) for i in range(self.n_children - len(children_nodes))])
            elif len(children_nodes) > self.n_children:
                children_nodes = children_nodes[:self.n_children]
            x = self.embed_vec(node, train_mode=train_mode)
            return self.merge(x, children_nodes, train_mode=train_mode)

    def label(self, v):
        return self.w(v)

    def predict(self, x,index=False):
        t = self.label(x)
        X_prob = F.softmax(t)
        indics_ = cuda.to_cpu(X_prob.data.argmax(axis=1))
        if index:
            return indics_
        else:
            return self.classes_[indics_]

    def predict_proba(self, x):
        t = self.label(x)
        X_prob = F.softmax(t)
        return cuda.to_cpu(X_prob.data)[0]

    def loss(self, x, y, train_mode=False):
        w = self.label(x)
        label = self.xp.array([y], self.xp.int32)
        t = chainer.Variable(label, volatile=not train_mode)
        return F.softmax_cross_entropy(w, t)


