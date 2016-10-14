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

class RecursiveBaseTree(chainer.Chain):
    def __init__(self,n_children, n_units, n_label,dropout, classes=None):
        super(RecursiveBaseTree, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()
        self.n_children = n_children
        self.dropout = dropout

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def params_count(self):
        count = 0
        for child in self.children():
            for p in child.params():
                count += reduce(operator.mul, p.data.shape, 1)
        return count

    def leaf(self, x, train_mode):
        return self.embed_vec(x, train_mode)

    def embed_vec(self, x, train_mode):
        word = self.xp.array([self.feature_dict.astnodes.index(x)], self.xp.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        return self.embed(w)

    def traverse(self, node, train_mode):
        c,h = self.traverse_rec(node, train_mode= train_mode)
        # self.lstm.reset_state()
        return F.dropout(h,ratio=self.dropout,train=train_mode)

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

    def predict(self, x,index):
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

    def loss(self, x, y, train_mode):
        w = self.label(x)
        label = self.xp.array([y], self.xp.int32)
        t = chainer.Variable(label, volatile=not train_mode)
        return F.softmax_cross_entropy(w, t)

class RecursiveTreeLSTM(RecursiveBaseTree):
    def __init__(self, n_children, n_units, n_label,dropout, classes=None):
        super().__init__(n_children, n_units, n_label,dropout, classes)

        self.add_link("treelstm", FastTreeLSTM(self.n_children, n_units, n_units))

    def leaf(self, x, train_mode):
        vec = super().leaf(x,train_mode=train_mode)
        return self.treelstm(None,None,vec)

    def merge(self, x, children,train_mode):
        c_list, h_list = zip(*children)
        return self.treelstm(F.concat(c_list,axis=0), F.concat(h_list,axis=0), x)

class RecursiveSLSTM(RecursiveBaseTree):
    def __init__(self, n_children, n_units, n_label, classes=None):
        super().__init__(n_children, n_units, n_label, classes)

        self.add_link("slstm", F.SLSTM(self.n_children, n_units, n_units))

    def leaf(self, x, train_mode):
        vec = super().leaf(x,train_mode=train_mode)
        return self.slstm(vec,None,)

    def merge(self, x, children,train_mode):
        c_list, h_list = zip(*children)
        return self.slstm(F.concat(c_list,axis=0), F.concat(h_list,axis=0))





