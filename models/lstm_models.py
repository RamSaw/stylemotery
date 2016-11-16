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


class RecursiveBaseLSTM(chainer.Chain):
    def __init__(self, n_units, n_label, dropout, classes=None):
        super(RecursiveBaseLSTM, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()
        self.dropout = dropout

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def leaf(self, x, train_mode):
        return self.embed_vec(x, train_mode)

    def embed_vec(self, x, train_mode):
        word = self.xp.array([self.feature_dict.astnodes.index(x)], self.xp.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        #return F.dropout(self.embed(w),ratio=self.dropout,train=train_mode)
        return self.embed(w)

    def params_count(self):
        count = 0
        for child in self.children():
            for p in child.params():
                count += reduce(operator.mul, p.data.shape, 1)
        return count

    def traverse(self, node, train_mode):
        children_ast = list(children(node))
        if len(children_ast) == 0:
            # leaf node
            return self.leaf(node, train_mode=train_mode)
        else:
            # internal node
            children_nodes = []
            for child in children_ast:
                child_node = self.traverse(child, train_mode=train_mode)
                children_nodes.append(child_node)
            x = self.embed_vec(node, train_mode=train_mode)
            return self.merge(x, children_nodes, train_mode=train_mode)

    def label(self, v):
        return self.w(v)

    def predict(self, x, index=False):
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


class RecursiveLSTM(RecursiveBaseLSTM):
    def __init__(self, n_units, n_label, layers, dropout, classes=None, peephole=False):
        super(RecursiveLSTM, self).__init__(n_units, n_label, dropout=dropout, classes=classes)
        self.layers = layers
        self.base_lstm = L.StatefulPeepholeLSTM if peephole else L.LSTM
        for i in range(1, layers + 1):
            self.add_link("lstm" + str(i), self.base_lstm(n_units, n_units))

    def one_step(self, x, train_mode):
        h = x
        layers = []
        for i in range(1, self.layers + 1):
            layers.append(getattr(self, "lstm" + str(i)))
        for layer in layers:
            if self.dropout > 0.0:
                h = F.dropout(layer(h), ratio=self.dropout, train=train_mode)
            else:
                h = layer(h)
        return h

    def reset_states(self):
        for i in range(1, self.layers + 1):
            layer = getattr(self, "lstm" + str(i))
            layer.reset_state()

    def merge(self, x, children, train_mode):
        # forward
        timestamps = []
        h0 = self.one_step(x, train_mode)  # self.batch(
        for child in children:
            h0 = self.one_step(child, train_mode)
        self.reset_states()
        return h0


class RecursiveBiLSTM(RecursiveLSTM):
    def __init__(self, n_units, n_label, layers, dropout, peephole, classes=None):
        super(RecursiveBiLSTM, self).__init__(n_units, n_label, layers=layers, peephole=peephole, dropout=dropout,
                                              classes=classes)
        self.dropout = dropout
        for i in range(1, layers + 1):
            self.add_link("blstm" + str(i), self.base_lstm(n_units, n_units))
            self.add_link("w_v" + str(i), L.Linear(2 * n_units, n_units))

    # def merge(self, x, children, train_mode):
    #     # forward
    #     h0 = self.lstm1(x)  # self.batch(
    #     for child in children:
    #         h0 = self.lstm1(child)
    #     self.lstm1.reset_state()
    #
    #     # backword
    #     for child in reversed(children):
    #         h1 = self.lstm2(child)
    #     h1 = self.lstm2(x)
    #     self.lstm2.reset_state()
    #     return self.w_v(F.dropout(F.concat((h0, h1), axis=1), ratio=self.dropout, train=train_mode))

    def merge(self, x, children, train_mode):
        root = x
        leaves = children
        for idx in range(1, self.layers + 1):
            # forward
            fw_results = []
            flstm = getattr(self, "lstm{0}".format(idx))
            fw_results.append(flstm(root))
            for child in leaves:
                fw_results.append(flstm(child))
            # backword
            bw_results = []
            blstm = getattr(self, "blstm{0}".format(idx))
            for child in reversed(leaves):
                bw_results.append(blstm(child))
            bw_results.append(blstm(root))

            w_v = getattr(self, "w_v{0}".format(idx))
            if idx == self.layers:
                fh0 = fw_results[-1]
                bh0 = bw_results[-1]
                h_v = w_v(F.dropout(F.concat((fh0, bh0), axis=1), ratio=self.dropout, train=train_mode))
            else:
                h_values = []
                for fh, bh in zip(fw_results, bw_results):
                    h_values.append(w_v(F.dropout(F.concat((fh, bh), axis=1), ratio=self.dropout, train=train_mode)))
                root = h_values[0]
                leaves = h_values[1:]

        self.reset_states()
        return h_v

    def reset_states(self):
        for i in range(1, self.layers + 1):
            layer = getattr(self, "lstm" + str(i))
            layer.reset_state()
            layer = getattr(self, "blstm" + str(i))
            layer.reset_state()


class RecursiveResidualLSTM(RecursiveLSTM):
    def __init__(self, n_units, n_label, layers, dropout, peephole, classes=None):
        super(RecursiveResidualLSTM, self).__init__(n_units, n_label, layers=layers, dropout=dropout, peephole=peephole,
                                                    classes=classes)

    def merge(self, x, children, train_mode=True):
        # forward
        timestamps = []
        h0 = self.one_step(x, train_mode)  # self.batch(
        for child in children:
            h0 = self.one_step(child, train_mode)
        self.reset_states()
        return h0

    def one_step(self, x, train_mode):
        h = x
        for i in range(1, self.layers + 1):
            h_prev = h
            lstm_layer = getattr(self, "lstm" + str(i))
            h = lstm_layer(h) + h_prev
            if self.dropout > 0.0:
                h = F.dropout(h, ratio=self.dropout, train=train_mode)
        return h


class RecursiveTreeBiLSTM(RecursiveLSTM):
    def __init__(self, n_units, n_label, layers,dropout, peephole, classes=None):
        super(RecursiveTreeBiLSTM, self).__init__(n_units, n_label, layers=layers, peephole=peephole, dropout=dropout,
                                                  classes=classes)
        self.dropout = dropout
        for i in range(1, layers + 1):
            self.add_link("blstm" + str(i), self.base_lstm(n_units, n_units))
            if i < layers:
                self.add_link("w_v" + str(i), L.Linear(2 * n_units, n_units))
        self.add_link("h_l",F.Linear(n_units, 4 * n_units))
        self.add_link("h_r",F.Linear(n_units, 4 * n_units))

    def merge(self, x, children, train_mode):
        root = x
        leaves = children
        for idx in range(1, self.layers + 1):
            # forward
            fw_results = []
            flstm = getattr(self, "lstm{0}".format(idx))
            fw_results.append(flstm(root))
            for child in leaves:
                fw_results.append(flstm(child))
            c_r = flstm.c
            # backword
            bw_results = []
            blstm = getattr(self, "blstm{0}".format(idx))
            for child in reversed(leaves):
                bw_results.append(blstm(child))
            bw_results.append(blstm(root))
            c_l = blstm.c
            if idx < self.layers:
                w_v = getattr(self, "w_v{0}".format(idx))
                h_values = []
                for fh, bh in zip(fw_results, bw_results):
                    h_values.append(w_v(F.dropout(F.concat((fh, bh), axis=1), ratio=self.dropout, train=train_mode)))
                root = h_values[0]
                leaves = h_values[1:]
        self.reset_states()
        h_l = self.h_l(F.dropout(bw_results[-1], ratio=self.dropout, train=train_mode))
        h_r = self.h_r(F.dropout(fw_results[-1], ratio=self.dropout, train=train_mode))
        c,h =F.slstm(c_l, c_r, h_l,h_r)
        return F.dropout(h, ratio=self.dropout, train=train_mode)


    def reset_states(self):
        for i in range(1, self.layers + 1):
            layer = getattr(self, "lstm" + str(i))
            layer.reset_state()
            layer = getattr(self, "blstm" + str(i))
            layer.reset_state()
