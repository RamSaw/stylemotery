import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from ast_tree.ASTVectorizater import TreeFeatures
from memory_cell.treelstm import TreeLSTM


class RecursiveNet(chainer.Chain):
    def __init__(self,n_children, n_units, n_label, classes=None):
        super(RecursiveNet, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()
        self.n_children = n_children

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        self.add_link("lstm", TreeLSTM(self.n_children, n_units, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def leaf(self, x, train_mode=False):
        p = self.embed_vec(x, train_mode)
        return self.lstm(None, None, p)

    def embed_vec(self, x, train_mode=False):
        word = self.xp.array([self.feature_dict.astnodes.index(x)], self.xp.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        return self.embed(w)

    def merge(self, x, children):
        c_list, h_list = zip(*children)
        return self.lstm(c_list, h_list, x)

    def label(self, v):
        return self.w(v)

    def predict(self, x):
        t = self.label(x)
        X_prob = F.softmax(t)
        indics_ = cuda.to_cpu(X_prob.data.argmax(axis=1))
        return self.classes_[indics_]

    def loss(self, x, y, train_mode=False):
        w = self.label(x)
        label = self.xp.array([y], self.xp.int32)
        t = chainer.Variable(label, volatile=not train_mode)
        return F.softmax_cross_entropy(w, t)


class RecursiveLSTMNet(chainer.Chain):
    def __init__(self, n_units, n_label, classes=None):
        super(RecursiveLSTMNet, self).__init__()
        self.classes_ = classes
        self.feature_dict = TreeFeatures()

        self.add_link("embed", L.EmbedID(self.feature_dict.astnodes.size() + 1, n_units))
        self.add_link("batch1", L.BatchNormalization(n_units))
        # self.add_link("batch2", L.BatchNormalization(n_units))
        # self.add_link("batch3", L.BatchNormalization(n_units))
        self.add_link("lstm1", L.LSTM(n_units, n_units))
        # self.add_link("lstm2", L.LSTM(n_units, n_units))
        # self.add_link("lstm3", L.LSTM(n_units, n_units))
        self.add_link("w", L.Linear(n_units, n_label))

    def leaf(self, x, train_mode=False):
        return self.embed_vec(x, train_mode)

    def embed_vec(self, x, train_mode=False):
        word = self.xp.array([self.feature_dict.astnodes.index(x)], self.xp.int32)
        w = chainer.Variable(word, volatile=not train_mode)
        return self.embed(w)

    def merge(self, x, children, train_mode=False):
        # c_list,h_list = zip(*children)
        # h0 = self.lstm1(self.batch1(x))  # self.batch(
        # h1 = self.lstm2(self.batch2(h0))  # self.batch(
        # h2 = F.dropout(self.lstm3(self.batch3(h1)),train=train_mode)  # self.batch(
        h0 = self.lstm1(x)  # self.batch(
        for child in children:
            h0 = self.lstm1(child)
        # h1 = F.dropout(self.lstm2(self.batch2(h0)),train=train_mode)  # self.batch(
        # h2 = F.dropout(self.lstm3(self.batch3(h1)),train=train_mode)  # self.batch(
        self.lstm1.reset_state()
        # self.lstm2.reset_state()
        # self.lstm3.reset_state()
        return F.dropout(h0)

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
