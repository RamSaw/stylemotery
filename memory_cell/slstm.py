import chainer
import numpy
from chainer import functions as F
from chainer import variable
from chainer import link
from chainer.functions import split_axis
from chainer.links.connection import linear
from chainer import initializers
from chainer.functions.array import concat

from memory_cell.simple_lstm import LSTMBase


def numpy_extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in range(4)]

def chainer_extract_gates(x):
    r = F.reshape(x,(len(x.data), x.data.shape[1] // 4, 4) + x.data.shape[2:])
    return [r[:, :, i] for i in range(4)]

class SLSTM(link.Chain):
    U_I_H = "U_i_h{0}"
    U_O_H = "U_o_h{0}"
    U_A_H = "U_a_h{0}"
    U_F_H = "U_f{0}_h{1}"

    def __init__(self, children, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(SLSTM, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size, initialW=0)
        )
        self.state_size = out_size
        self.n_children = children

        for i in range(0, 4 * out_size, out_size):
            initializers.init_weight(self.upward.W.data[i:i + out_size, :], upward_init)
        a, i, f, o = numpy_extract_gates(self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializers.init_weight(a, bias_init)
        initializers.init_weight(i, bias_init)
        initializers.init_weight(f, forget_bias_init)
        initializers.init_weight(o, bias_init)

        #hidden unit gates for each child
        for i in range(self.n_children):
            self.add_link(self.U_I_H.format(i), linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))
            self.add_link(self.U_O_H.format(i), linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))
            self.add_link(self.U_A_H.format(i), linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))

            for j in range(self.n_children):
                self.add_link(self.U_F_H.format(i, j),linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))

    def __call__(self, c, h):
        tree_in = self.upward(x)
        a_gate,i_gate,f_gate,o_gate = chainer_extract_gates(tree_in)

        f_gates = F.concat([f_gate for i in range(self.n_children)], axis=0)

        if h is not None:
            for i in range(self.n_children):
                U_i_var = getattr(self, self.U_I_H.format(i))
                U_o_var = getattr(self, self.U_O_H.format(i))
                U_a_var = getattr(self, self.U_A_H.format(i))

                h_v = F.reshape(h[i], (1, -1))
                a_gate += U_a_var(h_v)
                i_gate += U_i_var(h_v)
                o_gate += U_o_var(h_v)

                fs = F.concat([getattr(self, self.U_F_H.format(i, j))(h_v) for j in range(self.n_children)], axis=0)
                f_gates += fs
        if c is None:
            c = variable.Variable(self.xp.zeros((len(x.data) * self.n_children, self.state_size), dtype=x.data.dtype),
                                  volatile=x.volatile)

        i = F.sigmoid(i_gate)
        f = F.sigmoid(f_gates)
        o = F.sigmoid(o_gate)
        a = F.tanh(a_gate)

        c_prev = f * c
        c_cur = a * i + F.reshape(F.sum(c_prev, axis=0), (1, -1))
        h = o * F.tanh(c_cur)

        return c_cur, h

    def func(self,c, h):
        # self.a1 = numpy.tanh(a1)
        # self.i1 = _sigmoid(i1)
        # self.f1 = _sigmoid(f1)
        #
        # self.a2 = numpy.tanh(a2)
        # self.i2 = _sigmoid(i2)
        # self.f2 = _sigmoid(f2)
        #
        # self.o = _sigmoid(o1 + o2)
        # self.c = self.a1 * self.i1 + self.a2 * self.i2 + \
        #          self.f1 * c_prev1 + self.f2 * c_prev2
        #
        # h = self.o * numpy.tanh(self.c)
        a1, i1, f1, o1 = chainer_extract_gates(h)
        a2, i2, f2, o2 = chainer_extract_gates(h)


        self.c = F.Tanh(a1) * F.Sigmoid(i1) + F.Sigmoid(f1) * c_prev1 + \
                 F.Tanh(a2) * F.Sigmoid(i2) + F.Sigmoid(f2) * c_prev2

        self.o = F.Sigmoid(o1 + o2)
        h = self.o * numpy.tanh(self.c)

class BiSLSTM(LSTMBase):
    def __init__(self, in_size, out_size, **kwargs):
        super(BiSLSTM, self).__init__(in_size, out_size, **kwargs)
        self.reset_state()

    def to_cpu(self):
        super(BiSLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(BiSLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, c, h):
        assert isinstance(c, chainer.Variable)
        assert isinstance(h, chainer.Variable)
        c_ = c
        h_ = h
        if self.xp == numpy:
            c_.to_cpu()
            h_.to_cpu()
        else:
            c_.to_gpu()
            h_.to_gpu()
        self.c = c_
        self.h = h_

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x):
        batch = x.shape[0]
        lstm_in = self.upward(x)
        h_rest = None
        if self.h is not None:
            h_size = self.h.shape[0]
            if batch == 0:
                h_rest = self.h
            elif h_size < batch:
                msg = ('The batch size of x must be equal to or less than the '
                       'size of the previous state h.')
                raise TypeError(msg)
            elif h_size > batch:
                h_update, h_rest = split_axis.split_axis(
                    self.h, [batch], axis=0)
                lstm_in += self.lateral(h_update)
            else:
                lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((batch, self.state_size), dtype=x.dtype),
                volatile='auto')

        return self.c, lstm_in