import chainer
import numpy
from chainer import functions as F
from chainer import variable
from chainer import link
from chainer import links as L
from chainer import initializers


def numpy_extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in range(4)]


def chainer_extract_gates(x):
    r = F.reshape(x, (len(x.data), x.data.shape[1] // 4, 4) + x.data.shape[2:])
    return [r[:, :, i] for i in range(4)]


class FastTreeLSTM(link.Chain):
    U_I_H = "U_i_h{0}"
    U_O_H = "U_o_h{0}"
    U_A_H = "U_a_h{0}"
    U_F_H = "U_f{0}_h{1}"

    def __init__(self, children, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(FastTreeLSTM, self).__init__(
            upward=L.Linear(in_size, 4 * out_size, initialW=0)
        )
        for i in range(0, 4 * out_size, out_size):
            initializers.init_weight(self.upward.W.data[i:i + out_size, :], upward_init)
        a, i, f, o = numpy_extract_gates(self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializers.init_weight(a, bias_init)
        initializers.init_weight(i, bias_init)
        initializers.init_weight(f, forget_bias_init)
        initializers.init_weight(o, bias_init)

        self.state_size = out_size
        self.n_children = children
        for i in range(self.n_children):
            self.add_link(self.U_I_H.format(i), L.Linear(out_size, out_size, initialW=lateral_init, nobias=True))
            self.add_link(self.U_O_H.format(i), L.Linear(out_size, out_size, initialW=lateral_init, nobias=True))
            self.add_link(self.U_A_H.format(i), L.Linear(out_size, out_size, initialW=lateral_init, nobias=True))

            for j in range(self.n_children):
                self.add_link(self.U_F_H.format(i, j), L.Linear(out_size, out_size, initialW=lateral_init, nobias=True))

    def __call__(self, c, h, x):
        tree_in = self.upward(x)
        a_gate, i_gate, f_gate, o_gate = chainer_extract_gates(tree_in)

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


class FasterTreeLSTM(link.Chain):
    def __init__(self, children, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(FasterTreeLSTM, self).__init__(
            upward=L.Linear(in_size, 4 * out_size, initialW=0),
            lateral=L.Linear(children * out_size, 3 * children * out_size, initialW=0, nobias=True),
            forget=L.Linear(children * out_size, children * children * out_size, initialW=0, nobias=True)
        )
        self.state_size = out_size
        self.n_children = children

        for i in range(0, 4 * out_size, out_size):
            initializers.init_weight(self.upward.W.data[i:i + out_size, :], upward_init)
            for j in range(0, 4 * out_size, out_size):
                initializers.init_weight(self.lateral.W.data[i + j:i + j + out_size, :], lateral_init)
            for j in range(0, self.n_children * out_size, out_size):
                initializers.init_weight(self.forget.W.data[i + j:i + j + out_size, :], lateral_init)

        a, i, f, o = numpy_extract_gates(self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializers.init_weight(a, bias_init)
        initializers.init_weight(i, bias_init)
        initializers.init_weight(f, forget_bias_init)
        initializers.init_weight(o, bias_init)

    def __call__(self, c, h, x):
        tree_in = self.upward(x)
        a_gate, i_gate, f_gate, o_gate = chainer_extract_gates(tree_in)

        f_gates = F.concat([f_gate for i in range(self.n_children)], axis=0)

        if h is not None:
            h_v = self.lateral(h)

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


if __name__ == "__main__":
    fast = FasterTreeLSTM(2, 300, 300, lateral_init=2, upward_init=2)
    c_prev1 = chainer.Variable(numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32))
    c_prev2 = chainer.Variable(numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32))
    x1 = chainer.Variable(numpy.random.uniform(-1, 1, (3, 8, 4)).astype(numpy.float32))
    x2 = chainer.Variable(numpy.random.uniform(-1, 1, (3, 8, 4)).astype(numpy.float32))

    x = F.concat((x1, x2), axis=0)
    c = F.concat((c_prev1, c_prev2), axis=0)

    print(x.data.shape)
    print(c.data.shape)
