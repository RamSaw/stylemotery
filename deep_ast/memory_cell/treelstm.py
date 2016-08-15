import chainer
import numpy
from chainer import functions as F
from chainer import link
from chainer.links.connection import linear


class TreeLSTMBase(link.Chain):
    U_I_H = "U_i_h{0}"
    U_O_H = "U_o_h{0}"
    U_A_H = "U_a_h{0}"
    U_F_H = "U_f{0}_h{1}"

    def __init__(self, children, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(TreeLSTMBase, self).__init__(
            W_i=linear.Linear(in_size, out_size, initialW=upward_init, initial_bias=bias_init),
            W_f=linear.Linear(in_size, out_size, initialW=upward_init, initial_bias=forget_bias_init),
            W_a=linear.Linear(in_size, out_size, initialW=upward_init, initial_bias=bias_init),
            W_o=linear.Linear(in_size, out_size, initialW=upward_init, initial_bias=bias_init),
        )
        self.state_size = out_size
        self.n_children = children
        for i in range(self.n_children):
            U_i_name = self.U_I_H.format(i)
            U_o_name = self.U_O_H.format(i)
            U_a_name = self.U_A_H.format(i)

            self.add_link(U_i_name, linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))
            self.add_link(U_o_name, linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))
            self.add_link(U_a_name, linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))

            for j in range(self.n_children):
                U_f_h_name = self.U_F_H.format(i, j)
                self.add_link(U_f_h_name, linear.Linear(out_size, out_size, initialW=lateral_init, nobias=True))


class TreeLSTM(TreeLSTMBase):
    def __init__(self, children, in_size, out_size, **kwargs):
        super(TreeLSTM, self).__init__(children, in_size, out_size, **kwargs)

    def __call__(self, c, h, x):
        if isinstance(x.data, numpy.ndarray):
            xp = numpy
        # else:
        #     xp = cuda.cupy
        a_gate = self.W_a(x)
        i_gate = self.W_i(x)
        f_gate = self.W_f(x)
        o_gate = self.W_o(x)
        f_gates = [f_gate for i in range(self.n_children)]

        if h is not None:
            for i in range(self.n_children):
                U_i_name = self.U_I_H.format(i)
                U_o_name = self.U_O_H.format(i)
                U_a_name = self.U_A_H.format(i)

                U_i_var = getattr(self, U_i_name)
                U_o_var = getattr(self, U_o_name)
                U_a_var = getattr(self, U_a_name)

                a_gate += U_a_var(h[i])
                i_gate += U_i_var(h[i])
                o_gate += U_o_var(h[i])

                for j in range(self.n_children):
                    U_f_name = self.U_F_H.format(i, j)
                    U_f_var = getattr(self, U_f_name)
                    f_gates[i] += U_f_var(h[i])

        # lstm_in = numpy.concatenate((a_gate.data, i_gate.data, o_gate.data))
        # lstm_in = numpy.concatenate([lstm_in] + [g.data for g in f_gates])

        if c is None:
            c = self.xp.zeros((len(x.data) * self.n_children, self.state_size), dtype=x.data.dtype)
        else:
            c = self.xp.concatenate([state.data for state in c])

        self.i = F.sigmoid(i_gate)
        self.f = [F.sigmoid(fs) for fs in f_gate]
        self.o = F.sigmoid(o_gate)
        self.a = F.tanh(a_gate)

        c_prev = [(x * y).data for x, y in zip(self.f, c)]
        self.c = self.a * self.i + numpy.sum(c_prev, axis=0)
        h = self.o * F.tanh(self.c)

        return self.c, h




if __name__ == "__main__":
    import numpy, chainer, chainer.functions as F
    from chainer import links as L

    x = chainer.Variable(numpy.ones((10, 10), dtype=numpy.float32))
    gru = TreeLSTM(lateral_init=1, upward_init=1,bias_init=0, forget_bias_init=0,in_size=10, out_size=10,children=2)
    # gru = L.StatefulGRU(bias_init=0,inner_init=1,in_size=10, out_size=10)
    lstm = L.LSTM(lateral_init=1, upward_init=1,
                 bias_init=0, forget_bias_init=0,in_size=10, out_size=10)
    y1 = gru(x)[1]
    y2 = lstm(x)

    print(y1.data)
    print(y2.data)

    print(y1.data.shape)
    print(y2.data.shape)

    print(numpy.allclose(y1.data,y2.data))
