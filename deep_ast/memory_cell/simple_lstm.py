from chainer import functions as F
from chainer import link
from chainer import variable
from chainer.links.connection import linear


class LSTMBase(link.Chain):
    def __init__(self, in_size, out_size,
                 init=None,upward_init=None,lateral_init=None, inner_init=None, bias_init=0, forget_bias_init=0):
        super(LSTMBase, self).__init__(
            W_i=linear.Linear(in_size, out_size,initialW=upward_init, initial_bias=bias_init),
            U_i=linear.Linear(out_size, out_size,initialW=lateral_init, nobias=True),

            W_f=linear.Linear(in_size, out_size,initialW=upward_init, initial_bias=forget_bias_init),
            U_f=linear.Linear(out_size, out_size,initialW=lateral_init, nobias=True),

            W_a=linear.Linear(in_size, out_size,initialW=upward_init, initial_bias=bias_init),
            U_a=linear.Linear(out_size, out_size,initialW=lateral_init, nobias=True),

            W_o=linear.Linear(in_size, out_size,initialW=upward_init, initial_bias=bias_init),
            U_o=linear.Linear(out_size, out_size,initialW=lateral_init, nobias=True),
        )
        self.state_size = out_size

class StatelessLSTM(LSTMBase):
    def __call__(self, c, h, x):
        a = self.W_a(x)
        i = self.W_i(x)
        f = self.W_f(x)
        o = self.W_o(x)

        if h is not None:
            a += self.U_a(x)
            i += self.U_i(x)
            f += self.U_f(x)
            o += self.U_o(x)

        if c is None:
            xp = self.xp
            c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')

        a = F.tanh(a)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        o = F.sigmoid(o)

        c = a * i + f * c
        h = o * F.tanh(c)
        return c, h


class StatefulLSTM(LSTMBase):
    def __init__(self, in_size, out_size, **kwargs):
        super(StatefulLSTM, self).__init__(in_size, out_size, **kwargs)
        self.c = self.h = None

    def __call__(self, x):
        a = self.W_a(x)
        i = self.W_i(x)
        f = self.W_f(x)
        o = self.W_o(x)

        if self.h is not None:
            a += self.U_a(x)
            i += self.U_i(x)
            f += self.U_f(x)
            o += self.U_o(x)

        if self.c is None:
            xp = self.xp
            c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')

        self.a = F.tanh(a)
        self.i = F.sigmoid(i)
        self.f = F.sigmoid(f)
        self.o = F.sigmoid(o)

        self.c = self.a * self.i + self.f * c
        self.h = self.o * F.tanh(self.c)
        return self.h



if __name__ == "__main__":
    import numpy, chainer, chainer.functions as F
    from chainer import links as L

    x = chainer.Variable(numpy.ones((10, 10), dtype=numpy.float32))
    # gru = TreeLSTM(lateral_init=1, upward_init=1,bias_init=0, forget_bias_init=0,in_size=10, out_size=10,children=2)
    gru = L.StatefulGRU(bias_init=0,inner_init=1,in_size=10, out_size=10)
    lstm = L.LSTM(lateral_init=1, upward_init=1,
                 bias_init=0, forget_bias_init=0,in_size=10, out_size=10)
    y1 = gru(x)[1]
    y2 = lstm(x)

    print(y1.data)
    print(y2.data)

    print(y1.data.shape)
    print(y2.data.shape)

    print(numpy.allclose(y1.data,y2.data))