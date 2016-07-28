import numpy
import six

import chainer
from deep_ast.s_lstm import slstm_func
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class LSTMBase(link.Chain):
    def __init__(self, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(LSTMBase, self).__init__(
            upward=linear.Linear(in_size, 4 * out_size, initialW=0),
            lateral=linear.Linear(out_size, 4 * out_size,
                                  initialW=0, nobias=True),
        )
        self.state_size = out_size
        for i in six.moves.range(0, 4 * out_size, out_size):
            initializers.init_weight(
                self.lateral.W.data[i:i + out_size, :], lateral_init)
            initializers.init_weight(
                self.upward.W.data[i:i + out_size, :], upward_init)

        a, i, f, o = slstm_func._extract_gates(
            self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializers.init_weight(a, bias_init)
        initializers.init_weight(i, bias_init)
        initializers.init_weight(f, forget_bias_init)
        initializers.init_weight(o, bias_init)

class LSTM(LSTMBase):
    def __init__(self, in_size, out_size, **kwargs):
        super(LSTM, self).__init__(in_size, out_size, **kwargs)
        self.reset_state()

    def to_cpu(self):
        super(LSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(LSTM, self).to_gpu(device)
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
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, self.h = slstm_func.LSTM()(self.c, lstm_in)
        return self.h