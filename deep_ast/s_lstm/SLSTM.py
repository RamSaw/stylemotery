import numpy
import six

import chainer
from deep_ast.s_lstm import slstm_func
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable
import numpy as np

class LSTMBase(link.Chain):
    def __init__(self, in_size, out_size,
                 init=None, inner_init=None, bias_init=0, forget_bias_init=0):
        super(LSTMBase, self).__init__(
            W_i=linear.Linear(in_size, out_size,initialW=init, initial_bias=bias_init),
            U_i=linear.Linear(out_size, out_size,initialW=inner_init, nobias=True),

            W_f=linear.Linear(in_size, out_size,initialW=init, initial_bias=bias_init),
            U_f=linear.Linear(out_size, out_size,initialW=inner_init, nobias=True),

            W_c=linear.Linear(in_size, out_size,initialW=init, initial_bias=bias_init),
            U_c=linear.Linear(out_size, out_size,initialW=inner_init, nobias=True),

            W_o=linear.Linear(in_size, out_size,initialW=init, initial_bias=bias_init),
            U_o=linear.Linear(out_size, out_size,initialW=inner_init, nobias=True),

            # upward=linear.Linear(in_size, 4 * out_size, initialW=0),
            # lateral=linear.Linear(out_size, 4 * out_size,
            #                       initialW=0, nobias=True),
        )
        self.state_size = out_size
        # for i in six.moves.range(0, 4 * out_size, out_size):
        #     initializers.init_weight(
        #         self.lateral.W.data[i:i + out_size, :], lateral_init)
        #     initializers.init_weight(
        #         self.upward.W.data[i:i + out_size, :], upward_init)

        # a, i, f, o = slstm_func._extract_gates(
        #     self.upward.b.data.reshape(1, 4 * out_size, 1))

        #init weight
        initializers.init_weight(self.W_a.W.data, bias_init)
        initializers.init_weight(self.W_i.W.data, bias_init)
        initializers.init_weight(self.W_f.W.data, forget_bias_init)
        initializers.init_weight(self.W_o.W.data, bias_init)

        # init bias
        initializers.init_weight(self.W_a.b.data, bias_init)
        initializers.init_weight(self.W_i.b.data, bias_init)
        initializers.init_weight(self.W_f.b.data, forget_bias_init)
        initializers.init_weight(self.W_o.b.data, bias_init)


class LSTM(LSTMBase):
    def __init__(self, in_size, out_size, **kwargs):
        super(LSTM, self).__init__(in_size, out_size, **kwargs)

    def __call__(self, c, h, x):
        """Returns new cell state and updated output of LSTM.

        Args:
            c (~chainer.Variable): Cell states of LSTM units.
            h (~chainer.Variable): Output at the previous time step.
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            tuple of ~chainer.Variable: Returns ``(c_new, h_new)``, where
                ``c_new`` represents new cell state, and ``h_new`` is updated
                output of LSTM units.

        """
        lstm_in = self.upward(x)
        if h is not None:
            lstm_in += self.lateral(h)
        if c is None:
            xp = self.xp
            c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        return slstm_func.LSTM()(c, lstm_in)

    def upward(self, x):
        a = self.W_a(x)
        i = self.W_i(x)
        f = self.W_f(x)
        o = self.W_o(x)
        return np.concatenate([a, i, f, o])


