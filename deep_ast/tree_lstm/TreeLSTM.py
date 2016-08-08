import numpy as np
import numpy
import six
from chainer import cuda
from chainer import function
from chainer import link
from chainer import variable
from chainer.links.connection import linear
from chainer.utils import type_check
from chainer import initializers
from deep_ast.tree_lstm import treelstm_func

class TreeLSTMBase(link.Chain):
    def __init__(self, in_size, out_size,
                 lateral_init=None, upward_init=None,
                 bias_init=0, forget_bias_init=0):
        super(TreeLSTMBase, self).__init__(
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

        a, i, f, o = treelstm_func._extract_gates(
            self.upward.b.data.reshape(1, 4 * out_size, 1))
        initializers.init_weight(a, bias_init)
        initializers.init_weight(i, bias_init)
        initializers.init_weight(f, forget_bias_init)
        initializers.init_weight(o, bias_init)


class TreeLSTM(TreeLSTMBase):
    def __init__(self, in_size, out_size, **kwargs):
        super(TreeLSTM, self).__init__(in_size, out_size, **kwargs)

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
        return treelstm_func.TreeLSTMFunction()(c, lstm_in)

    def upward(self, x):
        a = self.W_a(x)
        i = self.W_i(x)
        f = self.W_f(x)
        o = self.W_o(x)
        return np.concatenate([a, i, f, o])


    def lateral(self, x):
        a = self.W_a(x)
        i = self.W_i(x)
        f = self.W_f(x)
        o = self.W_o(x)
        return np.concatenate([a, i, f, o])


