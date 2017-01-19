import chainer
import numpy
from chainer import variable
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.links.connection import linear
from chainer.links.connection.lstm import LSTMBase

class ConditionalLSTM(LSTMBase):
    def __init__(self, in_size, out_size, **kwargs):
        super(ConditionalLSTM, self).__init__(in_size, out_size, **kwargs)
        self.add_link("w_y", linear.Linear(in_size, in_size, nobias=True))
        self.reset_state()

    def to_cpu(self):
        super(ConditionalLSTM, self).to_cpu()
        if self.c is not None:
            self.c.to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(ConditionalLSTM, self).to_gpu(device)
        if self.c is not None:
            self.c.to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, c, h):
        """Sets the internal state.

        It sets the :attr:`c` and :attr:`h` attributes.

        Args:
            c (~chainer.Variable): A new cell states of LSTM units.
            h (~chainer.Variable): A new output at the previous time step.

        """
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
        """Resets the internal state.

        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def forward(self, x, y):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        if self.upward.has_uninitialized_params:
            in_size = x.size // x.shape[0]
            self.upward._initialize_params(in_size)
            self._initialize_params()

        batch = x.shape[0]
        lstm_in = self.upward(x)

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

        r = reshape.reshape(lstm_in, (len(lstm_in.data), lstm_in.data.shape[1] // 4, 4) + lstm_in.data.shape[2:])
        a, i, f, o = [r[:, :, i] for i in range(4)]

        # self.c, y = lstm.lstm(self.c,lstm_in)

        a = tanh.tanh(a)  # tanh.tanh(a)
        i = sigmoid.sigmoid(i)
        f = sigmoid.sigmoid(f)
        o = sigmoid.sigmoid(o)

        self.c = a * i + f * self.c + tanh(self.w_y(y))
        self.h = o * tanh.tanh(self.c)

        return self.h




# X = numpy.random.random((1, 2)).astype(numpy.float32)
# Y = numpy.random.random((1, 10)).astype(numpy.float32)
# lstm = L.LSTM(2, 2, lateral_init=1, upward_init=1, bias_init=1, forget_bias_init=1)
# plstm = PhasedLSTM(2, 2, lateral_init=1, upward_init=1, bias_init=1, forget_bias_init=1)
# print(lstm(lstm(X)).data)
# print(plstm(plstm(X)).data)
