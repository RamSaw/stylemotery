import chainer
import numpy
from chainer import functions


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class TestLSTM():

    def setUp(self):
        hidden_shape = (3, 2, 4)
        x_shape = (self.batch, 8, 4)
        y_shape = (self.batch, 2, 4)
        self.c_prev = numpy.random.uniform(-1, 1, hidden_shape).astype(self.dtype)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)


    def flat(self):
        self.c_prev = self.c_prev[:, :, 0].copy()
        self.x = self.x[:, :, 0].copy()

    def check_forward(self, c_prev_data, x_data):
        c_prev = chainer.Variable(c_prev_data)
        x = chainer.Variable(x_data)

        c, h = functions.lstm(c_prev, x)
        batch = len(x_data)

        # Compute expected out
        a_in = self.x[:, [0, 4]]
        i_in = self.x[:, [1, 5]]
        f_in = self.x[:, [2, 6]]
        o_in = self.x[:, [3, 7]]

        c_expect = _sigmoid(i_in) * numpy.tanh(a_in) + \
            _sigmoid(f_in) * self.c_prev[:batch]
        h_expect = _sigmoid(o_in) * numpy.tanh(c_expect)

        print("state      = ",numpy.allclose(c_expect, c.data[:batch]))
        print("hidden     = ",numpy.allclose(h_expect, h.data))
        print("prev state = ",numpy.allclose(c_prev_data[batch:], c.data[batch:]))

    def test_forward_cpu(self):
        self.check_forward(self.c_prev, self.x)

    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()


if __name__ == "__main__":
    test = TestLSTM()
    test.batch = 3
    test.dtype = numpy.float32

    test.setUp()
    print("Testing forward")
    test.test_forward_cpu()

    print("Testing flat forward")
    test.test_flat_forward_cpu()