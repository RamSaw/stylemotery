import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class TestSLSTM():
    def setUp(self):
        self.c_prev1 = numpy.random.uniform(-1,
                                            1, (3, 2, 4)).astype(numpy.float32)
        self.c_prev2 = numpy.random.uniform(-1,
                                            1, (3, 2, 4)).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(numpy.float32)

    def flat(self):
        self.c_prev1 = self.c_prev1[:, :, 0].copy()
        self.c_prev2 = self.c_prev2[:, :, 0].copy()
        self.x1 = self.x1[:, :, 0].copy()
        self.x2 = self.x2[:, :, 0].copy()

    def check_forward(self, c_prev1_data, c_prev2_data, x1_data, x2_data):
        c_prev1 = chainer.Variable(c_prev1_data)
        c_prev2 = chainer.Variable(c_prev2_data)

        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)

        c, h = functions.slstm(c_prev1, c_prev2, x1, x2)


        # Compute expected out
        a1_in = self.x1[:, [0, 4]]
        i1_in = self.x1[:, [1, 5]]
        f1_in = self.x1[:, [2, 6]]
        o1_in = self.x1[:, [3, 7]]
        a2_in = self.x2[:, [0, 4]]
        i2_in = self.x2[:, [1, 5]]
        f2_in = self.x2[:, [2, 6]]
        o2_in = self.x2[:, [3, 7]]

        c_expect = _sigmoid(i1_in) * numpy.tanh(a1_in) + \
                   _sigmoid(i2_in) * numpy.tanh(a2_in) + \
                   _sigmoid(f1_in) * self.c_prev1 + \
                   _sigmoid(f2_in) * self.c_prev2
        h_expect = _sigmoid(o1_in + o2_in) * numpy.tanh(c_expect)

        print("state      = ",numpy.allclose(c_expect, c.data))
        print("hidden     = ",numpy.allclose(h_expect, h.data))


    def test_forward_cpu(self):
        self.check_forward(self.c_prev1, self.c_prev2, self.x1, self.x2)

    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()



if __name__ == "__main__":
    test = TestSLSTM()
    test.batch = 3
    test.dtype = numpy.float32

    test.setUp()
    print("Testing forward")
    test.test_forward_cpu()

    print("Testing flat forward")
    test.test_flat_forward_cpu()