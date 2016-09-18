import math

from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import links as L
from chainer import functions as F
import numpy as np


def numpy_extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in range(4)]


in_size = 10
out_size = 4
children = 1
bias_init = 0
forget_bias_init = 0

x1 = 1 * np.ones((1, in_size))
x2 = 2 * np.ones((1, in_size))
x3 = 3 * np.ones((1, in_size))
x4 = 4 * np.ones((1, in_size))
x = np.concatenate((x1, x2, x3, x4), axis=0)
# test lstm
F.concat
# upward = L.Linear(in_size, 4 * out_size, initialW=0),
# lateral = L.Linear(children * out_size, 3 * children * out_size, initialW=0, nobias=True)
# a, i, f, o = numpy_extract_gates(upward.b.data.reshape(1, 4 * out_size, 1))
# initializers.init_weight(a, bias_init)
# initializers.init_weight(i, bias_init)
# initializers.init_weight(f, forget_bias_init)
# initializers.init_weight(o, bias_init)
gi = np.eye(in_size, out_size)
gf = np.eye(in_size, out_size)
go = np.eye(in_size, out_size)
gu = np.eye(in_size, out_size)
gates = np.concatenate((gi, gf, go, gu), axis=1)

out = x.dot(gates)
print(out)
print(numpy_extract_gates(out))
