import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy
import six

import chainer
from chainer.functions.activation import lstm
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable


gates = 4
def _extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // gates, gates) + x.shape[2:])
    return [r[:, :, i] for i in six.moves.range(gates)]

in_size = 60
out_size = 20
x=variable.Variable(numpy.ones((100,in_size),dtype=numpy.float32))
u=linear.Linear(in_size, gates * out_size, initialW=0)
print(u.W.data.shape)
print(u.W.data.T.shape)
print(x.data.shape)
print(u.debug_print())
print(x.debug_print())
t = _extract_gates(u(x).data)
#self.upward.b.data.reshape(1, 4 * out_size, 1)
for i in t:
    print(i.shape)