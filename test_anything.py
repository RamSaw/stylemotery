import numpy, chainer, chainer.functions as F
from chainer import links as L
x = chainer.Variable(numpy.ones((3,10),dtype=numpy.float32))
f = F.Identity()
gru = L.StatefulGRU(10,10)
y = gru(x)
y += gru(x)

y.backward()

print(y.data)