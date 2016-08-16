import numpy as np, chainer, chainer.functions as F
from chainer import variable
from chainer import links as L
import random

x = variable.Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),volatile="off")
print(x.data)
gru = L.StatefulGRU(3, 2)
linear = L.Linear(3, 3)
# y = x ** 2 - 2 * x + 1  # gru(x) * gru(x) * gru(x)
y =linear(x)

y.grad = np.ones((2, 3), dtype=np.float32)
# y = variable.Variable(y.data * gru(x).data,volatile="off")

y.backward(retain_grad=True)

print(y.grad)
print(x.grad)
# print(y.data)
# print(y.grad)
