import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np


def _extract_gates(x):
    r = F.reshape(x,(len(x.data), x.data.shape[1] // 4, 4) + x.data.shape[2:])
    return [r[:, :, i] for i in range(4)]

embed = L.EmbedID(10, 10)
lstm = L.LSTM(5,2)

word1 = np.array([1], np.int32)
word2 = np.array([2], np.int32)
w1 = chainer.Variable(word1, volatile="off")
w2 = chainer.Variable(word2, volatile="off")
# print(.data.shape)
e = embed(F.concat((w1,w2),axis=0))
print(e.data.shape)
# lstm_e = lstm(e)
# print(lstm_e.data.shape)
w1 = chainer.Variable(np.array([1,2,3], np.float32), volatile="off")
w2 = chainer.Variable(np.array([10,20,30], np.float32), volatile="off")
print(w1 * w2)

v = chainer.Variable(np.random.randn(100,400), volatile="off")
splits = _extract_gates(v)#F.split_axis(v,100,axis=0)
print(v.data.shape)
print(len(splits))