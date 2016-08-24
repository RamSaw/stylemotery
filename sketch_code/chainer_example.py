import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np

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