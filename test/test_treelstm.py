import chainer
import numpy
from chainer import functions as F
from chainer import variable
from chainer import link
import chainer.links as L
from chainer import initializers

from memory_cell.bn_lstm import BNLSTM
from memory_cell.treelstm import FastTreeLSTM

if __name__ == "__main__":
    x = chainer.Variable(numpy.ones((1, 10), dtype=numpy.float32))
    treelstm2 = FastTreeLSTM(lateral_init=1, upward_init=1, bias_init=0, forget_bias_init=0, in_size=10, out_size=10, children=1)
    # gru = L.StatefulGRU(bias_init=0,inner_init=1,in_size=10, out_size=10)
    lstm1 = L.LSTM(lateral_init=1, upward_init=1,bias_init=0, forget_bias_init=0, in_size=10, out_size=10)

    y2c1, y2h1 = treelstm2(None,None,x)
    # y2c2, y2h2 = treelstm2(None,None,x)
    # y2 = treelstm2(F.concat((y2c1,y2c2),axis=0), F.concat((y2h1,y2h2),axis=0),x)[1]
    y2 = treelstm2(y2c1,y2h1,x)[1]

    y3 = lstm1(x)
    y3 = lstm1(x)

    print(y2.data)
    print(y3.data)

    print(y2.data.shape)
    print(y3.data.shape)

    print(numpy.allclose(y2.data, y3.data))
