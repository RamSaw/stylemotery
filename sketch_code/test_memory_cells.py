


import chainer.links as L
import numpy as np
from chainer.links.connection import linear
import chainer.functions as F
def extract_gates(x):
    r = F.reshape(x,(len(x.data), x.data.shape[1] // 4, 4) + x.data.shape[2:])
    return [r[:, :, i] for i in range(4)]

if __name__ == "__main__":
    dim = 4
    a1 = linear.Linear(dim,4*dim)
    a, i, f, o = extract_gates(a1.W)

    print(a1.W.data.shape)
    print(a.data.shape)
    print(i.data.shape)
    print(f.data.shape)
    print(o.data.shape)

    print(F.concat([a.data.reshape(1,1,16) for i in range(6)],axis=0).shape)
    print(F.matmul(a,i,transa=False,transb=True).shape)
    print(F.batch_matmul(F.concat((a,a)),F.concat((i,i)),transa=False,transb=True).shape)