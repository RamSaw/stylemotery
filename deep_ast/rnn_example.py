



import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
embed = L.EmbedID(20, 40)


print(embed(np.array([1])).data)