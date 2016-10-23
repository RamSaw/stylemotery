from chainer import optimizers

from models.tree_models import RecursiveTreeLSTM
from utils.fun_utils import get_basefolder, parse_src_files, print_model, generate_trees, make_backward_graph
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import variable
import numpy as np

class MultiLayerPerceptron(chainer.Chain):

    def __init__(self, n_in, n_hidden, n_out):
        # Create and register three layers for this MLP
        super(MultiLayerPerceptron, self).__init__(
            layer1=L.Linear(n_in, n_hidden),
            layer2=L.Linear(n_hidden, n_hidden),
            layer3=L.Linear(n_hidden, n_out),
        )

    def __call__(self, x):
        # Forward propagation
        h1 = F.relu(self.layer1(x))
        print(h1.data.shape)
        o = F.expand_dims(h1,axis=0)
        print(o.data.shape)
        # o = F.reshape(h1.data,h1.data.shape)#variable.Variable(h1.data,volatile=False)
        h2 = F.relu(self.layer2(o[0]))
        return self.layer3(h2)

input1 = variable.Variable(np.random.randn(30,10).astype(dtype=np.float32))
input2 = variable.Variable(np.random.randn(30,10).astype(dtype=np.float32))
input3 = variable.Variable(np.random.randn(30,10).astype(dtype=np.float32))
input4 = variable.Variable(np.random.randn(30,10).astype(dtype=np.float32))

input = F.concat((F.expand_dims(input1,axis=0),F.expand_dims(input2,axis=0)),axis=0)
print(input.data.shape)

loss = F.batch_matmul(input,input,transa=False, transb=True)

print(loss.data.shape)
# batch_loss += model(input)
# batch_loss.backward()
make_backward_graph(R"C:\Users\bms\PycharmProjects" ,"treelstm" ,[loss])
