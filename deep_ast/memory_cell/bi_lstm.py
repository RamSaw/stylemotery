import chainer
import chainer.functions as F
import chainer.links as L


class BRNN(chainer.Chain):
    """
    refer to Alex Graves, Navdeep Jaitly. Towards End-to-End Speech Recognition with Recurrent Neural Networks

    """
    def __init__(self, n_infeature, n_inner, n_outfeature, train=True):
        super(BRNN, self).__init__(
            f_unit=L.LSTM(n_infeature, n_inner),  # forward unit
            b_unit=L.LSTM(n_infeature, n_inner),  # backward unit
            fo_unit=L.Linear(n_inner, n_outfeature),  # forward output unit
            bo_unit=L.Linear(n_inner, n_outfeature),  # backward output unit
        )
        self.train = train

    def reset_state(self):
        self.f_unit.reset_state()
        self.b_unit.reset_state()

    def __call__(self, input_seq):  # input_seq is a sequence of Variable
        seq_length = len(input_seq)

        f_layer = [self.f_unit(x) for x in input_seq]  # forward layer
        b_layer = [self.b_unit(input_seq[i]) for i in range(seq_length-1, -1, -1)]  # backward layer

        fo_layer = [self.fo_unit(x) for x in f_layer]
        bo_layer = [self.bo_unit(b_layer[i]) for i in range(seq_length-1, -1, -1)]

        return [F.dropout(fo_layer[i]+bo_layer[i], train=self.train) for i in range(seq_length)]