import numpy as np
import numpy
import six
from chainer import cuda
from chainer import function
from chainer import link
from chainer import variable
from chainer.links.connection import linear
from chainer.utils import type_check
from chainer import initializers


# def _extract_gates(x, parts=4):
#     r = x.reshape((len(x), x.shape[1] // parts, parts) + x.shape[2:])
#     return [r[:, :, i] for i in range(parts)]

def _extract_gates(x, parts=4):
    pivot = len(x) // parts
    return [x[i:i+pivot] for i in range(0,len(x),pivot)]

def _extract_allgates(x, parts=4):
    r = x.reshape((len(x), x.shape[1] // parts, parts) + x.shape[2:])
    all_gates = [r[:, :, i] for i in range(3)]
    f_gates = [r[:, :, i] for i in range(3, parts)]
    return all_gates, f_gates


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class TreeLSTMFunction(function.Function):
    def __init__(self, children=1, in_dim=100, out_dim=200):
        self.n_children = children

    def forward_cpu(self, inputs):
        c_prev, x = inputs

        gates = _extract_gates(x,self.n_children+3)
        (a, i, o) = gates[:3]
        f = gates[3:]

        c = _extract_gates(c_prev,self.n_children)

        self.i = _sigmoid(i)
        self.f = [_sigmoid(fs) for fs in f]
        self.o = _sigmoid(o)
        self.a = numpy.tanh(a)

        self.c = self.a * self.i + np.sum([np.dot(x,y) for x,y in zip(self.f, c)])
        h = self.o * numpy.tanh(self.c)

        return self.c, h

    def forward_gpu(self, inputs):
        c_prev, x = inputs
        a, i, f, o = _extract_gates(x)

        self.c, h = cuda.elementwise(
            'T c_prev, T a, T i_, T f, T o', 'T c, T h',
            '''
                COMMON_ROUTINE;
                c = aa * ai + af * c_prev;
                h = ao * tanh(c);
            ''',
            'lstm_fwd', preamble=_preamble)(c_prev, a, i, f, o)

        return self.c, h

    def backward_cpu(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x = inputs
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        co = numpy.tanh(self.c)
        gc_prev = gh * self.o * _grad_tanh(co) + gc  # multiply f later
        ga[:] = gc_prev * self.i * _grad_tanh(self.a)
        gi[:] = gc_prev * self.a * _grad_sigmoid(self.i)
        gf[:] = gc_prev * c_prev * _grad_sigmoid(self.f)
        go[:] = gh * co * _grad_sigmoid(self.o)
        gc_prev *= self.f  # multiply f here

        return gc_prev, gx

    def backward_gpu(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x = inputs
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        a, i, f, o = _extract_gates(x)
        gc_prev = xp.empty_like(c_prev)
        cuda.elementwise(
            'T c_prev, T c, T gc, T gh, T a, T i_, T f, T o',
            'T gc_prev, T ga, T gi, T gf, T go',
            '''
                COMMON_ROUTINE;
                T co = tanh(c);
                T temp = gh * ao * grad_tanh(co) + gc;
                ga = temp * ai * grad_tanh(aa);
                gi = temp * aa * grad_sigmoid(ai);
                gf = temp * c_prev * grad_sigmoid(af);
                go = gh * co * grad_sigmoid(ao);
                gc_prev = temp * af;
            ''',
            'lstm_bwd', preamble=_preamble)(
            c_prev, self.c, gc, gh, a, i, f, o,
            gc_prev, ga, gi, gf, go)

        return gc_prev, gx
