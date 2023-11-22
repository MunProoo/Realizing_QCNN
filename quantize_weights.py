# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf

from tensorflow.python.framework import tensor_shape, ops
from tensorflow.python.ops import standard_ops, nn, variable_scope, math_ops, control_flow_ops
from tensorflow.python.eager import context
from tensorflow.python.training import optimizer, training_ops
import keras.backend as K

import numpy as np

import theano.tensor as T

import pandas as pd



def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded-x)

def binary_sigmoid_unit(x):
    return round_through(hard_sigmoid(x))

def clip_through(x, min_val, max_val):
    '''Element-wise clipping with gradient propagation
    Analogue to round_through
    '''
    clipped = K.clip(x, min_val, max_val)
    clipped_through= x + K.stop_gradient(clipped-x)
    return clipped_through

def clip_through(x, min, max):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,min,max)
    return x + K.stop_gradient(clipped - x)

def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    return K.clip((x+1)/2, 0, 1)

def binary_tanh_unit(x):
    return 2.*round_through(_hard_sigmoid(x))-1.

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

def map(W, a, b, min, max):
    # a------W--b => min------------W----max // 선형 변환
    #W = -(min*(W-b) + max*(a-W))/(a-b)
    W = (W-a)*(max-min)/(b-a) + min
    return W









def quantize_weights(W, method = 'real', H=1, nb = 2, fixed_max = 0):

    if method == "real":
        return W
    elif method == 'binary':
        return binarization(W, H)

    elif method == 'radix':
        return radix(W, nb)

    elif method == 'radix_ReLU':
        return radix_ReLu(W, nb)

    elif method == 'quantize':
        return quantize(W, nb)

    elif method == 'quantized_relu':
        return quantized_relu(W, nb)

    elif method == 'quantized_tanh':
        return quantized_tanh(W, nb)

    elif method == 'radix_ReLU_fixed':
        return radix_ReLu_fixed(W, nb, fixed_max)










###############################################################################################quantize
def radix(W, nb):
    #W = tf.Print(W, [W], summarize=5000)

    #W = K.clip(W, -0.1, 0.1)

    #clip_through(W, -1.5, 1.5)

    from_max = tf.reduce_max(W)
    from_min = tf.reduce_min(W)

    #nb=round_through(from_max-from_min) #가중치 정수화만



    W = map(W, from_min, from_max, -nb, nb)  ##good

    #W = K.clip(round_through(W), -nb, nb)

    #W = tf.Print(W, [W], summarize=10)

    #W = round_through(W)


    #W = tf.floormod(abs(W),nb)*(W/(abs(W)+0.000001))# - (round_through((nb-abs(W))/((nb-abs(W))+0.000001))-1)*W##############이거 사용하면 nan 값이 왜 많이 생기지.......................
    out = round_through(W)

    #out = map(out, nb * from_min, nb * from_max, -nb, nb) ##너무 커진 가중치 범위 다시 줄이기
    #out = round_through(out)


    # a = tf.div(from_max, 4)
    #
    # W = tf.div(W, a)
    # out = W
    return out

def radix_ReLu(W, nb):
    #W = tf.Print(W, [W], summarize=2000)


    W = tf.nn.relu(W)

    #W = K.clip(W,0,80)




    from_max = tf.reduce_max(W)
    #from_max = tf.Print(from_max, [from_max], summarize=1)
    from_min = tf.reduce_min(W)
    #nb=round_through(from_max-from_min) #가중치 정수화만
    #from_max = tf.Print(from_max, [from_max], summarize=1)
    W = map(W, from_min, from_max, 0, nb*2)#good

    #W = map(W, 0, from_max, 0, from_max)

    #W = map(W, from_min, from_max, nb * from_min, nb * from_max)


    #W = K.clip(round_through(W), 0, nb*2)

    #W = tf.Print(W, [W], summarize=10000)

    #W = round_through(W)


    #W = tf.floormod(abs(W),nb)*(W/(abs(W)+0.000001))# - (round_through((nb-abs(W))/((nb-abs(W))+0.000001))-1)*W##############이거 사용하면 nan 값이 왜 많이 생기지.......................
    out = round_through(W)

    #out = map(out, nb * from_min, nb * from_max, -nb, nb) ##너무 커진 가중치 범위 다시 줄이기
    #out = round_through(out)

    #tf.div : 나눗셈의 몫
    # a = tf.div(from_max, 4)
    #
    # W = tf.div(W, a)

    #out = W
    return out


def quantize(W, nb = 16, clip_throughs=False):

    '''The weights' binarization function,

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''

    max = tf.reduce_max(W)
    min = tf.reduce_min(W)

    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    if clip_throughs:
        Wq = clip_through(round_through(W*m),-m,m-1)/m
    else:
        Wq = K.clip(round_through(W*m), -m, m)/m  #+(1/(m*2)) #+(1/(m*2))는 가중치들의 값에 0없이 벨런스 맞추기
        #Wq = tf.Print(Wq, [Wq], summarize=20)

    #Wq = tf.Print(Wq,[Wq],summarize=20000)


    return Wq


def quantized_relu(W, nb=16):

    '''The weights' binarization function,

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    #non_sign_bits = nb-1
    #m = pow(2,non_sign_bits)
    #Wq = K.clip(round_through(W*m),0,m-1)/m

    nb_bits = nb
    Wq = K.clip(2. * (round_through(_hard_sigmoid(W) * pow(2, nb_bits)) / pow(2, nb_bits)) - 1., 0,
                1 - 1.0 / pow(2, nb_bits - 1))
    return Wq


def quantized_tanh(W, nb=16):

    '''The weights' binarization function,

    # Reference:
    - [QuantizedNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}

    '''
    non_sign_bits = nb-1
    m = pow(2,non_sign_bits)
    #W = tf.Print(W,[W],summarize=20)
    Wq = K.clip(round_through(W*m),-m,m-1)/m
    #Wq = tf.Print(Wq,[Wq],summarize=20)
    return Wq
################################################################################################################

################################################################################################################binarize
def binarization(W, H=1, binary=True, deterministic=True, stochastic=False, srng=None):
    max = tf.reduce_max(W)
    min = tf.reduce_min(W)

    W = map(W, min, max, 0, 1)

    Wb = (round_through(W)-0.5)*2

    #Wb = tf.Print(Wb,[Wb],summarize=1000)




    #Wb = H * binary_tanh_unit(W / H)


    return Wb











def radix_ReLu_fixed(W, nb, fixed_max):

    W = tf.nn.relu(W)


    # fixed_max = tf.Print(fixed_max,[fixed_max],summarize=1)


    W = map(W, 0, fixed_max, 0, nb*2)#good


    # a = tf.div(fixed_max, 4)
    #
    # W = tf.div(W, a)


    W = round_through(W)


    #W = tf.Print(W, [W], summarize=100)
    out = W
    return out
