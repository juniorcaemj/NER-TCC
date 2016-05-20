from __future__ import absolute_import

import numpy as np
import theano
import theano.tensor as T
from keras import initializations, regularizers, constraints

from keras import backend as K
from keras.engine import Layer


class FixedEmbeddingWithMask1(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size.
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    input_ndim = 2

    def __init__(self, input_dim, output_dim, init='uniform', input_length=None, W_regularizer=None,
                 activity_regularizer=None, W_constraint=None, mask_zero=False, weights=None, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(FixedEmbeddingWithMask1, self).__init__(**kwargs)

    def build(self, input_shape):
        #self.input = T.imatrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.params = [] #No update of the weight
        self.regularizers = []
        #if self.W_regularizer:
        #    self.W_regularizer.set_param(self.W)
        #    self.regularizers.append(self.W_regularizer)

        #if self.activity_regularizer:
        #    self.activity_regularizer.set_layer(self)
        #    self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            #self.set_weights(self.initial_weights)
            self.W.set_value(self.initial_weights[0])
            #self.W = self.initial_weights[0]
			
    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 1)

    '''def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None
        else:
            return T.ones_like(X) * (1 - T.eq(X, 1))'''

    def get_output_shape_for(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return (input_shape[0], input_length, self.output_dim)

    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            B = K.random_binomial((self.input_dim,), p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        out = K.gather(W, x)
        return out

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_length, self.output_dim)

    '''def get_output(self, train=False):
        X = self.get_input(train)
        out = self.W[X]
        return out'''

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "input_length": self.input_length,
                  "mask_zero": self.mask_zero,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None,
				  'dropout': self.dropout}
        base_config = super(FixedEmbeddingWithMask1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def shared_zeros(shape, dtype=theano.config.floatX, name='', n=1):
    shape = shape if n == 1 else (n,) + shape
    return theano.shared(np.zeros(shape, dtype=dtype), name=name)
def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)
def floatX(val):
    return np.asarray(val, dtype=theano.config.floatX)
