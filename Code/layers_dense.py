import numpy as np
import theano
import theano.tensor as T
from lasagne.nonlinearities import identity, tanh
from lasagne.init import GlorotUniform, GlorotNormal, Uniform, Constant, Orthogonal, Normal
from lasagne.regularization import l2, apply_penalty
from lasagne.layers import Layer, MergeLayer, Gate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX
alpha_regf = lambda a: (0.5 * T.log1p(T.exp(-a)) - (0.03 + 1.0 / (1.0 + T.exp(-(1.5 * (a + 1.3)))) * 0.64))
hard_sigmoid = lambda x: T.nnet.hard_sigmoid(x)

def alloc_zeros_matrix(*dims):
            return T.alloc(np.cast[floatX](0.), *dims)

class Dense(Layer):
    def __init__(self, incoming, num_units, W=GlorotUniform(), b=Constant(0.), nonlinearity=identity, **kwargs):
        super(Dense, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        self.num_units = num_units
        self.num_inputs = int(self.input_shape[-1])
        self._srng = RandomStreams(seed=np.random.randint(10e6))
        self.name = 'Dense'
        
        self.W = self.add_param(W, (self.num_inputs, self.num_units), name="W")
        self.b = self.add_param(b, (self.num_units,), name="b")#,regularizable=False)
    
    def pre_activation(self, input, **kwargs):
        return T.dot(input, self.W)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        return self.nonlinearity(self.pre_activation(input, **kwargs) + self.b)
    
class BayesianDense(Dense):
    def __init__(self, incoming, num_units, log_sigma2_init = Constant(-3.), W=GlorotUniform(),
                 b=Constant(0.), nonlinearity=identity, \
                 train_sigma=True, **kwargs):
        super(BayesianDense, self).__init__(incoming, num_units, W, b, nonlinearity, **kwargs)
        self.log_sigma2 = self.add_param(log_sigma2_init, (self.num_inputs, self.num_units), name="log_sigma2")
        self.reg = True
        self.name = 'DenseSparseVDO'
        self.thresh = 3.

    def clip(self, mtx, to=8):
        mtx = T.switch(T.le(mtx, -to), -to, mtx)
        mtx = T.switch(T.ge(mtx, to), to, mtx)
        return mtx
    
    def pre_activation(self, input, deterministic = False, clip_test = False, clip_train = False):
        if deterministic:
            if clip_test:
                log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
                clip_mask = T.ge(log_alpha, self.thresh)
                return T.dot(input, T.switch(clip_mask, 0, self.W))
            else:
                return T.dot(input, self.W)
        W = self.W
        sigma2 = T.exp(self.log_sigma2)
        if clip_train:
            log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
            clip_mask = T.ge(log_alpha, self.thresh)
            W = T.switch(clip_mask, 0, self.W)  
            sigma2 = T.switch(clip_mask, 0, T.exp(self.log_sigma2))
        mu = T.dot(input, W)
        si = T.sqrt(T.dot(input * input, sigma2)+1e-8)
        if input.ndim == 2:
            return mu + self._srng.normal(mu.shape, avg=0, std=1, dtype=floatX) * si
        else:
            return mu + self._srng.normal((mu.shape[0],1,mu.shape[2]), avg=0, std=1, dtype=floatX) * si
    
    def eval_reg(self, train_size):
        log_alpha = self.clip(self.log_sigma2 - T.log(self.W ** 2))
        return alpha_regf(log_alpha).sum() / train_size
    
    def get_ard(self):
        log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
        return {"w": (log_alpha < self.thresh)}
    
    def get_reg(self):
        log_alpha = self.log_sigma2.get_value() - 2 * np.log(np.abs(self.W.get_value()))
        return '%.1f, %.1f' % (log_alpha.min(), log_alpha.max())
  
