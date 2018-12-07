import numpy as np
import theano
import theano.tensor as T
#import lasagne
from lasagne.nonlinearities import identity, tanh
from lasagne.init import GlorotUniform, GlorotNormal, Uniform, Constant, Orthogonal
from lasagne.regularization import l2, apply_penalty
from lasagne.layers import Layer, MergeLayer, Gate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX
alpha_regf = lambda a: (0.5 * T.log1p(T.exp(-a)) - (0.03 + 1.0 / (1.0 + T.exp(-(1.5 * (a + 1.3)))) * 0.64))
hard_sigmoid = lambda x: T.nnet.hard_sigmoid(x)
alpha_regf_np = lambda a: (0.5 * np.log1p(np.exp(-a)) - (0.03 + 1.0 / (1.0 + np.exp(-(1.5 * (a + 1.3)))) * 0.64))

def alloc_zeros_matrix(*dims):
            return T.alloc(np.cast[floatX](0.), *dims)      
    
class Embedding(Layer):
    def __init__(self, incoming, input_size, output_size, W=Uniform(range=0.05), **kwargs):
        super(Embedding, self).__init__(incoming, **kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self._srng = RandomStreams(seed=np.random.randint(10e6))
        self.name = 'Embedding'
        
        self.W = self.add_param(W, (input_size, output_size), name="W")

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input] 
    
class BayesianEmbedding(Embedding):
    def __init__(self, incoming, input_size, output_size, log_sigma_init=-3.0, config="DC", **kwargs):
        Embedding.__init__(self, incoming, input_size, output_size, \
                           **kwargs)
        self.reg = True
        self.name = 'BayesianEmbedding'

        self.config = config
        if self.config[0] in {"L", "N"}:
            self.log_sigma_w = self.add_param(Constant(log_sigma_init), (self.input_size, self.output_size), name="log_sigma_w")
        if self.config[1] in {"L", "N"}:
            self.mu_in = self.add_param(Constant(1.0), shape=(input_size,), 
                                 name="mu_in", regularizable=False, trainable=True)
            self.log_sigma_in = self.add_param(Constant(log_sigma_init), shape=(input_size,), name="log_in", regularizable=False)
        self.thresh = 3.0
        
    def clip_func(self, mtx, to=8):
        mtx = T.switch(T.le(mtx, -to), -to, mtx)
        mtx = T.switch(T.ge(mtx, to), to, mtx)
        return mtx
    
    def generate_noise_and_clip(self, num_batch, \
                                    deterministic = False,\
                                    clip=False):
        if not deterministic:
            if self.config[0] in {"L", "N"}:
                self.w_noise = self._srng.normal((self.input_size, self.output_size), avg=0, std=1, dtype=floatX) * T.exp(self.log_sigma_w)
            else: # D
                self.w_noise = T.zeros(1, dtype=floatX)
            if self.config[1] in {"L", "N"}:
                self.in_noise = self._srng.normal((num_batch, self.input_size), avg=0, std=1, dtype=floatX) * T.exp(self.log_sigma_in) + self.mu_in
            else: # C
                self.in_noise = T.ones(1, dtype=floatX)
        else:
            self.w_noise = T.zeros(1, dtype=floatX)
            self.in_noise = T.ones(1, dtype=floatX) if self.config[1] not in {"L", "N"}\
                            else self.mu_in
                            
        if clip:
            if self.config[0] == "L":
                log_alpha_w = self.clip_func(2*self.log_sigma_w-T.log(self.W**2))
                self.w_clip = T.le(log_alpha_w, self.thresh)
            else:
                self.w_clip = T.ones(1, dtype=floatX)
            if self.config[1] == "L":
                log_alpha_in = self.clip_func(2*self.log_sigma_in-T.log(self.mu_in**2))
                self.in_clip = T.le(log_alpha_in, self.thresh)
            else:
                self.in_clip = T.ones(1, dtype=floatX)
        else:
            self.w_clip = T.ones(1, dtype=floatX)
            self.in_clip = T.ones(1, dtype=floatX)
    
    def get_output_for(self, input, deterministic=False, clip=False, **kwargs):
        self.generate_noise_and_clip(input.shape[0], deterministic, clip, **kwargs)
        shape = (input.shape[0], input.shape[1])
        if (not deterministic) and self.config[1] in {"N", "L"}:
            return ((self.W+self.w_noise)*self.w_clip)[input] *\
               (self.in_noise*self.in_clip)\
               [T.extra_ops.repeat(T.arange(input.shape[0]),\
                                            input.shape[1]), input.ravel()]\
                                            .reshape(shape)[:, :, None]
        else:
            return ((self.W+self.w_noise)*self.w_clip*\
                   (self.in_noise*self.in_clip)[:, None])[input] 

    def eval_reg(self, train_size, **kwargs):
        if self.config[0] == "L":
            log_alpha_w = self.clip_func(2*self.log_sigma_w-T.log(self.W**2))
            KL = alpha_regf(log_alpha_w).sum()
        elif self.config[0] == "N":
            KL_element_w = - self.log_sigma_w + 0.5 * (T.exp(2*self.log_sigma_w) + self.W**2) - 0.5
            KL = T.sum(KL_element_w)
        else:
            KL = T.zeros(1, dtype=floatX).sum()
        
        if self.config[1] == "L":
            log_alpha_in = self.clip_func(2*self.log_sigma_in-T.log(self.mu_in**2))
            KL += alpha_regf(log_alpha_in).sum()
        elif self.config[1] == "N":
            KL_element_in = - self.log_sigma_in + 0.5 * (T.exp(2*self.log_sigma_in) + self.mu_in**2) - 0.5
            KL += T.sum(KL_element_in)
        return KL/train_size
    
    def get_ard(self, thresh=1.0, **kwargs):  
        if self.config[0] == "L":
            log_alpha_w = 2*self.log_sigma_w.get_value()-np.log(self.W.get_value()**2)
            mask_w = log_alpha_w < self.thresh
        else:
            mask_w = np.ones_like(self.W.get_value())
        if self.config[1] == "L":
            log_alpha_in = 2*self.log_sigma_in.get_value()-np.log(self.mu_in.get_value()**2)
            mask_in = log_alpha_in < self.thresh
        else:
            mask_in = mask_w.any(axis=1)
        mask_out = mask_w.any(axis=0)
        return {"w": mask_w, "z_voc":mask_in, "z_emb":mask_out}

    def get_reg(self):
        if self.config[0] == "L":
            log_alpha_w = 2*self.log_sigma_w.get_value()-np.log(self.W.get_value()**2)
            KL_w = alpha_regf_np(log_alpha_w).sum()
        elif self.config[0] == "N":
            KL_element_w = - self.log_sigma_w.get_value() + 0.5 * (np.exp(2*self.log_sigma_w.get_value()) + self.W.get_value()*2) - 0.5
            KL_w = np.sum(KL_element_w)
        else:
            KL_w = 0
        
        if self.config[1] == "L":
            log_alpha_in = 2*self.log_sigma_in.get_value()-np.log(self.mu_in.get_value()**2)
            KL_in = alpha_regf_np(log_alpha_in).sum()
        elif self.config[1] == "N":
            KL_element_in = - self.log_sigma_in.get_value + 0.5 * (np.exp(2*self.log_sigma_in.get_value()) + self.mu_in.get_value()**2) - 0.5
            KL_in = np.sum(KL_element_in)
        else:
            KL_in = 0
        
        return "%.4f, %.4f" % (KL_w, KL_in)