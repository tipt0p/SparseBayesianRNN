import numpy as np
import theano
import theano.tensor as T
#import lasagne
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
    
class LSTM(MergeLayer):
    def __init__(self, incoming, num_units,
                 ingate=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(0.),nonlinearity=hard_sigmoid),
                 forgetgate=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(1.),nonlinearity=hard_sigmoid),
                 cell=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(0.),nonlinearity=tanh),
                 outgate=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(0.),nonlinearity=hard_sigmoid),
                 hid_init=Constant(0.),
                 cell_init=Constant(0.),
                 learn_init=True,
                 nonlinearity=tanh,
                 backwards=False,
                 gradient_steps=-1,
                 mask_input=None,
                 only_return_final=False,
                 hid_prop = False,
                 **kwargs):
        incomings = incoming if hid_prop else [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
 
        super(LSTM, self).__init__(incomings, **kwargs)
 
        self.name = 'LSTM'
        self.nonlinearity = nonlinearity
        self.num_units = num_units
        self.num_inputs = np.prod(self.input_shapes[0][2:])
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.only_return_final = only_return_final
        self._srng = RandomStreams(seed=np.random.randint(10e6))
        self.hidden_noise = T.ones(1, dtype=floatX)
        self.hidden_clip = T.ones(1, dtype=floatX)
        self.mu_hid = T.ones(1, dtype=floatX)
        self.log_sigma2_hid = T.ones(1, dtype=floatX)
        self.learn_init = learn_init
        self.hid_prop = hid_prop
 
        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
           instance. """
            return (self.add_param(gate.W_in, (self.num_inputs, self.num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (self.num_units, self.num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (self.num_units,),
                                   name="b_{}".format(gate_name)),
                                   #regularizable=False),
                    gate.nonlinearity)
 
        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
 
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')
 
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')
 
        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')
    
        self.hid_init = self.add_param(hid_init, (1,self.num_units),
                name="hid_init", trainable=learn_init, regularizable=False)
        self.cell_init = self.add_param(cell_init, (1,self.num_units),
                name="cell_init", trainable=learn_init, regularizable=False)
 
    def generate_noise_and_clip(self, num_batch, seq_len, **kwargs):
        return
    
    def input_preactivation(self, input, gate_type, **kwargs):
        if gate_type == 'input':
            return T.dot(input, self.W_in_to_ingate)
        if gate_type == 'forget':
            return T.dot(input, self.W_in_to_forgetgate)
        if gate_type == 'cell':
            return T.dot(input, self.W_in_to_cell)
        if gate_type == 'output':
            return T.dot(input, self.W_in_to_outgate)
        
    def hidden_preactivation(self, hidden, gate_type, **kwargs):
        if gate_type == 'input':
            return T.dot(hidden, self.W_hid_to_ingate)
        if gate_type == 'forget':
            return T.dot(hidden, self.W_hid_to_forgetgate)
        if gate_type == 'cell':
            return T.dot(hidden, self.W_hid_to_cell)
        if gate_type == 'output':
            return T.dot(hidden, self.W_hid_to_outgate)
    
    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        if self.only_return_final:
            return input_shape[0], self.num_units
        elif self.hid_prop:
            return 2, input_shape[0], input_shape[1], self.num_units
        else:
            return input_shape[0], input_shape[1], self.num_units 
    
    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
 
        input = input.dimshuffle(1, 0, 2)
        num_batch = input.shape[1]
        seq_len = input.shape[0]
        self.generate_noise_and_clip(num_batch, seq_len, **kwargs)
        
        input_i = self.input_preactivation(input, 'input', **kwargs) + self.b_ingate
        input_f = self.input_preactivation(input, 'forget', **kwargs) + self.b_forgetgate
        input_c = self.input_preactivation(input, 'cell', **kwargs) + self.b_cell
        input_o = self.input_preactivation(input, 'output', **kwargs) + self.b_outgate
 
        def step(input_n_i, input_n_f, input_n_c, input_n_o, cell_previous, hid_previous, *args):
            ingate = self.nonlinearity_ingate(input_n_i + self.hidden_preactivation(hid_previous, 'input', **kwargs))
            forgetgate = self.nonlinearity_forgetgate(input_n_f + self.hidden_preactivation(hid_previous, 'forget', **kwargs))
            cell = forgetgate * cell_previous + ingate * self.nonlinearity_cell(
                input_n_c + self.hidden_preactivation(hid_previous, 'cell', **kwargs))
            outgate = self.nonlinearity_outgate(input_n_o + self.hidden_preactivation(hid_previous, 'output', **kwargs))
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]
 
        def step_masked(input_n_i, input_n_f, input_n_c, input_n_o, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n_i, input_n_f, input_n_c, input_n_o, cell_previous, hid_previous, *args)
 
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]
 
        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input_i, input_f, input_c, input_o, mask]
            step_fun = step_masked
        else:
            sequences = [input_i, input_f, input_c, input_o]
            step_fun = step
        
        non_seqs = [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate, self.hidden_noise, self.hidden_clip,
             self.mu_hid, self.log_sigma2_hid]
        
        if self.hid_prop:
            hid_init, cell_init = inputs[1][0,:,:], inputs[1][1,:,:]
        else:
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)
            cell_init = T.dot(T.ones((num_batch, 1)), self.cell_init)
        
        cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[cell_init, hid_init],
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]
 
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)
            cell_out = cell_out.dimshuffle(1, 0, 2)
 
            if self.backwards:
                hid_out = hid_out[:, ::-1]
            
        if self.hid_prop:
            return T.concatenate([hid_out[np.newaxis,:,:,:],cell_out[np.newaxis,:,:,:]])
        else:
            return hid_out

class BayesianLSTM(LSTM):
    """
    config: L probabilistic weight with lognormal prior, N probabilistic weight with standart normal prior, 
           D deterministic learnable weight, 
           C constant weight (1 for multiplicative and 0 for additive weights)\
           
           config[0]: W input_to_hidden, W hidden_to_hidden
                     L N D (C is not supported)
           config[1]: hat Z preactivation multiplicative weights
                     L N D C
           config[2]: Z input and hidden multiplicative weights
                     L N D C I R
    """
    def __init__(self, incoming, num_units,
                 log_sigma_in_init = -3., log_sigma_hid_init = -3.,
                 ingate=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(0.),nonlinearity=hard_sigmoid),
                 forgetgate=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(1.),nonlinearity=hard_sigmoid),
                 cell=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(0.),nonlinearity=tanh),
                 outgate=Gate(W_in=GlorotUniform(), W_hid=Orthogonal(gain=1.1), 
                        W_cell=None,b=Constant(0.),nonlinearity=hard_sigmoid),
                 hid_init=Constant(0.),
                 cell_init=Constant(0.),
                 learn_init=True,
                 nonlinearity=tanh,
                 backwards=False,
                 gradient_steps=-1,
                 mask_input=None,
                 only_return_final=False,
                 hid_prop = False,
                 config="DCC",
                 **kwargs):
 
        super(BayesianLSTM, self).__init__(incoming, num_units, ingate, forgetgate, cell, outgate,
                 hid_init, cell_init, learn_init, nonlinearity, backwards, gradient_steps, mask_input, 
                                            only_return_final, hid_prop, **kwargs)
        self.name = 'BayesianLSTM'
        self.reg = True
        self.config = config
        self.log_sigma_in_init = log_sigma_in_init
        self.log_sigma_hid_init = log_sigma_hid_init
        
        if self.config[0] in {"L", "N"}:
            self.logsig_w_in = self.add_param(Constant(log_sigma_in_init), (4, self.num_inputs, self.num_units), name="logsig_w_in")
            self.logsig_w_hid = self.add_param(Constant(log_sigma_hid_init), (4, self.num_units, self.num_units), name="logsig_w_hid")
        else:
            self.logsig_w_in = T.zeros(4, dtype=floatX)
        
        if self.config[2] in {"L", "N", "D", "I"}:
            self.mu_in = self.add_param(Constant(1), (self.num_inputs,), name="mu_in")
        if self.config[2] in {"L", "N", "I"}:
            self.logsig_in = self.add_param(Constant(log_sigma_in_init), (self.num_inputs,), name="logsig_in")
        if self.config[2] in {"L", "N", "D", "R"}:
            self.mu_hid = self.add_param(Constant(1), (self.num_units,), name="mu_hid")
        if self.config[2] in {"L", "N", "R"}:
            self.logsig_hid = self.add_param(Constant(log_sigma_hid_init), (self.num_units,), name="logsig_hid")
        
        if self.config[1] in {"L", "N", "D"}:
            self.mu_gates = self.add_param(Constant(1), (4, self.num_units), name="mu_gates")
        if self.config[1] in {"L", "N"}:
            self.logsig_gates = self.add_param(Constant(log_sigma_hid_init), (4, self.num_units), name="logsig_gates")
        
        self.input_noise = None
        self.hidden_noise = None
        self.input_clip = None
        self.hidden_clip = None
        self.thresh = 3.
            
    def clip_func(self, mtx, to=8):
        mtx = T.switch(T.le(mtx, -to), -to, mtx)
        mtx = T.switch(T.ge(mtx, to), to, mtx)
        return mtx

    def generate_noise_and_clip(self, num_batch, deterministic = False, clip = False):
        if not deterministic:
            if self.config[0] in {"L", "N"}:
                self.input_w_noise = self._srng.normal((4, self.num_inputs, self.num_units), avg=0, std=1, dtype=floatX) * T.exp(self.logsig_w_in)
                self.hidden_w_noise = self._srng.normal((4, self.num_units, self.num_units), avg = 0.0, std = 1.0, dtype=floatX) * T.exp(self.logsig_w_hid)
            else:
                self.input_w_noise = T.zeros(4, dtype=floatX)
                self.hidden_w_noise = T.zeros(4, dtype=floatX)
                
            if self.config[2] in {"L", "N"}:
                self.input_noise = self._srng.normal((num_batch, self.num_inputs), avg=0, std=1, dtype=floatX) * T.exp(self.logsig_in) + self.mu_in
                self.hidden_noise = self._srng.normal((num_batch, self.num_units), avg=0, std=1, dtype=floatX) * T.exp(self.logsig_hid) + self.mu_hid
            elif self.config[2] == "I":
                self.input_noise = self._srng.normal((num_batch, self.num_inputs), avg=0, std=1, dtype=floatX) * T.exp(self.logsig_in) + self.mu_in
                self.hidden_noise = T.ones(1, dtype=floatX)
            elif self.config[2] == "R":
                self.input_noise = T.ones(1, dtype=floatX)
                self.hidden_noise = self._srng.normal((num_batch, self.num_units), avg=0, std=1, dtype=floatX) * T.exp(self.logsig_hid) + self.mu_hid
            elif self.config[2] == "D":
                self.input_noise = self.mu_in
                self.hidden_noise = self.mu_hid
            else:
                self.input_noise = T.ones(1, dtype=floatX)
                self.hidden_noise = T.ones(1, dtype=floatX)
                
            if self.config[1] in {"L", "N"}:
                self.gates_noise = self._srng.normal((4, num_batch, self.num_units), avg=0, std=1, dtype=floatX) * T.exp(self.logsig_gates)[:, np.newaxis, :] + self.mu_gates[:, np.newaxis, :]
            elif self.config[1] == "D":
                self.gates_noise = self.mu_gates
            else:
                self.gates_noise = T.ones(4, dtype=floatX)
        else:
            self.input_w_noise = T.zeros(4, dtype=floatX)
            self.hidden_w_noise = T.zeros(4, dtype=floatX)
            if self.config[2] in {"L", "N", "D"}:
                self.input_noise = self.mu_in
                self.hidden_noise = self.mu_hid
            elif self.config[2] == "I":
                self.input_noise = self.mu_in
                self.hidden_noise = T.ones(1, dtype=floatX)
            elif self.config[2] == "R":
                self.input_noise = T.ones(1, dtype=floatX)
                self.hidden_noise = self.mu_hid
            else:
                self.input_noise = T.ones(1, dtype=floatX)
                self.hidden_noise = T.ones(1, dtype=floatX)
            if self.config[1] in {"L", "N", "D"}:
                self.gates_noise = self.mu_gates
            else:
                self.gates_noise = T.ones(4, dtype=floatX)
        if clip:
            if self.config[0] == "L":
                log_alpha_w_in = self.clip_func(2*self.logsig_w_in - T.log(T.concatenate((self.W_in_to_ingate[None,:,:], self.W_in_to_forgetgate[None,:,:], self.W_in_to_cell[None,:,:], self.W_in_to_outgate[None,:,:]), axis = 0) ** 2))
                self.input_w_clip = T.le(log_alpha_w_in, self.thresh)
                log_alpha_w_hid = self.clip_func(2*self.logsig_w_hid - T.log(T.concatenate((self.W_hid_to_ingate[None,:,:], self.W_hid_to_forgetgate[None,:,:], self.W_hid_to_cell[None,:,:], self.W_hid_to_outgate[None,:,:]), axis = 0) ** 2))
                self.hidden_w_clip = T.le(log_alpha_w_hid, self.thresh)
            else:
                self.input_w_clip = T.ones(4, dtype=floatX)
                self.hidden_w_clip = T.ones(4, dtype=floatX)
                
            if self.config[2] == "L":
                log_alpha_in = self.clip_func(2*self.logsig_in-T.log(self.mu_in**2))
                self.input_clip = T.le(log_alpha_in, self.thresh)
                log_alpha_hid = self.clip_func(2*self.logsig_hid-T.log(self.mu_hid**2))
                self.hidden_clip = T.le(log_alpha_hid, self.thresh)
            elif self.config[2] == "I":
                log_alpha_in = self.clip_func(2*self.logsig_in-T.log(self.mu_in**2))
                self.input_clip = T.le(log_alpha_in, self.thresh)
                self.hidden_clip = T.ones(1, dtype=floatX)
            elif self.config[2] == "R":
                self.input_clip = T.ones(1, dtype=floatX)
                log_alpha_hid = self.clip_func(2*self.logsig_hid-T.log(self.mu_hid**2))
                self.hidden_clip = T.le(log_alpha_hid, self.thresh)
            else:
                self.input_clip = T.ones(1, dtype=floatX)
                self.hidden_clip = T.ones(1, dtype=floatX)
                
            if self.config[1] == "L":
                log_alpha_gates = self.clip_func(2*self.logsig_gates-T.log(self.mu_gates**2))
                self.gates_clip = T.le(log_alpha_gates, self.thresh)
            else:
                self.gates_clip = T.ones(4, dtype=floatX)
            
        else:
            self.input_w_clip = T.ones(4, dtype=floatX)
            self.hidden_w_clip = T.ones(4, dtype=floatX)
            self.input_clip = T.ones(1, dtype=floatX)
            self.hidden_clip = T.ones(1, dtype=floatX)
            self.gates_clip = T.ones(4, dtype=floatX)
        return
    
    def input_preactivation(self, input, gate_type, deterministic = False, clip = False):
        def get_matrix(gate_type):
            if gate_type == 'input':
                return self.W_in_to_ingate, 0
            if gate_type == 'forget':
                return self.W_in_to_forgetgate, 1
            if gate_type == 'cell':
                return self.W_in_to_cell, 2
            if gate_type == 'output':
                return self.W_in_to_outgate, 3
        W, idx = get_matrix(gate_type)
        input = input * self.input_noise * self.input_clip
        return T.dot(input, (W + self.input_w_noise[idx])*self.input_w_clip[idx])
        
    def hidden_preactivation(self, hidden, gate_type, deterministic = False, clip = False):
        def get_matrix(gate_type):
            if gate_type == 'input':
                return self.W_hid_to_ingate, 0
            if gate_type == 'forget':
                return self.W_hid_to_forgetgate, 1
            if gate_type == 'cell':
                return self.W_hid_to_cell, 2
            if gate_type == 'output':
                return self.W_hid_to_outgate, 3
        W, idx = get_matrix(gate_type)
        return T.dot(hidden, (W + self.hidden_w_noise[idx])*self.hidden_w_clip[idx])
    
    def get_output_for(self, inputs, deterministic=False, clip=False, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
 
        input = input.dimshuffle(1, 0, 2)
        num_batch = input.shape[1]
        seq_len = input.shape[0]
        self.generate_noise_and_clip(num_batch, deterministic, clip)
        
        input_i = self.input_preactivation(input, 'input', **kwargs)
        input_f = self.input_preactivation(input, 'forget', **kwargs)
        input_c = self.input_preactivation(input, 'cell', **kwargs)
        input_o = self.input_preactivation(input, 'output', **kwargs)
 
        def step(input_n_i, input_n_f, input_n_c, input_n_o, cell_previous, hid_previous, *args):
            ingate = self.nonlinearity_ingate((input_n_i + self.hidden_preactivation(hid_previous, 'input', **kwargs))\
                    *self.gates_noise[0]*self.gates_clip[0]+self.b_ingate)
            forgetgate = self.nonlinearity_forgetgate((input_n_f + self.hidden_preactivation(hid_previous, 'forget', **kwargs))\
                    *self.gates_noise[1]*self.gates_clip[1]+self.b_forgetgate)
            cell = forgetgate * cell_previous + \
                    ingate * self.nonlinearity_cell((input_n_c + self.hidden_preactivation(hid_previous, 'cell', **kwargs))\
                    *self.gates_noise[2]*self.gates_clip[2]+self.b_cell)
            outgate = self.nonlinearity_outgate((input_n_o + self.hidden_preactivation(hid_previous, 'output', **kwargs))\
                    *self.gates_noise[3]*self.gates_clip[3]+self.b_outgate)
            hid = outgate*self.nonlinearity(cell) * self.hidden_noise * self.hidden_clip
            return [cell, hid]
 
        def step_masked(input_n_i, input_n_f, input_n_c, input_n_o, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n_i, input_n_f, input_n_c, input_n_o, cell_previous, hid_previous, *args)
 
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)
            return [cell, hid]
 
        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input_i, input_f, input_c, input_o, mask]
            step_fun = step_masked
        else:
            sequences = [input_i, input_f, input_c, input_o]
            step_fun = step
        
        non_seqs = [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate, \
                    self.b_ingate, self.b_forgetgate,
                    self.b_cell, self.b_outgate,
                    self.hidden_w_noise, self.hidden_w_clip,
                    self.input_noise, self.input_clip,
                    self.hidden_noise, self.hidden_clip,
                    self.gates_noise, self.gates_clip]
        
        if self.hid_prop:
            hid_init, cell_init = inputs[1][0,:,:], inputs[1][1,:,:]
        else:
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)
            cell_init = T.dot(T.ones((num_batch, 1)), self.cell_init)
        
        cell_out, hid_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[cell_init, hid_init],
            go_backwards=self.backwards,
            truncate_gradient=self.gradient_steps,
            non_sequences=non_seqs,
            strict=True)[0]
 
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            hid_out = hid_out.dimshuffle(1, 0, 2)
            cell_out = cell_out.dimshuffle(1, 0, 2)
 
            if self.backwards:
                hid_out = hid_out[:, ::-1]
        if self.hid_prop:
            return T.concatenate([hid_out[np.newaxis,:,:,:],cell_out[np.newaxis,:,:,:]])
        else:
            return hid_out
            
    def eval_reg(self, train_size):
        # W
        W_in = T.concatenate((self.W_in_to_ingate[None,:,:], self.W_in_to_forgetgate[None,:,:], self.W_in_to_cell[None,:,:], self.W_in_to_outgate[None,:,:]), axis = 0)
        if self.config[0] == "N":
            KL_element_in = - self.logsig_w_in + 0.5 * (T.exp(2*self.logsig_w_in) + W_in**2) - 0.5
            KL = T.sum(KL_element_in)
        elif self.config[0] == "L":
            log_alpha_w_in = self.clip_func(2*self.logsig_w_in - T.log(W_in**2))
            KL = alpha_regf(log_alpha_w_in).sum()
        else:
            KL = T.zeros(1, dtype=floatX).sum()
            
        W_hid = T.concatenate((self.W_hid_to_ingate[None,:,:], self.W_hid_to_forgetgate[None,:,:], self.W_hid_to_cell[None,:,:], self.W_hid_to_outgate[None,:,:]), axis = 0)
        if self.config[0] == "N":
            KL_element_hid = - self.logsig_w_hid + 0.5 * (T.exp(2*self.logsig_w_hid) + W_hid**2) - 0.5
            KL += T.sum(KL_element_hid)
        elif self.config[0] == "L":
            log_alpha_w_hid = self.clip_func(2*self.logsig_w_hid - T.log(W_hid**2))
            KL += alpha_regf(log_alpha_w_hid).sum()
        
        # neurons
        if self.config[2] == "L":
            log_alpha_hid = self.clip_func(2*self.logsig_hid - T.log(self.mu_hid**2))
            KL += alpha_regf(log_alpha_hid).sum()
            log_alpha_in = self.clip_func(2*self.logsig_in - T.log(self.mu_in**2))
            KL += alpha_regf(log_alpha_in).sum()
        elif self.config[2] == "I":
            log_alpha_in = self.clip_func(2*self.logsig_in - T.log(self.mu_in**2))
            KL += alpha_regf(log_alpha_in).sum()
        elif self.config[2] == "R":
            log_alpha_hid = self.clip_func(2*self.logsig_hid - T.log(self.mu_hid**2))
            KL += alpha_regf(log_alpha_hid).sum()
        elif self.config[2] == "N":
            KL_element = - self.logsig_hid + 0.5 * (T.exp(2*self.logsig_hid) + self.mu_hid**2) - 0.5
            KL += KL_element.sum()
            KL_element = - self.logsig_in + 0.5 * (T.exp(2*self.logsig_in) + self.mu_in**2) - 0.5
            KL += KL_element.sum()
        
        # gates
        if self.config[1] == "L":
            log_alpha_gates = self.clip_func(2*self.logsig_gates - T.log(self.mu_gates**2))
            KL += alpha_regf(log_alpha_gates).sum()
        elif self.config[1] == "N":
            KL_element = - self.logsig_gates + 0.5 * (T.exp(2*self.logsig_gates) + self.mu_gates**2) - 0.5
            KL += KL_element.sum()
        
        return KL / train_size
    
    def get_ard(self):
        # w
        if self.config[0] == "L":
            log_alpha_w_in = 2*self.logsig_w_in.get_value() - 2 * np.log(np.abs(np.concatenate((self.W_in_to_ingate.get_value()[None,:,:], self.W_in_to_forgetgate.get_value()[None,:,:], self.W_in_to_cell.get_value()[None,:,:], self.W_in_to_outgate.get_value()[None,:,:]), axis = 0)))
            mask_w_in = log_alpha_w_in < self.thresh
            log_alpha_w_hid = 2*self.logsig_w_hid.get_value() - 2 * np.log(np.abs(np.concatenate((self.W_hid_to_ingate.get_value()[None,:,:], self.W_hid_to_forgetgate.get_value()[None,:,:], self.W_hid_to_cell.get_value()[None,:,:], self.W_hid_to_outgate.get_value()[None,:,:]), axis = 0)))
            mask_w_hid = log_alpha_w_hid < self.thresh
        else:
            mask_w_in = np.ones((4,)+self.W_in_to_ingate.get_value().shape)
            mask_w_hid = np.ones((4,)+self.W_hid_to_ingate.get_value().shape)
        # neurons
        mask_in = mask_w_in.any(axis=2).any(axis=0)
        mask_hid_by_w = mask_w_hid.any(axis=2).any(axis=0)
        mask_hid_by_z = np.ones_like(mask_hid_by_w).astype(bool)
        if self.config[2] == "L":
            log_alpha_hid = 2*self.logsig_hid.get_value() - 2 * np.log(np.abs(self.mu_hid.get_value()))
            log_alpha_in = 2*self.logsig_in.get_value() - 2 * np.log(np.abs(self.mu_in.get_value()))
            mask_in = np.logical_and(log_alpha_in < self.thresh, mask_in)
            mask_hid_by_z = log_alpha_hid < self.thresh
        elif self.config[2] == "I":
            log_alpha_in = 2*self.logsig_in.get_value() - 2 * np.log(np.abs(self.mu_in.get_value()))
            mask_in = np.logical_and(log_alpha_in < self.thresh, mask_in)
        elif self.config[2] == "R":
            log_alpha_hid = 2*self.logsig_hid.get_value() - 2 * np.log(np.abs(self.mu_hid.get_value()))
            mask_hid_by_z = log_alpha_hid < self.thresh
        # gates
        if self.config[1] == "L":
            log_alpha_gates = 2*self.logsig_gates.get_value() - 2 * np.log(np.abs(self.mu_gates.get_value()))
            mask = np.concatenate([mask_w_in, mask_w_hid], axis=1)
            mask_gates = np.logical_and(log_alpha_gates < self.thresh, mask.any(axis=1))
        else:
            mask = np.concatenate([mask_w_in, mask_w_hid], axis=1)
            mask_gates = mask.any(axis=1)
    
        return {"w_input":mask_w_in,\
                "w_hidden": mask_w_hid,\
                "gates": mask_gates,\
                "z_input": mask_in,\
                "z_hidden_by_w": mask_hid_by_w,
                "z_hidden": mask_hid_by_z}
