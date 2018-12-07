from __future__ import print_function
from layers_dense import BayesianDense, Dense, BayesianDense_noLRT
from lasagne.layers import Gate
from lasagne.nonlinearities import tanh
from layers_lstm import BayesianLSTM
from lasagne.nonlinearities import sigmoid, softmax
from lasagne.init import Constant, Orthogonal
import theano.tensor as T
import lasagne.layers as ll
import numpy as np
import theano
from lasagne.layers import SliceLayer

class LMNet:
    def __init__(self, vocab_size, n_hidden, config, hid_prop = False, batch_size = 32):
        target = T.imatrix('target')
        inp = T.imatrix('input')
        inp_one_hot = T.eye(vocab_size)[inp]
        self.config = config
        net = ll.InputLayer(shape=(None, None, vocab_size), input_var=inp_one_hot)
        if hid_prop:
            hid = theano.shared(np.zeros((2,32, n_hidden), dtype=theano.config.floatX))#!!!!!!!!!! batch size 32
            hid_init = ll.InputLayer(shape=(2,None, n_hidden), name='hid_init',input_var=hid)
        self.lstm_layer = BayesianLSTM([net,hid_init] if hid_prop else net, n_hidden, only_return_final=False, learn_init=False, hid_prop = hid_prop,\
                                       config=config[:3], \
                                       ingate=Gate(W_in=Orthogonal(), W_hid=Orthogonal(gain=1.), \
                                                   W_cell=None,b=Constant(0.),nonlinearity=sigmoid), \
                                       forgetgate=Gate(W_in=Orthogonal(), W_hid=Orthogonal(gain=1.), \
                                                       W_cell=None,b=Constant(1. if hid_prop else 0.),nonlinearity=sigmoid), \
                                       cell=Gate(W_in=Orthogonal(), W_hid=Orthogonal(gain=1.), \
                                                 W_cell=None,b=Constant(0.),nonlinearity=tanh), \
                                       outgate=Gate(W_in=Orthogonal(), W_hid=Orthogonal(gain=1.), \
                                                    W_cell=None,b=Constant(0.),nonlinearity=sigmoid))
        
        if hid_prop:
            to_init_lstm = SliceLayer(self.lstm_layer,indices=-1, axis=2)
            self.dense = (BayesianDense_noLRT if config[-1]=="L" else Dense)(SliceLayer(self.lstm_layer,indices=0, axis=0), vocab_size, W=Orthogonal())
            self.hid = hid
            self.to_init_lstm = to_init_lstm
        else:
            self.dense = (BayesianDense if config[-1]=="L" else Dense)(self.lstm_layer, vocab_size, W=Orthogonal())
        self.net = ll.NonlinearityLayer(ll.ReshapeLayer(self.dense, (-1, vocab_size)),softmax)
        self.inp = inp
        self.target = target
    
    def compute_compression_masks(self):
        masks_lstm = self.lstm_layer.get_ard()
        masks_dense = self.dense.get_ard()
        # aggregate z_masks
        mask_vocabulary = masks_lstm["z_input"]
        # if hidden neuron doesn't influence LSTM, no guarantee it doesn't influence Dense
        mask_hidden = np.logical_or(masks_lstm["z_hidden_by_w"], masks_dense["w"].any(axis=1))
        mask_hidden = np.logical_and(mask_hidden, masks_lstm["z_hidden"])
        # further compress w
        masks_lstm["w_input"][:, np.logical_not(mask_vocabulary), :] = 0
        masks_lstm["w_input"][:, :, np.logical_not(mask_hidden)] = 0
        masks_lstm["w_hidden"][:, np.logical_not(mask_hidden), :] = 0
        masks_lstm["w_hidden"][:, :, np.logical_not(mask_hidden)] = 0
        masks_dense["w"][np.logical_not(mask_hidden)] = 0
        # further compress gates
        masks_lstm["gates"][:, np.logical_not(mask_hidden)] = 0
        return mask_vocabulary, mask_hidden,\
               masks_lstm["gates"],\
               masks_lstm["w_input"], masks_lstm["w_hidden"], masks_dense["w"]
    
    def evaluate_compression(self):
        mask_vocabulary, mask_hidden, mask_gates, \
        mask_w_inp, mask_w_hid, mask_w_dense = self.compute_compression_masks()
        # compute overall comression
        w_nonzero, w_all = 0, 0
        for w in [mask_w_inp, mask_w_hid, mask_w_dense]:
            w_nonzero += w.sum()
            w_all += w.size
        overall_compression = w_all / (w_nonzero+1e-8)
        # print compression per layer
        print("Compression per layers:")
        for layer_name, masks in [("LSTM", {"w_x":mask_w_inp, "w_h":mask_w_hid,\
                                              "z_x":mask_vocabulary, "z_h":mask_hidden,\
                                              "gates":mask_gates}),\
                                    ("Dense", {"w":mask_w_dense})]:
            print(layer_name, end=": ")
            for key, matrix in masks.items():
                print("(%s: %d/%d)"%(key, matrix.sum(), matrix.size), end=" ")
            #print(flush=True)
        print("Overall compression:", overall_compression)#, flush=True)