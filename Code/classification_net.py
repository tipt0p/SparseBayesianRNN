from __future__ import print_function
from layers_dense import BayesianDense, Dense
from layers_embedding import BayesianEmbedding
from layers_lstm import BayesianLSTM
from lasagne.nonlinearities import sigmoid, softmax
import theano.tensor as T
import lasagne.layers as ll
import numpy as np

class ClassificationNet:
    def __init__(self, vocab_size, n_emb, n_hidden, num_classes, config):
        target = T.ivector('target')
        inp = T.imatrix('input')
        mask = T.matrix('mask')
        self.config = config
        net = ll.InputLayer(shape=(None, None), input_var=inp)
        mask_l = ll.InputLayer(shape=(None, None), input_var=mask)
        self.emb_layer = BayesianEmbedding(net, vocab_size, n_emb, config=config[:2])
        self.lstm_layer = BayesianLSTM(self.emb_layer, n_hidden, only_return_final=True, mask_input=mask_l,
                           learn_init=False, config=config[2:5])
        self.net = (BayesianDense if config[-1]=="L" else Dense)(self.lstm_layer, num_classes, \
                             nonlinearity=sigmoid if num_classes==1 else softmax)
        self.inp = inp
        self.target = target
        self.mask = mask
    
    def compute_compression_masks(self):
        masks_embedding = self.emb_layer.get_ard()
        masks_lstm = self.lstm_layer.get_ard()
        masks_dense = self.net.get_ard()
        # aggregate z_masks
        mask_vocabulary = masks_embedding["z_voc"]
        mask_emb = np.logical_and(masks_embedding["z_emb"], masks_lstm["z_input"])
        # if hidden neuron doesn't influence LSTM, no guarantee it doesn't influence Dense
        mask_hidden = np.logical_or(masks_lstm["z_hidden_by_w"], masks_dense["w"].any(axis=1))
        mask_hidden = np.logical_and(mask_hidden, masks_lstm["z_hidden"])
        # further compress w
        masks_embedding["w"][np.logical_not(mask_vocabulary)] = 0
        masks_embedding["w"][:, np.logical_not(mask_emb)] = 0
        masks_lstm["w_input"][:, np.logical_not(mask_emb), :] = 0
        masks_lstm["w_input"][:, :, np.logical_not(mask_hidden)] = 0
        masks_lstm["w_hidden"][:, np.logical_not(mask_hidden), :] = 0
        masks_lstm["w_hidden"][:, :, np.logical_not(mask_hidden)] = 0
        masks_dense["w"][np.logical_not(mask_hidden)] = 0
        # further compress gates
        masks_lstm["gates"][:, np.logical_not(mask_hidden)] = 0
        return mask_vocabulary, mask_emb, mask_hidden,\
               masks_lstm["gates"],\
               masks_embedding["w"], masks_lstm["w_input"], masks_lstm["w_hidden"], masks_dense["w"]
    
    def evaluate_compression(self):
        mask_vocabulary, mask_emb, mask_hidden, mask_gates, \
        mask_w_emb, mask_w_inp, mask_w_hid, mask_w_dense = self.compute_compression_masks()
        # compute overall comression
        w_nonzero, w_all = 0, 0
        for w in [mask_w_emb, mask_w_inp, mask_w_hid, mask_w_dense]:
            w_nonzero += w.sum()
            w_all += w.size
        overall_compression = w_all / (w_nonzero+1e-8)
        # print compression per layer
        print("Compression per layers:")
        for layer_name, masks in [("Embedding", {"w":mask_w_emb, "z_voc":mask_vocabulary}),\
                                    ("LSTM", {"w_x":mask_w_inp, "w_h":mask_w_hid,\
                                              "z_x":mask_emb, "z_h":mask_hidden,\
                                              "gates":mask_gates}),\
                                    ("Dense", {"w":mask_w_dense})]:
            print(layer_name, end=": ")
            for key, matrix in masks.items():
                print("(%s: %d/%d)"%(key, matrix.sum(), matrix.size), end=" ")
            #print(flush=True)
        print("Overall compression:", overall_compression)#, flush=True)
        
    def compress(self):
        inv = np.logical_not
        mask_vocabulary, mask_emb, mask_hidden, mask_gates, \
        mask_w_emb, mask_w_inp, mask_w_hid, mask_w_dense = self.compute_compression_masks()
        params = ll.get_all_param_values(self.net)
        new_params = []
        i = 0
        W = params[0]
        if self.config[0] == "L":
            W[inv(mask_w_emb)] = 0
            i += 1
        if self.config[1] == "L":
            W *= params[i+1][:, np.newaxis]
            i += 2
        #W = W[mask_vocabulary]
        W[inv(mask_vocabulary)] = 0
        W = W[:, mask_emb]
        new_params.append(W)
        k1 = 2 if self.config[2] == "L" else 0
        k2 = 4 if self.config[4] == "L" else 0
        for j in range(4):
            W = params[i+1+j*3] # W^x
            if self.config[2] == "L":
                W[inv(mask_w_inp[j])] = 0
            if self.config[3] == "L":
                W *= params[i+k1+k2+15][j][np.newaxis, :]
                W[:, inv(mask_gates[j])] = 0
            if self.config[4] == "L":
                W *= params[i+k1+15][:, np.newaxis]
            W = W[mask_emb]
            W = W[:, mask_hidden]
            new_params.append(W)
            W = params[i+2+j*3] # W^h
            if self.config[2] == "L":
                W[inv(mask_w_hid[j])] = 0
            if self.config[3] == "L":
                W *= params[i+k1+k2+15][j][np.newaxis, :]
                W[:, inv(mask_gates[j])] = 0
            if self.config[4] == "L":
                W *= params[i+k1+17][:, np.newaxis]
            W = W[mask_hidden]
            W = W[:, mask_hidden]
            new_params.append(W)
            b = params[i+3+j*3] # b
            b = b[mask_hidden]
            new_params.append(b)
        new_params.append(params[i+13][:, mask_hidden])
        new_params.append(params[i+14][:, mask_hidden])
        j = i + k1 + k2 + (2 if self.config[3]=="L" else 0) + 14
        W = params[j+1]
        if self.config[5] == "L":
            W[inv(mask_w_dense)] = 0
            j += 1
        if self.config[4] == "L":
            W *= params[i+k1+17][:, np.newaxis]
        W = W[mask_hidden]
        new_params.append(W)
        b = params[j+1]
        new_params.append(b)
        return new_params, (mask_vocabulary.sum(), mask_emb.sum(), mask_hidden.sum())
            