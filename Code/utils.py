from __future__ import print_function
import numpy as np
import lasagne.layers as ll
import theano.tensor as T
import theano
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import total_norm_constraint, adam
import time
from collections import OrderedDict

def save_net(net, file_name):
    params = ll.get_all_param_values(net)
    print('Save model: ' + file_name)#, flush=True)
    np.save(file_name, params)

def get_char_lm_functions(net, inp, target, hid_prop = False, 
                          hid=None, to_init_lstm=None, mask=None,
                          grad_clip=10**9, learning_rate=0.001, 
                          loss_function=categorical_crossentropy,
                          val_loss_function=categorical_crossentropy, 
                          train_size = 1, 
                          test_types = ["MC", "MC_clip", "usual", "clip"]):
    #MC and MC_clip are not supported for hid_prop = True
    target_vec = T.reshape(target, (-1,))
    
    if hid_prop:
        last_hid, prediction = ll.get_output([to_init_lstm,net])
        reg = T.sum([layer.eval_reg(train_size) for i, layer in enumerate(ll.get_all_layers(net)[2:-1:2]) if 'reg' in layer.__dict__])
        #[2:-1:2] - depends on the network structure
    else:
        prediction = ll.get_output(net)
        reg = T.sum([layer.eval_reg(train_size) for i, layer in enumerate(ll.get_all_layers(net)[1:]) if 'reg' in layer.__dict__])
    loss = loss_function(prediction, target_vec).mean() + reg
    
    all_params = ll.get_all_params(net, trainable=True)
    all_grads = T.grad(loss, all_params)
    scaled_grads, total_norm = total_norm_constraint(all_grads, grad_clip, return_norm=True)
    
    updates = adam(scaled_grads, all_params, learning_rate)
    if hid_prop:
        updates[hid] = last_hid
    train_fn = theano.function([inp, target] + ([mask] if mask is not None else []),\
                               [loss, total_norm], updates=updates, allow_input_downcast=True)
    
    val_fns = {}
    for t in test_types:
        clip = ('clip' in t)   
        if 'MC' in t:
            #MC and MC_clip are not supported for hid_prop = True
            test_prediction = T.concatenate([T.shape_padright(ll.get_output(net, clip=clip)) for i in range(10)], axis = -1).mean(axis=2)
            test_loss = val_loss_function(test_prediction, target_vec).mean()
            val_fns[t] = theano.function([inp, target] + ([mask] if mask is not None else []),\
                                        test_loss, allow_input_downcast=True)
        elif hid_prop:
            test_last_hid, test_prediction = ll.get_output([to_init_lstm,net], deterministic=True, clip=clip)
            test_loss = val_loss_function(test_prediction, target_vec).mean()
            updates_test = OrderedDict()
            updates_test[hid] = test_last_hid
            val_fns[t] = theano.function([inp, target] + ([mask] if mask is not None else []),\
                                           test_loss, updates=updates_test, allow_input_downcast=True)
        else:
            test_prediction = ll.get_output(net, deterministic=True, clip=clip)
            test_loss = val_loss_function(test_prediction, target_vec).mean()
            val_fns[t] = theano.function([inp, target] + ([mask] if mask is not None else []),\
                                           test_loss, allow_input_downcast=True)
        
    tmp_fn = theano.function([], [], allow_input_downcast=True)
    
    if hid_prop:
        updates_reset = OrderedDict()
        updates_reset[hid] = theano.tensor.zeros_like(hid, dtype=theano.config.floatX)
        reset_hid_init_fn = theano.function([], [], updates=updates_reset, allow_input_downcast=True)
        return train_fn, val_fns, tmp_fn, reset_hid_init_fn
    
    return train_fn, val_fns, tmp_fn

def evaluate(data, val_fn, hid_prop = False, reset_hid_init_fn = None, use_all = True):
    data.to_first_batch()
    if hid_prop:
        reset_hid_init_fn()
    if use_all:
        num_batches = data.num_batches
    else:
        num_batches = min(1, data.num_batches)
    err = 0
    for d_batch in range(num_batches):
        batch = data.get_next_batch()
        err += val_fn(*batch)*batch[0].shape[1 if hid_prop else 0]
    if use_all:
        err /= (data.data.shape[0]-1) if hid_prop else data.num_examples 
    else:
        err /= num_batches * batch[0].shape[1 if hid_prop else 0]
    if hid_prop:
        err = np.exp(err)
    data.to_first_batch()
    return err
        
def print_evaluate(data, val_fns, hid_prop = False, reset_hid_init_fn = None, use_all = True):
    result = '(' + ','.join(val_fns.keys()) + '): ' + \
    ', '.join(['%.4f' % evaluate(data, val_fns[fn], hid_prop, reset_hid_init_fn, use_all) for fn in val_fns.keys()])
    return result

def train_char_lm_model(net, train_data, test_data, train_fn, val_fns, tmp_fn,
                        num_epochs, hid_prop = False, reset_hid_init_fn = None,
                        valid_data=None, print_fq = 1, save_fq = 0, file_name = "models/",\
                        sparsification_eval_fun=None):
    print("Training ...")
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        grad_norm, mean_grad_norm, tr_l = 0, 0, 0.
        train_data.new_epoch()
        if hid_prop:
            reset_hid_init_fn()
        for batch_index in range(train_data.num_batches):
            batch = train_data.get_next_batch()
            l, nor = train_fn(*batch)
            grad_norm = max(nor, grad_norm)
            mean_grad_norm += nor
            tr_l += l * batch[0].shape[1 if hid_prop else 0]
        mean_grad_norm /= train_data.num_batches
        tr_l /= (len(train_data.data)-1) if hid_prop else train_data.num_examples
        train_time = time.time()
        
        if (epoch) % print_fq == 0 or epoch == num_epochs:
            print("Epoch {} took {:.3f}s \t loss = {}, \t grad norm = {},\t{}".
              format(epoch, train_time - start_time, tr_l, grad_norm, mean_grad_norm))#, flush=True)
            print ("Train " + print_evaluate(train_data, val_fns, hid_prop, reset_hid_init_fn, False))#, flush=True)
            if valid_data:
                print ("Val " + print_evaluate(valid_data, val_fns, hid_prop, reset_hid_init_fn))#, flush=True)
            print ("Test " + print_evaluate(test_data, val_fns, hid_prop, reset_hid_init_fn))#, flush=True)
            print("Testing took {:.3f}s".format(time.time() - train_time))#, flush=True)
            if sparsification_eval_fun:
                sparsification_eval_fun()
        if save_fq and (epoch) % save_fq == 0:
            save_net(net, file_name+"_epoch_"+str(epoch)+".npy")
    return net
