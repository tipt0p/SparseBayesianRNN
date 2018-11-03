import numpy as np
import lasagne.layers as ll
import theano.tensor as T
import theano
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import total_norm_constraint, adam
import time

def save_net(net, file_name):
    params = ll.get_all_param_values(net)
    print('Save model: ' + file_name, flush=True)
    np.save(file_name, params)

def get_char_lm_functions(net, inp, target, mask=None,
                          grad_clip=10**9, learning_rate=0.001, 
                          loss_function=categorical_crossentropy,
                          val_loss_function=categorical_crossentropy, 
                          weight_decay = 0, 
                          train_size = 1, 
                          test_types = ["MC", "MC_clip", "usual", "clip"]):
    target_vec = T.reshape(target, (-1,))
    
    prediction = ll.get_output(net)
    reg = T.sum([layer.eval_reg(train_size) for i, layer in enumerate(ll.get_all_layers(net)[1:]) if 'reg' in layer.__dict__])
    loss = loss_function(prediction, target_vec).mean() + reg
    
    all_params = ll.get_all_params(net, trainable=True)
    all_grads = T.grad(loss, all_params)
    scaled_grads, total_norm = total_norm_constraint(all_grads, grad_clip, return_norm=True)
    
    updates = adam(scaled_grads, all_params, learning_rate)
    train_fn = theano.function([inp, target] + ([mask] if mask is not None else []),\
                               [loss, total_norm], updates=updates, allow_input_downcast=True)
    
    val_fns = {}
    if 'usual' in test_types:
        test_prediction = ll.get_output(net, deterministic=True)
        test_loss = val_loss_function(test_prediction, target_vec).mean()
        val_fns['usual'] = theano.function([inp, target] + ([mask] if mask is not None else []),\
                                           test_loss, allow_input_downcast=True)
    if 'clip' in test_types:
        test_prediction_clip = ll.get_output(net, deterministic=True, clip_test = True)
        test_loss_clip = val_loss_function(test_prediction_clip, target_vec).mean()
        val_fns['clip'] = theano.function([inp, target] + ([mask] if mask is not None else []),\
                                          test_loss_clip, allow_input_downcast=True)
    if 'MC' in test_types:
        test_prediction_MC = T.concatenate([T.shape_padright(ll.get_output(net)) for i in range(10)], axis = -1).mean(axis=2)
        test_loss_MC = val_loss_function(test_prediction_MC, target_vec).mean()
        val_fns['MC'] = theano.function([inp, target] + ([mask] if mask is not None else []),\
                                        test_loss_MC, allow_input_downcast=True)
    if 'MC_clip' in test_types:
        test_prediction_MC_clip = T.concatenate([T.shape_padright(ll.get_output(net, clip_train = True)) for i in range(10)],
                                                axis = -1).mean(axis=2)
        test_loss_MC_clip = val_loss_function(test_prediction_MC_clip, target_vec).mean()
        val_fns['MC_clip'] = theano.function([inp, target] + ([mask] if mask is not None else []),\
                                             test_loss_MC_clip, allow_input_downcast=True)
    tmp_fn = theano.function([inp] + ([mask] if mask else []), prediction)
    return train_fn, val_fns, tmp_fn

def evaluate(data, val_fn, use_all = True):
        data.to_first_batch()
        if use_all:
            num_batches = data.num_batches
        else:
            num_batches = min(1, data.num_batches)
        err = 0
        for d_batch in range(num_batches):
            batch = data.get_next_batch()
            err += val_fn(*batch)*batch[0].shape[0]
        if use_all:
            err /= data.num_examples
        else:
            err /= num_batches * batch[0].shape[0]
        data.to_first_batch()
        return err
        
def print_evaluate(data, val_fns, use_all = True):
    result = '(' + ','.join(val_fns.keys()) + '): ' + \
    ', '.join(['%.4f' % evaluate(data, val_fns[fn], use_all) for fn in val_fns.keys()])
    return result

def train_char_lm_model(net, train_data, test_data, train_fn, val_fns,
                        num_epochs,
                        valid_data=None, print_fq = 1, save_fq = 0, file_name = "models/",\
                        sparsification_eval_fun=None):
    print("Training ...")
    for epoch in range(1, num_epochs+1):
        start_time = time.time()
        grad_norm, mean_grad_norm, tr_l = 0, 0, 0.
        train_data.new_epoch()
        for batch_index in range(train_data.num_batches):
            batch = train_data.get_next_batch()
            l, nor = train_fn(*batch)
            grad_norm = max(nor, grad_norm)
            mean_grad_norm += nor
            tr_l += l * batch[0].shape[0]
        mean_grad_norm /= train_data.num_batches
        tr_l /= train_data.num_examples
        train_time = time.time()
        
        if (epoch) % print_fq == 0:
            print("Epoch {} took {:.3f}s \t loss = {}, \t grad norm = {},\t{}".
              format(epoch, train_time - start_time, tr_l, grad_norm, mean_grad_norm), flush=True)
            print ("Train " + print_evaluate(train_data, val_fns, False), flush=True)
            if valid_data:
                print ("Val " + print_evaluate(valid_data, val_fns), flush=True)
            print ("Test " + print_evaluate(test_data, val_fns), flush=True)
            print("Testing took {:.3f}s".format(time.time() - train_time), flush=True)
            if sparsification_eval_fun:
                sparsification_eval_fun()
        if save_fq and (epoch) % save_fq == 0:
            save_net(net, file_name+"_epoch_"+str(epoch)+".npy")
    return net
