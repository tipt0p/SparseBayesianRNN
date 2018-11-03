import sys
sys.path.append("../../Code/")
import numpy as np
import utils
import classification_data_utils as data_utils
from lasagne.objectives import binary_crossentropy
import lasagne.layers as ll
import theano.tensor as T

from classification_net import ClassificationNet

n_hidden = 128
n_emb = 300
seq_len = 200
maxlen = 200
save_fq = 200

batch_size = 128
index_from = 3
vocab_size = 20000
learning_rate = 0.0005
num_epoches = 1000

args = sys.argv[1:]
config = args[0]

np.random.seed(0)
(X_train, y_train, mask_train), (X_test, y_test, mask_test) = data_utils.load_matrix("imdb.npz",\
                                                        num_words=vocab_size,
                                                       maxlen=maxlen, mask=True)
X_val, y_val, mask_val = X_train[int(0.85*len(X_train)):], \
               y_train[int(0.85*len(y_train)):],\
               mask_train[int(0.85*len(y_train)):]
X_train, y_train, mask_train = X_train[:int(0.85*len(X_train))], \
               y_train[:int(0.85*len(y_train))],\
               mask_train[:int(0.85*len(y_train))]

file_name = "Results/weights_%s"%config

test_types = ["clip"]

train_data = data_utils.Reviews(X_train, y_train, batch_size, \
                                        mask=mask_train,
                                        shuffle=True)
val_data = data_utils.Reviews(X_val, y_val, batch_size, mask=mask_val)
test_data = data_utils.Reviews(X_test, y_test, batch_size, mask=mask_test)

np.random.seed(3)

class_net = ClassificationNet(vocab_size+index_from, n_emb, n_hidden, 1, config)
net, inp, target, mask = class_net.net, class_net.inp, class_net.target, class_net.mask

"""
### pretrained embeddings init
W = data_utils.init_embedding_with_word2vec(ll.get_all_param_values(net)[0], 
                                                    "imdb_words.txt",
                                                    "w2v_imdb.txt",
                                                    index_from=index_from)

params = ll.get_all_param_values(net)
params[0] = W
ll.set_all_param_values(net, params)
"""


def accuracy(prediction, target):
    return T.eq(T.ge(prediction, 0.5).ravel(), target).mean()
train_fn, val_fns, tmp_fn = utils.get_char_lm_functions(net, inp,
                                          target, mask=mask,
                                          learning_rate=learning_rate, 
                                          train_size=train_data.num_examples,\
                                          test_types=test_types, \
                                          loss_function=lambda a, b:\
                                          binary_crossentropy(T.clip(a.ravel(), 1.0e-6, 1.0 - 1.0e-6), b).mean(),
                                          val_loss_function=accuracy)

net = utils.train_char_lm_model(net, train_data, test_data, \
                                        train_fn, val_fns, 
                          num_epoches, valid_data=val_data,
                          print_fq = 20, save_fq = save_fq, file_name=file_name,\
                          sparsification_eval_fun=class_net.evaluate_compression)
utils.save_net(net, file_name+".npy")