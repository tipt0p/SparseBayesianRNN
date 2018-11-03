import sys
sys.path.append("../../Code/")
import numpy as np
import utils
import classification_data_utils as data_utils
from lasagne.objectives import categorical_crossentropy
import lasagne.layers as ll
import theano.tensor as T

from classification_net import ClassificationNet

n_hidden = 512
n_emb = 300
seq_len = 100
maxlen = 100
save_fq = 50

batch_size = 128
index_from = 3
vocab_size = 20000
learning_rate = 0.0005
num_epoches = 150

np.random.seed(0)
(X_train, y_train, mask_train), (X_test, y_test, mask_test) = data_utils.load_matrix("agnews_texts.npz",\
                                                        num_words=vocab_size,
                                                       maxlen=maxlen, mask=True)
y_train -= 1
y_test -= 1
X_val, y_val, mask_val = X_train[int(0.95*len(X_train)):], \
               y_train[int(0.95*len(y_train)):],\
               mask_train[int(0.95*len(y_train)):]
X_train, y_train, mask_train = X_train[:int(0.95*len(X_train))], \
               y_train[:int(0.95*len(y_train))],\
               mask_train[:int(0.95*len(y_train))]

test_types = ["clip"]

train_data = data_utils.Reviews(X_train, y_train, batch_size, \
                                        mask=mask_train,
                                        shuffle=True)
val_data = data_utils.Reviews(X_val, y_val, batch_size, mask=mask_val)
test_data = data_utils.Reviews(X_test, y_test, batch_size, mask=mask_test)

for config in ["LCLCCL", "LLLCCL", "LLLCLL", "LLLLLL"]:
    print("Configuration:", config)
    np.random.seed(3)
    class_net = ClassificationNet(vocab_size+index_from, n_emb, n_hidden, 4, config)
    net, inp, target, mask = class_net.net, class_net.inp, class_net.target, class_net.mask
    params = list(np.load("Results/weights_%s.npy"%config))
    ll.set_all_param_values(net, params)
    class_net.evaluate_compression()
    new_params, (vocab_size_, n_emb_, n_hidden_) = class_net.compress()
    class_net_ = ClassificationNet(vocab_size+index_from, n_emb_, n_hidden_, 4, "DCDCCD")
    net_, inp_, target_, mask_ = class_net_.net, class_net_.inp, class_net_.target, class_net_.mask
    ll.set_all_param_values(net_, new_params)
    def accuracy(prediction, target):
        return T.eq(T.argmax(prediction, axis=1), target).mean()
    _, val_fns_, _ = utils.get_char_lm_functions(net_, inp_,
                                          target_, mask=mask_,
                                          learning_rate=learning_rate, 
                                          train_size=train_data.num_examples,\
                                          test_types=test_types, \
                                          loss_function=lambda a, b:\
                                          categorical_crossentropy(T.clip(a, 1.0e-6, 1.0 - 1.0e-6), b).mean(),
                                          val_loss_function=accuracy)
    print("Accuracy:", utils.print_evaluate(test_data, val_fns_, use_all=True))
    print()
