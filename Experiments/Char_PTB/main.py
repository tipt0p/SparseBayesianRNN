import sys
sys.path.append("../../Code/")
from PTB_data_utils import *
from utils import *
import lm_losses as lml
from lasagne.init import Constant

from lm_net import LMNet

#import lasagne.layers as ll

ptb_path = "../../Data/ptb/"
fnames = ["ptb.char.train.txt", "ptb.char.valid.txt", "ptb.char.test.txt"]

args = sys.argv[1:]
config = args[0]

file_name = "Results/weights_%s"%config
test_types = ['clip']

n_hidden = 1000
seq_len = 100
grad_clip = 1
batch_size = 64
vocab_size = 50
learning_rate = 0.002

num_epoches = 250
save_fq = 50
print_fq = 1
rec_params={"prior":config} 

seed = 0

train_data, valid_data, test_data = load_PTB_char(ptb_path, fnames[0], fnames[1], fnames[2])
train_data = PTB_char(train_data, seq_len, batch_size)
valid_data = PTB_char(valid_data, seq_len, batch_size)
test_data = PTB_char(test_data, seq_len, batch_size)

np.random.seed(seed)
lm_net = LMNet(vocab_size, n_hidden, config)
net, inp, target = lm_net.net, lm_net.inp, lm_net.target

#params = ll.get_all_param_values(net)
#print(params)

train_fn, val_fns, tmp_fn = get_char_lm_functions(net, inp, target, 
                                          grad_clip = grad_clip, learning_rate = learning_rate,
                                          train_size=train_data.num_examples*(train_data.length-1),
                                          test_types=test_types,
                                          loss_function=lml.cross_entropy, val_loss_function=lml.norm_cross_entropy)

np.random.seed(seed)
net = train_char_lm_model(net, train_data, test_data, train_fn, val_fns, 
                          num_epoches, valid_data = valid_data,
                          print_fq = print_fq, save_fq = save_fq, 
                          file_name = file_name,
                          sparsification_eval_fun=lm_net.evaluate_compression)

#params = ll.get_all_param_values(net)
#print(params)
#sys.exit()
save_net(net, file_name+".npy")


