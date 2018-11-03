# Bayesian Compression of Recurrent neural networks

This repo contains the code for our EMNLP18 paper [Bayesian Compression for Natural Language Processing](https://arxiv.org/abs/1810.10927) and NIPS 2018 CDNNRIA Workshop paper [Baeysian Sparsification of Gated Recurrent Neural Networks](https://openreview.net/forum?id=ByMQgZHYoX).
We showed that Variational Dropout leads to extremely sparse solutions in recurrent neural networks. 

# Environment setup

```(bash)
sudo apt install virtualenv python-pip python-dev
virtualenv venv --system-site-packages
source venv/bin/activate

pip install numpy 'ipython[all]' matplotlib   
pip install --upgrade https://github.com/Theano/Theano/archive/rel-0.9.0.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Launch text classification experiments
Scripts for different datasets (IMDB, AGNews) are stored in Experiments folder.
```(bash)
cd Experiments/<dataset>
python main.py CONFIG
```
CONFIG is a 6-chars string defining which model to train. 
Each element of CONFIG is one of 3 chars:
* L: probabilistic weight with log-uniform prior and normal approximate posterior
* D: deterministic learnable weight
* C: constant weight with all elements equal to 1. 

6 chars stand for:
* CONFIG[0]: embedding layer, individual weights
* CONFIG[1]: embedding layer, group variabes for vocabulary elements
* CONFIG[2]: LSTM layer, individual weights
* CONFIG[3]: LSTM layer, group variables for preactivation of gates
* CONFIG[4]: LSTM layer, group variables for input and hidden neurons
* CONFIG[5]: fully-connected layer, individual weights

Configs for paper [Bayesian Compression for Natural Language Processing](https://arxiv.org/abs/1810.10927):
* Original: DCDCCD 
* Sparse-VD: LCLCCL
* SparseVD-Voc: LLLCCL

Configs for paper [Baeysian Sparsification of Gated Recurrent Neural Networks](https://openreview.net/forum?id=ByMQgZHYoX):
* Original: DCDCCD 
* SparseVD W: LCLCCL
* SparseVD W+N: LLLCLL
* SparseVD W+G+N: LLLLLL

During training, accuracy and compression rates are printed to the output stream.
