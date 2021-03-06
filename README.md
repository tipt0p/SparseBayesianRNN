# Bayesian Compression of Recurrent neural networks

This repo contains the code for our EMNLP18 paper [Bayesian Compression for Natural Language Processing](https://arxiv.org/abs/1810.10927) and NeurIPS 2018 CDNNRIA Workshop paper [Baeysian Sparsification of Gated Recurrent Neural Networks](https://arxiv.org/abs/1812.05692).
We showed that Variational Dropout leads to extremely sparse solutions in recurrent neural networks. 

Code uses [Lasagne + Theano](https://lasagne.readthedocs.io/en/latest/user/installation.html).

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

Configs for paper [Baeysian Sparsification of Gated Recurrent Neural Networks](https://arxiv.org/abs/1812.05692):
* Original: DCDCCD 
* SparseVD W: LCLCCL
* SparseVD W+N: LLLCLL
* SparseVD W+G+N: LLLLLL

During training, accuracy and compression rates are printed to the output stream.
Results from the papers are obtained with Python 3.6.3, theano 1.0.1 and lasagne 0.2.dev1.

## Launch language generation experiments
Scripts for char-level and word-level tasks on PTB dataset are stored in Experiments folder (Char_PTB, Word_PTB).
```(bash)
cd Experiments/<dataset>
python main.py CONFIG
```
CONFIG is a 4-chars string defining which model to train. 
Each element of CONFIG is one of 3 chars:
* L: probabilistic weight with log-uniform prior and normal approximate posterior
* D: deterministic learnable weight
* C: constant weight with all elements equal to 1. 

There are also 2 additional options for group sparsification of neurons in LSTM layer:
* R: L option ONLY for hidden neurons
* I: L option ONLY for input neurons

4 chars stand for:
* CONFIG[0]: LSTM layer, individual weights
* CONFIG[1]: LSTM layer, group variables for preactivation of gates
* CONFIG[2]: LSTM layer, group variables for input and hidden neurons
* CONFIG[3]: fully-connected layer, individual weights

Configs for paper [Bayesian Compression for Natural Language Processing](https://arxiv.org/abs/1810.10927):
* Original: DCCD 
* Sparse-VD: LCCL
* SparseVD-Voc: LCIL

Configs for paper [Baeysian Sparsification of Gated Recurrent Neural Networks](https://arxiv.org/abs/1812.05692):
* Original: DCCD 
* SparseVD W: LCCL
* SparseVD W+N: LCRL
* SparseVD W+G+N: LLRL

During training, accuracy and compression rates are printed to the output stream.
Results from the papers are obtained with Python 2.7.13, theano 0.9.0.dev-RELEASE and lasagne 0.2.dev1.

# Citation

If you found this code useful please cite one of our papers:
```
@InProceedings{SparseBayesianRNN,
  author = 	"Chirkova, Nadezhda and Lobacheva, Ekaterina and Vetrov, Dmitry",
  title = 	"Bayesian Compression for Natural Language Processing",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"2910--2915",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1319"
}
```
```
@InProceedings{SparseBayesianGatedRNN,
  author = 	"Lobacheva, Ekaterina and Chirkova, Nadezhda and Vetrov, Dmitry",
  title = 	"Bayesian Sparsification of Gated Recurrent Neural Networks",
  booktitle = 	"Proceedings of the Workshop on Compact Deep Neural Networks with industrial applications, NeurIPS 2018",
  year = 	"2018"
}
```
