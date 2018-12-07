from __future__ import division
import numpy as np
import os
import re
from collections import defaultdict
import operator
import pandas as pd

import struct
import sys

def texts_to_sequences(texts):
    texts = [[i for i in seq.lower().split(" ") if i] for seq in texts]
    word_counts = {}
    for seq in texts:
        for w in seq:
            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
    res = []
    for seq in texts:
         vect = []
         for w in seq:
            i = word_index.get(w)
            if i is not None:
                vect.append(i)
         res.append(vect)
    return res
   
class Reviews:
    def __init__(self, X, y, batch_size, mask=None, shuffle=False):
        self.X = X
        self.y = y
        self.mask = mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_examples = len(X)
        self.num_batches = int(np.ceil(self.num_examples/float(self.batch_size)))
        self.inds = [(i*self.batch_size, min(self.num_examples, (i+1)*self.batch_size)) for i in range(0, self.num_batches)]
        
    def new_epoch(self):
        if self.shuffle:
            inds = np.random.permutation(np.arange(self.num_examples))
            self.X = self.X[inds]
            self.y = self.y[inds]
            if self.mask is not None:
                self.mask = self.mask[inds]
        self.batch_ind = 0
        
    def to_first_batch(self):
        self.batch_ind = 0
      
    def get_next_batch(self):
        batch_start, batch_end = self.inds[self.batch_ind]
        self.batch_ind += 1
        if self.mask is None:
            return self.X[batch_start:batch_end], self.y[batch_start:batch_end]
        else:
            return self.X[batch_start:batch_end], self.y[batch_start:batch_end],\
                   self.mask[batch_start:batch_end]
    
def load_matrix(path, num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, mask=False, **kwargs):
    """
    Modified code from Keras
    Loads data matrixes from npz file, crops and pads seqs and returns
    shuffled (x_train, y_train), (x_test, y_test)
    """
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if not num_words:
        num_words = max([max(x) for x in xs])
    if not maxlen:
        maxlen = max([len(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    xs_new = []
    if mask:
        masks = []
    for x in xs:
        x = x[:maxlen] # crop long sequences
        if oov_char is not None: # replace rare or frequent symbols 
            x = [w if (skip_top <= w < num_words) else oov_char for w in x]
        else: # or filter rare and frequent symbols
            x = [w for w in x if skip_top <= w < num_words]
        x_padded = np.zeros(maxlen)#, dtype = 'int32')
        x_padded[-len(x):] = x
        #x_padded[:len(x)] = x
        xs_new.append(x_padded)    
        if mask:
            mask_ = np.zeros(maxlen)
            mask_[-len(x):] = 1
            masks.append(mask_)
            
    idx = len(x_train)
    x_train, y_train = np.array(xs_new[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs_new[idx:]), np.array(labels[idx:])
    if mask:
        mask_train, mask_test = np.array(masks[:idx]), np.array(masks[idx:])
    
    if not mask:
        return (x_train, y_train), (x_test, y_test)
    else:
        return (x_train, y_train, mask_train), (x_test, y_test, mask_test)


def generate_vocab_paths(paths, lower=False):
    vocab_freqs = defaultdict(int)
    doc_counts = defaultdict(int)
    if type(paths) == str:
        paths = [paths]
    for path in paths:
            for doc_name in os.listdir(path):
                with open(os.path.join(path, doc_name)) as doc_f:
                    doc = doc_f.read()
                    if lower:
                        doc = doc.lower()
                    doc_seen = set()

                    tokens = [s for s in re.split(r'\W+', doc) 
                                  if s and not s.isspace()]

                    for token in tokens:
                        vocab_freqs[token] += 1
                        if token not in doc_seen:
                            doc_counts[token] += 1
                            doc_seen.add(token)
    return vocab_freqs, doc_counts

def generate_vocab_tab(tab_path, lower=False):
    vocab_freqs = defaultdict(int)
    doc_counts = defaultdict(int)
    tab = pd.read_csv(tab_path).values
    for row in tab:
        label, doc = float(row[0]), " ".join(row[1:])
        if lower:
            doc = doc.lower()
        doc_seen = set()
        
        tokens = [s for s in re.split(r'\W+', doc) 
                  if s and not s.isspace()]

        for token in tokens:
            vocab_freqs[token] += 1
            if token not in doc_seen:
                doc_counts[token] += 1
                doc_seen.add(token)
    return vocab_freqs, doc_counts

def generate_vocab(paths=None,\
                       max_vocab_size=100*1000,\
                   doc_count_threshold=1, lower=False,
                      eos_symb="</s>",
                  save_path="words.txt",
                  tab_path=None):
    
        if paths is not None:
            vocab_freqs, doc_counts = generate_vocab_paths(paths, lower)
        elif tab is not None:
            vocab_freqs, doc_counts = generate_vocab_tab(tab_path, lower)
        else:
            raise ValueError("Either paths or tab must be provided")
        
        vocab_freqs = dict((term, freq) for term, freq in 
                           vocab_freqs.items()
                          if doc_counts[term] > doc_count_threshold)

        # Sort by frequency
        ordered_vocab_freqs = sorted(vocab_freqs.items(), \
                                     key=operator.itemgetter(1), \
                                     reverse=True)

        # Limit vocab size
        ordered_vocab_freqs = ordered_vocab_freqs[:max_vocab_size]

        # Add EOS token
        ordered_vocab_freqs.append((eos_symb, 1))
        
        words = [pair[0] for pair in ordered_vocab_freqs]
        with open(save_path, "w") as fout:
            fout.write("\n".join(words))
        return ordered_vocab_freqs

def process_text_agnews(tab_path_train, tab_path_test, \
                      path_to_save="./", \
                      lower=True, max_vocab_size=100*1000,
                      doc_count_threshold=1):
    """
    adapted from TensorFlow: 
    https://github.com/tensorflow/models/blob/
           master/research/adversarial_text/data/
           
    transforms texts into ids seq
    creates npz file with 5 keys: x_train, x_test (arrays of lists), 
                                  y_train, y_test (arrays of labels),
                                  array of words sorted by desc. freq
    """
    
    eos_symb = "</s>"
    def generate_texts(tab_path, vocab_freqs, lower=True):
        docs = []
        targets = []
        vocab_ids = dict([(line[0].strip(), i) for i, line 
                  in enumerate(vocab_freqs)])
        tab = pd.read_csv(tab_path).values
        for row in tab:
            label, doc = int(row[0]), " ".join(row[1:])
            if lower:
                doc = doc.lower()

            tokens = [s for s in re.split(r'\W+', doc) 
                      if s and not s.isspace()]
            ids = []
            for token in tokens:
                if token in vocab_ids:
                    ids.append(vocab_ids[token])
            ids.append(vocab_ids[eos_symb])
            if len(ids) < 2:
                continue
            docs.append(ids)
            targets.append(label)

        return docs, targets

    vocab_freqs = generate_vocab(tab_path=tab_path_train, \
                                 max_vocab_size=max_vocab_size,\
                                 doc_count_threshold=doc_count_threshold, \
                                 lower=lower,\
                                 save_path=os.path.join(path_to_save, "agnews_words.txt"))
    docs_train, targets_train = generate_texts(tab_path_train,\
                                               vocab_freqs, lower)
    docs_test, targets_test = generate_texts(tab_path_test,\
                                             vocab_freqs, lower)
    np.savez(os.path.join(path_to_save, "agnews_texts"), x_train=docs_train, x_test=docs_test,
            y_train=targets_train, y_test=targets_test, 
            vocab=np.array([item[0] for item in vocab_freqs]))
    return path_to_save+".npz"
    
def process_text_imdb(paths_train, paths_test, labels, \
                      path_to_save="./", \
                      lower=False, max_vocab_size=100*1000,
                      doc_count_threshold=1):
    """
    adapted from TensorFlow: 
    https://github.com/tensorflow/models/blob/
           master/research/adversarial_text/data/
           
    transforms texts into ids seq
    creates npz file with 5 keys: x_train, x_test (arrays of lists), 
                                  y_train, y_test (arrays of labels),
                                  array of words sorted by desc. freq
    """
    
    eos_symb = "</s>"
    
    def generate_texts(paths, labels, vocab_freqs, lower=False):
        docs = []
        targets = []
        vocab_ids = dict([(line[0].strip(), i) for i, line 
                  in enumerate(vocab_freqs)])
        for path, label in zip(paths, labels):
            for doc_name in os.listdir(path):
                with open(os.path.join(path, doc_name)) as doc_f:
                    doc = doc_f.read()
                    if lower:
                        doc = doc.lower()
                    tokens = [s for s in re.split(r'\W+', doc) 
                              if s and not s.isspace()]
                    ids = []
                    for token in tokens:
                        if token in vocab_ids:
                            ids.append(vocab_ids[token])
                    ids.append(vocab_ids[eos_symb])
                    if len(ids) < 2:
                        continue
                    docs.append(ids)
                    targets.append(label)
        docs = np.array(docs)
        targets = np.array(targets)
        return docs, targets
    
    vocab_freqs = generate_vocab(paths_train, max_vocab_size,\
                                 doc_count_threshold, lower, \
                                 save_path=os.path.join(path_to_save, "imdb_words.txt"))
    docs_train, targets_train = generate_texts(paths_train, labels, \
                                               vocab_freqs, lower)
    docs_test, targets_test = generate_texts(paths_test, labels, \
                                             vocab_freqs, lower)
    np.savez(os.path.join(path_to_save, "imdb_texts"), x_train=docs_train, x_test=docs_test,
            y_train=targets_train, y_test=targets_test, 
            vocab=np.array([item[0] for item in vocab_freqs]))
    return path_to_save+".npz"

def filter_large_embeddings_file(emb_fn, vocab_fn, path_to_save="./", max_vectors_num=52000):
    words = open(vocab_fn).read().split("\n")
    words = set(words)
    FLOAT_SIZE = 4
    vectors = dict()
    
    with open(emb_fn, 'rb') as f:
        c = None

        # read the header
        header = b""
        while c != b"\n":
            c = f.read(1)
            header += c

        num_vectors, vector_len = (int(x) for x in header.split())

        while len(vectors) < max_vectors_num: # processing the whole file takes too much time
            word = b""        
            while True:
                c = f.read(1)
                if c == b" ":
                    break
                word += c

            binary_vector = f.read(FLOAT_SIZE * vector_len)
            if str(word, "utf-8") in words:
                vectors[word] = [ struct.unpack_from('f', binary_vector, i)[0] 
                              for i in range(0, len(binary_vector), FLOAT_SIZE) ]

            sys.stdout.write("%d, %d%%\r" % (len(vectors), len(vectors) / num_vectors * 100))
            sys.stdout.flush()
    
    with open(path_to_save, 'w') as fout:
        for key in vectors:
            fout.write(str(key, "utf-8")+";"+str(vectors[key])+"\n")
            
def filter_large_embeddings_txt(emb_fn, vocab_fn, path_to_save="./", max_vectors_num=52000):
    vectors = {}
    with open(emb_fn) as fin:
        for line in fin:
            items = line[:-1].split(" ")
            key = items[0]
            if key in words:
                value = items[1:]
                value = [float(x) for x in value]
                vectors[key] = value
            sys.stdout.write("%d\r" % (len(vectors)))
            sys.stdout.flush()
            if len(vectors) == max_vectors_num:
                break
    
    with open(path_to_save, 'w') as fout:
        for key in vectors:
            fout.write(str(key, "utf-8")+";"+str(vectors[key])+"\n")

    
def init_embedding_with_word2vec(matrix, vocab_file, w2v_file, index_from=0):
    """
    matrix is np.ndarray from embedding layer
    """
    words = open(vocab_file).read().split("\n")
    vocab_size = matrix.shape[0] - index_from
    vectors = {}
    with open(w2v_file) as fin:
        for line in fin:
            key, value = line[:-1].split(";")
            value = eval(value)
            vectors[key] = value
    if matrix.shape[1] != len(list(vectors.items())[0][1]):
        raise ValueError("Shapes of embedding layer and of pretrained embeddings mismatch")
    n = 0
    for i, word in enumerate(words[:vocab_size]):
        if word in vectors:
            matrix[i+index_from] = np.array(vectors[word])
            n += 1
    print("Initialized %d embeddings"%n)
    return matrix
