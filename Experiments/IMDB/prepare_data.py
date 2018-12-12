import os
import sys
sys.path.append("../../Code")
import classification_data_utils

# numpy data without vocab: https://s3.amazonaws.com/text-datasets/imdb.npz

# gen data and vocab from texts
if not os.path.exists("aclImdb"):
    print("Downloading text data")
    os.system("wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
    os.system("tar -xf aclImdb_v1.tar.gz")
    
paths_train = ["aclImdb/train/pos", "aclImdb/train/neg"]
paths_test = ["aclImdb/test/pos", "aclImdb/test/neg"]
labels = [1, 0]

classification_data_utils.process_text_imdb(paths_train, paths_test, labels,
                                          lower=False, path_to_save="../../Data/IMDB")
# creates imdb_texts.npz and imdb_words.txt

# important: you should download file GoogleNews-vectors-negative300.bin from
# https://code.google.com/archive/p/word2vec/
classification_data_utils.filter_large_embeddings_file("GoogleNews-vectors-negative300.bin",\
                                                       "../../Data/IMDB/imdb_words.txt", path_to_save="../../Data/IMDB/w2v_imdb.txt")
# creates w2v_imdb.txt
