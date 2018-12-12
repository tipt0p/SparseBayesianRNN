import os
import sys
sys.path.append("../../Code/")
from classification_data_utils import process_text_agnews, filter_large_embeddings_txt

os.system("git clone https://github.com/mhjabreel/CharCNN")

tab_path_train = "CharCNN/data/ag_news_csv/train.csv"
tab_path_test = "CharCNN/data/ag_news_csv/test.csv"

process_text_agnews(tab_path_train, tab_path_test, \
                      path_to_save="../../Data/AGNews/agnews_texts", \
                      lower=True, max_vocab_size=100*1000,
                      doc_count_threshold=1)
# creates agnews_texts.npz and agnews_words.txt

# important: you should download file glove.840B.300d.txt from
# https://nlp.stanford.edu/projects/glove/
filter_large_embeddings_txt("glove.840B.300d.txt",\
                            "../../Data/AGNews/agnews_words.txt",\
                            path_to_save="../../Data/IMDB/glove_agnews.txt")
# creates glove_agnews.txt
