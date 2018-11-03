import os
import sys
sys.path.append("../../Code/")
from classification_data_utils import process_text_agnews

os.system("git clone https://github.com/mhjabreel/CharCNN")

tab_path_train = "CharCNN/data/ag_news_csv/train.csv"
tab_path_test = "CharCNN/data/ag_news_csv/test.csv"

process_text_agnews(tab_path_train, tab_path_test, \
                      path_to_save="./agnews_texts", \
                      lower=True, max_vocab_size=100*1000,
                      doc_count_threshold=1)
