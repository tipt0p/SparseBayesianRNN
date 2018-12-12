### How to run code on AGNews

#### 1. Download data and pretrained embeddings
Unpack an [archive with the data and GloVe embeddings](https://drive.google.com/file/d/1U8eoySFDnLzCtqMGm_PK-TbGfwyMPupL/view?usp=sharing) to ../../Data folder.

We also release the code which was used to obtain these files: prepare_data.py

#### 2. Train model
```(bash)
python main.py CONFIG
```
CONFIG is a 6-chars string (see [main README](https://github.com/ars-ashuha/variational-dropout-rnn/blob/EMNLP_clean/README.md) for details)
During training, accuracy and compression rates are printed to the output stream.

#### (optional) 3. Compress model
```(bash)
python run_compression.py
```
This code performs actual compression of the weights and evaluates network with compressed weights.  
