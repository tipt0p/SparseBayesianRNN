### How to run code on AGNews

#### 1. Dowload data
```(bash)
python download_data.py
```

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
