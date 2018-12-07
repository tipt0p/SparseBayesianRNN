import numpy as np

def load_PTB_char(ptb_path, fname_train, fname_val, fname_test):
    with open(ptb_path + fname_train,"r") as fin:
        train = fin.read()[::2]
    with open(ptb_path + fname_val,"r") as fin:
        val = fin.read()[::2]
    with open(ptb_path + fname_test,"r") as fin:
        test = fin.read()[::2]
    
    chars = list(set(train + val + test))
    VOCAB_SIZE = len(chars)
    char_to_id = { ch:id for id,ch in enumerate(chars) }
    id_to_char = { id:ch for id,ch in enumerate(chars) }
    
    train = [char_to_id[ch] for ch in train]
    val = [char_to_id[ch] for ch in val]
    test = [char_to_id[ch] for ch in test]
    return np.array(train, dtype = 'int32'), np.array(val, dtype = 'int32'), np.array(test, dtype = 'int32')

class PTB_char:
    def __init__(self, data, length, batch_size, augment=True):
        self.length = length
        self.augment = augment
        self.batch_size = batch_size
        self.data = data
        self.num_examples = int(len(self.data) / self.length)
        if self.augment:
            # -1 so we have one self.length worth of room for augmentation
            self.num_examples -= 1
        
        self.num_batches = int(np.ceil(self.num_examples/float(batch_size)))
        self.max_offset = len(self.data) - self.num_examples * self.length
        self.batch_ind = 0
        self.inds = np.arange(0,self.num_examples) * self.length
        self.inds = [(self.inds[(i*batch_size):min(self.num_examples, (i+1)*batch_size)]) 
                                for i in range(0,self.num_batches)]
    
    def new_epoch(self):
        offset = 0
        if self.augment:
            offset = np.random.randint(self.max_offset)
        self.inds = np.arange(0,self.num_examples) * self.length + offset
        np.random.shuffle(self.inds)
        self.inds = [(self.inds[(i*self.batch_size):min(self.num_examples, (i+1)*self.batch_size)]) 
                                for i in range(0,self.num_batches)]
        self.batch_ind = 0
        
    def to_first_batch(self):
        self.batch_ind = 0
    
    def get_next_batch(self):
        self.batch_ind += 1
        batch = np.array([self.data[i:i+self.length] for i in self.inds[self.batch_ind-1]])
        return batch[:,:-1],batch[:,1:]

    
def load_PTB_word(ptb_path, fname_train, fname_val, fname_test):
    def read_words(fname):
        with open(ptb_path + fname,"r") as fin:
            data = fin.read()[::2]
        data_long = []
        for l in data.split():
            data_long.extend(l.split('_'))
            data_long.append('<eos>')
        return data_long
     
    train, val, test = read_words(fname_train),read_words(fname_val),read_words(fname_test)
    
    words = list(set(train + val + test))
    VOCAB_SIZE = len(words)
    print(VOCAB_SIZE)
    word_to_id = { w:id for id,w in enumerate(words) }
    id_to_word = { id:w for id,w in enumerate(words) }
    
    
    train = [word_to_id[w] for w in train]
    val = [word_to_id[w] for w in val]
    test = [word_to_id[w] for w in test]
    return np.array(train, dtype = 'int32'), np.array(val, dtype = 'int32'), np.array(test, dtype = 'int32')

class PTB_word:
    def __init__(self, data, length, batch_size):
        self.length = length
        self.batch_size = batch_size
        self.data = data
        self.data = self.data[:self.batch_size*int(len(self.data) / self.batch_size)]
        self.data = self.data.reshape((self.batch_size,-1)).T
        self.batch_ind = 0
        self.num_batches = int(np.ceil(len(data)/float(batch_size)/float(self.length)))
    
    def new_epoch(self):
        self.batch_ind = 0
        
    def to_first_batch(self):
        self.batch_ind = 0
    
    def get_next_batch(self):
        seq_len = min(self.length, len(self.data) - self.batch_ind)
        batch = self.data[self.batch_ind:self.batch_ind+seq_len+1]
        self.batch_ind += seq_len
        return batch[:-1].T,batch[1:].T