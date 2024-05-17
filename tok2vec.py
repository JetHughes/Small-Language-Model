import tensorflow as tf
import numpy as np
import gzip
import pickle
import os

def get_onehot(i, vocab_size):
    onehot = np.zeros(vocab_size, dtype='uint8')
    onehot[i] = 1
    return onehot      

class tok2vecDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, window_size, vocab_size, batch_size, method="cbow"):
        # Save all the training text and parameters of the data generator
        self.ids = ids          
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.method = method

        if self.method == "skipgram":
            self.batch_size = self.batch_size * self.window_size-1

        # Compute the number of samples - it's the length of the text minus the window size
        self.num_samples = len(self.ids)-window_size-1
        self.on_epoch_end()

    def __len__(self):
        return self.num_samples // self.batch_size

    def __data_generation(self, list_IDs_temp):
        if self.window_size % 2 != 0:
            raise ValueError("window_size must be even")

        X = np.zeros((self.batch_size, self.vocab_size),dtype='int')
        y = np.zeros((self.batch_size, self.vocab_size),dtype='int')

        for i, ID in enumerate(list_IDs_temp):
            window = [get_onehot(i, self.vocab_size) for i in self.ids[ID:ID+self.window_size+1]]
            target = get_onehot(self.ids[ID+self.window_size//2], self.vocab_size)
            if self.method == "cbow":                
                X[i] = sum(window)-target
                y[i] = target
            elif self.method == "skipgram":                
                for token in window:
                    if not np.array_equal(token, target):
                        X[i] = token
                        y[i] = target

        return X, y            
    
    def __getitem__(self, index):        
        # Generate indexes of the batch
        # If method is skipgram the batch size is multiplied by the window size
        list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]

        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        # Shuffle the tokens
        self.list_IDs = np.arange(self.num_samples)
        np.random.shuffle(self.list_IDs)

def get_tok2vec_model(vocab_size, vec_dim):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(vec_dim, activation='tanh', use_bias=False, input_shape=(vocab_size,)))
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax', use_bias=False))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def tok2Vec(vocab_size, ids, window_size, vec_dim, epochs, method="cbow", savename=None, load_from_file=True):   
    # Load the model from file if it exists"cbow", 
    if savename and load_from_file:
        model_save_name = savename + ".h5"
        history_save_name = "history_" + savename + ".hist"
        if os.path.exists(model_save_name) and os.path.exists(history_save_name):
            model = tf.keras.models.load_model(model_save_name)
            with gzip.open(history_save_name) as f:
                history = pickle.load(f)
            return model.get_weights()[0], history
    
    # train val split
    n_train = (int)(0.9 * len(ids))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    # Create cbow sequences
    train = tok2vecDataGenerator(train_ids, window_size, vocab_size, batch_size=32, method=method)
    val = tok2vecDataGenerator(val_ids, window_size, vocab_size, batch_size=32, method=method)

    # Create model
    model = get_tok2vec_model(vocab_size, vec_dim)

    # Train model
    train_info = model.fit(train, 
                validation_data=(val),
                epochs=epochs,
                shuffle=True,
                verbose=1)
    model.summary()

    if savename:
        model.save(model_save_name, save_format='h5')
        with gzip.open(history_save_name, 'w') as f:
            pickle.dump(train_info.history, f)

    return model.get_weights()[0] , train_info.history