from tokeniser import Tokeniser
from load_text import load_prideandprejudice
import tensorflow as tf
import numpy as np
import sys
import os

class Embedding(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), 
            initializer='random_normal', 
            trainable=True)
        
    def get_embeddings(self):
        return self.w

    def call(self, inputs):
        return tf.nn.tanh(tf.matmul(inputs, self.w))
        
class tok2vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dimensions):
        super().__init__()
        self.embedding = Embedding(embedding_dimensions)
        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def get_embeddings(self):
        return self.embedding.get_embeddings()

    def call(self, inputs):
        x = self.embedding(inputs)
        return self.dense(x)
    
        
def id_to_onehot(i, vocab_size):
    onehot = np.zeros(vocab_size)
    onehot[i] = 1
    return onehot      

def Tok2Vec(vocab_size, ids, window_size, vec_dim, epochs, filename=None):    
    # Create cbow sequences
    onehot = [id_to_onehot(i, vocab_size) for i in ids]
    window_size

    if window_size % 2 != 0:
        raise ValueError("window_size must be even")

    x = []
    y = []

    for i in range(window_size//2, len(ids) - window_size//2):
        sequence = np.array(onehot[i - window_size // 2:i] + onehot[i+1:i + window_size // 2 + 1])
        x.append(sum(sequence))
        y.append(onehot[i])

    x = np.array(x)
    y = np.array(y)

    # train test val split
    n = len(x)
    n_train = int(0.8 * n)
    n_test = int(0.1 * n)
    n_val = n - n_train - n_test

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:n_train + n_test]
    y_test = y[n_train:n_train + n_test]

    x_val = x[:n_val] 
    y_val = y[:n_val]

    # Create model

    model = tok2vec(vocab_size, vec_dim)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, 
                validation_data=(x_val, y_val),
                epochs=epochs,
                shuffle=True,
                verbose=1)
    model.summary()

    return model.get_embeddings()
        