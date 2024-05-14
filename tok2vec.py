import tensorflow as tf
import numpy as np

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
        
class Tok2VecModel(tf.keras.Model):
    def __init__(self, vocab_size, vec_dim):
        super().__init__()
        self.embedding = Embedding(vec_dim)
        # self.embedding = tf.keras.layers.Dense(vec_dim, activation='tanh')
        self.dense = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def get_embeddings(self):
        return self.embedding.get_embeddings()
        # return self.embedding.get_weights()[0]

    def call(self, inputs):
        x = self.embedding(inputs)
        return self.dense(x)
    
        
def id_to_onehot(i, vocab_size):
    # onehot = np.zeros(vocab_size, dtype='uint8')
    onehot = np.zeros(vocab_size)
    onehot[i] = 1
    return onehot      

def tok2Vec(vocab_size, ids, window_size, vec_dim, epochs, filename=None):    
    # Create cbow sequences
    onehot = [id_to_onehot(i, vocab_size) for i in ids]

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
    n_train = int(0.9 * n)
    n_val = n - n_train

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_val = x[:n_val] 
    y_val = y[:n_val]

    # Create model

    model = Tok2VecModel(vocab_size, vec_dim)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_info = model.fit(x_train, y_train, 
                validation_data=(x_val, y_val),
                epochs=epochs,
                shuffle=True,
                verbose=1)
    model.summary()

    return model.get_embeddings(), train_info
        