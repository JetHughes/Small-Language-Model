from transformer import *
from tokeniser import Tokeniser 
from load_text import load_prideandprejudice, load_warandpeace
import tensorflow as tf
import numpy as np
import os
from tok2vec import tok2Vec
import sys

seq_len = 10     #Length of the input sequence to the transformer
vec_dim = 150    #Dimension of the embedding vectors
window_size = 8  #Size of the window for the tok2vec model
embedding_epochs = 10  #Number of epochs to train the embedding for
epochs = 10      #Number of epochs to train the transformer for
text_length = 120000 # / 121810 / 700000
vocab_size = 1000
embedding = "CUSTOM" # BERT, CUSTOM
dataset = "prideandprejudice" # warandpeace, prideandprejudice
load_embedding = False

tokeniser_filename= f'vocab_{str(vocab_size)}_{str(dataset)}.json'
embedding_filename = f'tok2vec_{str(vocab_size)}_{str(vec_dim)}_{str(text_length)}_{str(window_size)}_{str(embedding_epochs)}'

# Load text for training  
if dataset == "prideandprejudice":
    text = load_prideandprejudice(max_words=text_length)
elif dataset == "warandpeace":
    text = load_warandpeace(max_words=text_length)

# Train/Load tokeniser
if os.path.exists(tokeniser_filename):
    print("Loading tokeniser from '%s'..." % (tokeniser_filename))
    tokeniser = Tokeniser.load(tokeniser_filename)
else:
    # Create a new tokeniser, train it on the text and save it to disk
    tokeniser = Tokeniser(vocab_size=vocab_size)
    print("Building BPE tokeniser...")
    tokeniser.train(text, verbose=True)
    print("Saving tokeniser to '%s'..." % (tokeniser_filename))
    tokeniser.save(tokeniser_filename)
print("Converting training text to tokens...")
ids = tokeniser.encode(text, verbose=True)  

# Train/Load the embedding
print("Training/Loading embedding...")
w,_ = tok2Vec(vocab_size, ids, window_size, vec_dim, embedding_epochs, savename=embedding_filename, load_from_file=load_embedding)

# Create a data generator
print("Loading data generator...")
train_data = predictTextDataGenerator(ids=ids, seq_len=seq_len, batch_size=32)

# Create a new sequential model
model = tf.keras.models.Sequential()
model.add(FixedEmbedding(w, seq_len))
# model.add(OneHotEmbedding(vocab_size, seq_len))
model.add(PositionalEncoding(vec_dim=vec_dim, seq_len=seq_len))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))
model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

learning_rate = CustomSchedule(vec_dim)
opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)

model.compile(optimizer=opt,
                loss=masked_loss,
                metrics=[masked_accuracy])
model.summary()
model.fit(train_data, epochs=epochs)

prompt = "It is a truth universally acknowledged"

print(prompt, end='')
sys.stdout.flush()

# Encode prompt to tokens
tokens = tokeniser.encode(prompt)

for i in range(1,100):
    # Check if prompt is more than seq_len, if so, truncate, grabbing the
    # last seq_len tokens
    if len(tokens) >= seq_len:
        tokens = tokens[-seq_len:]
    # Index of the last token, which is going to be the 
    # index of the output stream that we are going to use for prediction
    j = len(tokens)-1

    # If the prompt is less than seq_len, pad it with zeros
    if len(tokens) < seq_len:
        x = np.concatenate([tokens,np.zeros((seq_len-len(tokens)),dtype='int')], axis=0)
    else:
        x = np.array(tokens)

    # Since the transformer expect input to be of shape (num_examples, seq_len), and
    # at this point x is just a vector of seq_len integers, we need to add a dimension
    # to change x to a tensor of shape (1, seq_len)     
    x = np.expand_dims(x,axis=0)

    # Compute output of the transformer
    y = model.predict(x,verbose=False)
    # The output will be of dmension (1, seq_len, vocab_size), but we are only interested in
    # the token that follow the prompt, at position j in the output stream.  
    # And so y[:,j,:] is a (1, vocab_size) tensor of probabilities of the next token in the sequence.
    # and we want to find the token with the highest probability.
    y = np.argmax(y[:,j,:])
    
    # Decode the token back to text
    t = tokeniser.decode(y)
    # Print it
    print(t, end='')
    sys.stdout.flush()
    # Apend the token (integer) to the prompot tokens
    tokens.append(y)

print("\n")


