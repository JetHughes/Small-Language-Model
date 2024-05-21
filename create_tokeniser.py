from transformer import *
from tokeniser import Tokeniser, plot_tok2vec
from load_text import load_prideandprejudice, load_warandpeace
from tok2vec import tok2Vec
import matplotlib.pyplot as plt

seq_len = 128    #Length of the input sequence to the transformer
vec_dim = 150    #Dimension of the embedding vectors
window_size = 8  #Size of the window for the tok2vec model
embedding_epochs = 40  #Number of epochs to train the embedding for
transformer_epochs = 5      #Number of epochs to train the transformer for
text_length = 800000 # / 121810 / 700000
vocab_size = 1500
embedding_type = "TOK2VEC" # BERT, TOK2VEC, ONEHOT
method = "skipgram"
dataset = "warandpeace" # warandpeace, prideandprejudice
load_embedding = True

tokeniser_filename= f'vocab/vocab_{vocab_size}_{dataset}.json'
tok2vec_savename = f'tok2vec_{vocab_size}_{vec_dim}_{text_length}_{window_size}_{embedding_epochs}_{method}_{dataset}'
transformer_savename = f'transformer_{seq_len}_{embedding_type}_{transformer_epochs}_{tok2vec_savename}'

# Load text for training  
print("Loading " + dataset + "...")
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
tokeniser.plot(text, ids)