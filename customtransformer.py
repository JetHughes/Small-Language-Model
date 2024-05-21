from transformer import *
from tokeniser import Tokeniser 
from load_text import load_prideandprejudice, load_warandpeace
import os
from tok2vec import tok2Vec

seq_len = 128    #Length of the input sequence to the transformer
vec_dim = 150    #Dimension of the embedding vectors
vec_dims = [50, 100, 150, 200, 250, 300]
window_size = 8  #Size of the window for the tok2vec model
embedding_epochs = 5  #Number of epochs to train the embedding for
transformer_epochs = 5      #Number of epochs to train the transformer for
text_length = 800000 # / 121810 / 700000
vocab_size = 2000
embedding_type = "TOK2VEC" # BERT, TOK2VEC, ONEHOT
method = "skipgram"
dataset = "warandpeace" # warandpeace, prideandprejudice
load_embedding = True


for embedding_epochs in [5, 10, 20]:
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

    if embedding_type == "TOK2VEC":
        # Train/Load the embedding
        print("Training/Loading embedding...")
        embedding,_ = tok2Vec(vocab_size, ids, 
                    window_size, vec_dim, 
                    embedding_epochs, 
                    method=method, 
                    savename=tok2vec_savename, 
                    load_from_file=load_embedding)
        print("Embedding shape: " + str(embedding.shape))
    else:
        embedding = None

    n = len(ids)
    train_ids = ids[:int(0.9*n)]
    valid_ids = ids[int(0.9*n):]

    # Create a data generator
    print("Loading data generator...")
    train_data = predictTextDataGenerator(ids=train_ids, seq_len=seq_len, batch_size=32)
    valid_data = predictTextDataGenerator(ids=valid_ids, seq_len=seq_len, batch_size=32)

    transformer = Transformer(seq_len, embedding, embedding_type, embedding_shape=(vocab_size, vocab_size))
    transformer.train(train_data, valid_data, transformer_epochs)
    transformer.save(transformer_savename)
    # transformer = Transformer.load(f'transformer_{seq_len}_{embedding_type}_{transformer_epochs}_{tok2vec_savename}')
    # transformer.predict("It is a truth universally acknowledged", tokeniser)


for vec_dim in vec_dims:
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

    if embedding_type == "TOK2VEC":
        # Train/Load the embedding
        print("Training/Loading embedding...")
        embedding,_ = tok2Vec(vocab_size, ids, 
                    window_size, vec_dim, 
                    embedding_epochs, 
                    method=method, 
                    savename=tok2vec_savename, 
                    load_from_file=load_embedding)
        print("Embedding shape: " + str(embedding.shape))
    else:
        embedding = None

    n = len(ids)
    train_ids = ids[:int(0.9*n)]
    valid_ids = ids[int(0.9*n):]

    # Create a data generator
    print("Loading data generator...")
    train_data = predictTextDataGenerator(ids=train_ids, seq_len=seq_len, batch_size=32)
    valid_data = predictTextDataGenerator(ids=valid_ids, seq_len=seq_len, batch_size=32)

    transformer = Transformer(seq_len, embedding, embedding_type, embedding_shape=(vocab_size, vocab_size))
    transformer.train(train_data, valid_data, transformer_epochs)
    transformer.save(transformer_savename)
    # transformer = Transformer.load(f'transformer_{seq_len}_{embedding_type}_{transformer_epochs}_{tok2vec_savename}')
    # transformer.predict("It is a truth universally acknowledged", tokeniser)
