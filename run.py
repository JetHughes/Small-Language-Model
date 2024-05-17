from tokeniser import Tokeniser
from load_text import load_prideandprejudice, load_warandpeace
from tok2vec import tok2Vec
from transformer import *
import sys
import os

# Parameters for tok2vec
vocab_size = 1000 #Size of the vocabulary
vec_dim = 100
window_size = 8
epochs = 10
text_length = 50000
dataset = "warandpeace"
method = "cbow"

# Parameters for transformer
seq_len = 128    

# Load text
tokeniser_savename = f'vocab_{str(vocab_size)}_{str(dataset)}.json'
text = load_prideandprejudice(text_length)

# Check if tokeniser has been saved to disk
if os.path.exists(tokeniser_savename):
    # Load tokeniser from disk
    print("Loading tokeniser from '%s'..." % (tokeniser_savename))
    tokeniser = Tokeniser.load(tokeniser_savename)
else:
    # Create a new tokeniser, train it on the text and save it to disk
    tokeniser = Tokeniser(vocab_size=vocab_size)
    print("Building BPE tokeniser...")
    tokeniser.train(text, verbose=True)
    print("Saving tokeniser to '%s'..." % (tokeniser_savename))
    tokeniser.save(tokeniser_savename)

ids = tokeniser.encode(text, verbose=True)
# tokeniser.plot(ids=ids) 

# Train/Load tok2vec model
print("Training tok2vec model...")
tok2vec_savename = f'tok2vec_{vocab_size}_{vec_dim}_{text_length}_{window_size}_{epochs}_{method}_{dataset}'
w, history = tok2Vec(vocab_size, ids, window_size, vec_dim, epochs, method=method, savename=tok2vec_savename)

# Train transformer model
print("Training transformer model...")

train_data = predictTextDataGenerator(ids=ids, seq_len=seq_len, batch_size=32)

# Create a new sequential model
model = tf.keras.models.Sequential()
model.add(FixedEmbedding(w, seq_len))
# model.add(OneHotEmbedding(vocab_size, seq_len))
model.add(PositionalEncoding(vec_dim=vec_dim, seq_len=seq_len))
model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))
# model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))
# model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))
# model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))
# model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))
# model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))
model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

learning_rate = CustomSchedule(vec_dim)
opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model.compile(optimizer=opt,
                loss=masked_loss,
                metrics=[masked_accuracy])
model.summary()
model.fit(train_data, epochs=epochs)
# model.save(f'transformer_{seq_len}_{tok2vec_savename}.h5')

prompt = "Well, Prince, so Genoa and Lucca are now just family estates of the"

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
