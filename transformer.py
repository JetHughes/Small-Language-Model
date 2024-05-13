__author__ = "Lech Szymanski"
__organization__ = "COSC420, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import tensorflow as tf
import numpy as np

''' 
Helper function for building and training the transformer model.  These methods follow 
tutorial from https://www.tensorflow.org/text/tutorials/transformer, though it
has been modified to be simpler and more modular, so that you can just use these
custom layer in a sequential keras model just like you used convolutional and dense
layers in the previous assignments.  The tutorial is a good reference for understanding
the transformer model, but the code is more complex than it needs to be for this assignment.
'''

def masked_loss(label, pred):
  '''
  Compute the masked loss for the transformer model.  This loss function is better than
  using the sparse categorical crossentropy loss function because it ignores the padding
  tokens when computing the loss.  The padding tokens are the zeros in the label tensor.
  
  Import this loss function into your code and use it in the model.compile() method like so:

  from transformer import masked_loss
  model.compile(... loss=masked_loss, ...)
  '''
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  '''
  Compute the masked accuracy for the transformer model.  This accuracy function is better than
  standard accuracy because it ignores the padding tokens when computing the accuracy.  The padding
  tokens are the zeros in the label tensor.

  Import this accuracy function into your code and use it in the model.compile() method like so:

  from transformer import masked_accuracy
  model.compile(... metrics=[masked_accuracy], ...)

  '''  
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

class OneHotEmbedding(tf.keras.layers.Layer):
  ''' 
  This is a custom layer that converts integer tokens to one-hot vectors.  This is useful
  for the input layer of the transformer model, where the input is a sequence of integers
  representing tokens from the vocabulary.  The output of this layer is a sequence of one-hot
  vectors, where each one-hot vector represents a token from the vocabulary.
  
  Import this layer into your code and use it in the model like so:

  from transformer import OneHotEmbedding
  model = tf.keras.Sequential([  
    OneHotEmbedding(vocab_size, seq_len),
    ...
  ])

  where vocab_size is the size of the vocabulary and seq_len is the length of the input sequence.
  '''

  def __init__(self, vocab_size, seq_len):
    super().__init__(input_shape=(seq_len,))
    self.vocab_size = vocab_size

  def compute_output_shape(self, input_shape):
        return (input_shape[0], self.vocab_size)

  def call(self, x):
    x = tf.cast(x,dtype=tf.int64)
    y = tf.one_hot(x, depth = self.vocab_size)
    return y    

class FixedEmbedding(tf.keras.layers.Layer):
  '''
  This is a custom layer that uses a fixed, pretrained word embedding to convert integer tokens
  to word vectors.  This is useful for the input layer of the transformer model, where the
  input is a sequence of integers representing tokens from the vocabulary.  The output of this
  layer is a sequence of word vectors, where each word vector represents a token from the vocabulary.

  Import this layer into your code and use it in the model like so:

  from transformer import FixedEmbedding
  model = tf.keras.Sequential([
    FixedEmbedding(w, seq_len),
    ...
  ])  
  
  where w is a numpy array of shape (vocab_size, vec_dim) containing the pretrained word vectors,
  and seq_len is the length of the input sequence.
  '''
  def __init__(self, w, seq_len):
    super().__init__(input_shape=(seq_len,))
    self.vocab_size, self.vec_dim = w.shape
    self.w = tf.constant(w, dtype=tf.float32)

  def compute_output_shape(self, input_shape):
        return (input_shape[0], self.vec_dim)

  def call(self, x):
    x = tf.cast(x,dtype=tf.int64)
    y=tf.gather(self.w,x)
    return y    

class TrainableEmbedding(tf.keras.layers.Embedding):
  '''
  This is a custom layer that learns word embeddings from the data.  This is useful for the input
  layer of the transformer model, where the input is a sequence of integers representing tokens from
  the vocabulary.  The output of this layer is a sequence of word vectors, where each word vector
  represents a token from the vocabulary.  In this case however, the original embedding is random
  and it's modified during transformer's training.

  Import this layer into your code and use it in the model like so:

  from transformer import TrainableEmbedding
  model = tf.keras.Sequential([
    TrainableEmbedding(vocab_size, vec_dim),
    ...
  ])

  where vocab_size is the size of the vocabulary and vec_dim is the dimension of the word vectors.
  '''
  def __init__(self, vocab_size, vec_dim):
    super().__init__(vocab_size, vec_dim, mask_zero=True)


class PositionalEncoding(tf.keras.layers.Layer):
  '''
  This is a custom layer that adds positional encoding to the input sequence.  This is useful for the
  input layer of the transformer model, where the input is a sequence of word vectors.  The output of this
  layer is a sequence of word vectors with positional encoding added to them.
  
  Import this layer into your code and use it in the model like so:

  from transformer import PositionalEncoding
  model = tf.keras.Sequential([
    <embedding_layer>,
    PositionalEncoding(vec_dim, seq_len),
    ...
  ])

  where vec_dim is the dimension of the word vectors and seq_len is the length of the input sequence.
  '''
  def __init__(self, vec_dim, seq_len):
    super().__init__()
    self.vec_dim = vec_dim
    self.pos_encoding = tf.cast(PositionalEncoding.positional_encoding(seq_len, vec_dim), dtype=tf.float32)

  @staticmethod
  def positional_encoding(seq_len, vec_dim):
    depth = vec_dim
    depth = depth//2

    positions = np.arange(seq_len)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1) 

    return pos_encoding


  def call(self, x, mask=None):
    length = tf.shape(x)[1]
    # This factor sets the relative scale of the embedding and positional_encoding.
    x *= tf.math.sqrt(tf.cast(self.vec_dim, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class MultiheadSelfAttention(tf.keras.layers.Layer):
  '''
  This is a custom layer that performs self-attention on the input sequence.  This is useful for the
  transformer layer of the transformer model, where the input is a sequence of word vectors with positional
  encoding added to them.  The output of this layer is a sequence of word vectors with self-attention applied
  to them.

  You don't need to import this layer into your code, because it's used by the TransformerLayer class below.
  '''

  def __init__(self, num_heads, key_dim):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=key_dim)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
    self.attn_scores = None


  def call(self, x):

    attn_output, attn_scores = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True,
        return_attention_scores=True)
    self.attn_scores = attn_scores
    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x


class FeedForward(tf.keras.layers.Layer):
  '''
  This is a custom layer that performs feed-forward transformation on the input sequence.  This is useful
  for the transformer layer of the transformer model, where the input is a sequence of word vectors with
  self-attention applied to them.  The output of this layer is a sequence of word vectors with feed-forward
  transformation applied to them.
  
  You don't need to import this layer into your code, because it's used by the TransformerLayer class below.
  '''
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class TransformerLayer(tf.keras.layers.Layer):
  '''
  This is a custom layer that combines self-attention and feed-forward transformation.  This is useful for
  the transformer model, where the input is a sequence of word vectors with positional encoding added to them.
  The output of this layer is a sequence of word vectors with self-attention and feed-forward transformation
  applied to them.
  
  Import this layer into your code and use it in the model like so:
  
  from transformer import TransformerLayer
  
  model = tf.keras.Sequential([
    ...
    TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=2048),
    ...
  ])

  where vec_dim is the dimension of the word vectors, key_dim is the dimension of the key/value/query vectors used in
  the self-attention mechanism, num_heads is the number of heads in the multi-head attention mechanism, and dff is the
  dimension of the feed-forward network in the transformer layer.'''

  def __init__(self,
               vec_dim,
               key_dim,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(TransformerLayer, self).__init__()

    self.self_attention = MultiheadSelfAttention(
        num_heads=num_heads,
        key_dim=key_dim)

    self.ffn = FeedForward(vec_dim, dff, dropout_rate=dropout_rate)

  def call(self, x):
    x = self.self_attention(x=x)

    # Cache the last attention scores for plotting later
    self.attn_scores = self.self_attention.attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
  

if __name__ == '__main__':
    '''
    This is an example of how to build and train your transformer model.  This example uses the pretrained
    BERT tokeniser and embedding.  For this to work you need to install the transformers and tf-keras 
    librariries in our cosc420 environment.  You can do that by running the following commands:

    conda activate cosc420
    pip install transformers
    pip install tf-keras    
    '''

    from tokeniser import Tokeniser
    from load_text import load_prideandprejudice
    import sys

    ''' 
    Rather than feeding just training data as in the previous assignments, for this assignment it's
    best to use a custom written data generator.  This is a way for you to contol how the batches
    of training data are created.  Here's a really simple data generator, that, in an epoch, randomly
    picks words from text and creates a batch of training data of input and target output sequences of
    fixed length.
    '''
    class predictTextDataGenerator(tf.keras.utils.Sequence):
    
      def __init__(self, ids, seq_len,batch_size):
          '''
          Constructor for the data generator.  

          param ids: A list of integers representing the tokens in the training text.
          param seq_len: The length of the input and target sequences for the transformer model.
          param batch_size: The number of sequences in each batch for training          
          '''

          # Save all the training text and parameters of the data generator
          self.ids = ids          
          self.seq_len = seq_len
          self.batch_size = batch_size

          # Compute the number of samples - it's the length of the text minus the sequence length
          self.num_samples = len(self.ids)-seq_len-1
          # Run the on_epoch_end() method - which scrambles the data into the batchs
          # (this method will also be run during trainin at the end of each training epoch)
          self.on_epoch_end()

      def __len__(self):
          '''
          You must provide this method to tell the model how many batches there are in an epoch.

          returns The number of batches in an epoch.
          '''
          return self.num_samples // self.batch_size

      def __data_generation(self, list_IDs_temp):
          '''
          This method generates a batch of training data for the model.  It's called by the
          __getitem__() method which is called by the model during training.

          param list_IDs_temp: A list of integers representing the indexes of the training data
          to be included in the batch.
          returns A tuple of input and target output sequences for the model.
          '''

          # The input and target sequences are both of shape (batch_size, seq_len) and
          # are integer ids of the tokens (the transformer model will convert these to word vectors based
          # on the embedding you specify)
          X = np.zeros((self.batch_size, self.seq_len),dtype='int')
          y = np.zeros((self.batch_size, self.seq_len),dtype='int')

          # For each index in the list of indexes...
          for i, ID in enumerate(list_IDs_temp):
              #...get the sequence of tokens from the training of length seq_len starting at
              #index ID.  In this case the input sequence is the sequence spans the entire
              #length of seq_len, but you might also train on shorter sequences, padded with zeros.
              #makse_loss will included padded inputs/outputs.
              X[i,:seq_len] = self.ids[ID:ID+seq_len]
              #....and the sequence of target tokens, which is the sequence of tokens from the
              #training text of length seq_len starting at index ID+1 (offset by one, to match
              #the next word in the output to current word in the input)
              y[i,:seq_len] = self.ids[ID+1:ID+seq_len+1]


          return X, y

      def __getitem__(self, index):
          '''
          This method is called by the model during training to get a batch of training data.
          
          param index: The index of the batch to get.
          returns A tuple of input and target output sequences for the model.
          '''
          
          # Generate indexes of the batch
          list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]

          # Generate data
          X, y = self.__data_generation(list_IDs_temp)

          return X, y

      def on_epoch_end(self):
          '''
          This method is called at the end of each epoch of training.  It shuffles the data
          so that the batches are different in each epoch.
          '''
          
          # Shuffle the tokens
          self.list_IDs = np.arange(self.num_samples)
          np.random.shuffle(self.list_IDs)

    # Custom learning rate schedule for the transformer model - taken directly from
    # https://www.tensorflow.org/text/tutorials/transformer
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
      def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

      def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # Load text for training
    text = load_prideandprejudice(max_words=1000)
 
    seq_len = 10     #Length of the input sequence to the transformer
    vec_dim = 768    #Dimension of the embedding vectors

    epochs = 2       #Number of epochs to train for

    # This loads both the tokeniser and the pretrained BERT embedding, for your own
    # embedding you will have separate tokeniser and embedding loaders.
    tokeniser = Tokeniser.load_bert()

    # Conver text to token ids
    print("Converting training text to tokens...")
    ids = tokeniser.encode(text)

    # Create a data generator
    print("Loading data generator...")
    train_data = predictTextDataGenerator(ids=ids, seq_len=seq_len, batch_size=32)

    # Get the vocabulary sice of the tokeniser
    vocab_size = tokeniser.vocab_size

    # Create a new sequential model
    model = tf.keras.models.Sequential()

    # Fetch the (vocab_size, vec_dim)-shape embedding matrix for the BERT tokeniser,
    # for your own embedding will have to fetch the embedding matrix from your tok2vec
    # model
    w = tokeniser.get_embedding()

    # The first layer of the model is the embedding layer.  The fixed embedding is conveyed
    # in the w argument passed in, which is a numpy array of shape (vocab_size, vec_dim).  You
    # also need to specify the seq_len of your input sequence.  The input then is a tensorf of
    # shape (num_examples, seq_len) of integers representing tokens from the vocabulary; the
    # output is a (num_examples, seq_len, vec_dim) tensor of word vectors.
    model.add(FixedEmbedding(w, seq_len))

    # Positional endcoding is added to the embedding. This layer needs to know the vec_dim of
    # the embedding space and the seq_len of the input sequence.  The input is a tensor of shape
    # (num_examples, seq_len, vec_dim) of word vectors; the output is the same shape with positional 
    # encoding added to the word vectors, of shape (num_examples, seq_len, vec_dim).
    model.add(PositionalEncoding(vec_dim=vec_dim, seq_len=seq_len))

    # The transformer layer is added to the model.  This layer needs to know the vec_dim of the
    # word vectors, the key_dim of the key/value/query vectors used in the self-attention mechanism,
    # the number of heads in the multi-head attention mechanism, and the dimension of the feed-forward
    # network in the transformer layer.  The input is a tensor of shape (num_examples, seq_len, vec_dim)
    # of word vectors with positional encoding added to them; the output is of shape (num_examples, seq_len, vec_dim).  You can have sever transformer layers in the model, just like you can have several dense or convolutional layers.
    model.add(TransformerLayer(vec_dim=vec_dim, key_dim=32, num_heads=8, dff=256))

    # The final dense layer of the netowork is added.  This layer has a softmax activation function and
    # outputs a tensor of shape (num_examples, seq_len, vocab_size) of probabilities of the next token in
    # the sequence for each position in the input.  The input is a tensor of shape (num_examples, seq_len, vec_dim); the output is of shape (num_examples, seq_len, vocab_size).
    model.add(tf.keras.layers.Dense(vocab_size, activation='softmax'))

    # This custom learning schedule (which varies the learning rate) is much better to use than
    # a fixed learning rate. 
    learning_rate = CustomSchedule(vec_dim)
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

    # Complie the modle.  The masked_loss and maske_accuracy (provided at the top of this file) are
    # better than the standard loss and accuracy functions because they ignore the padding tokens in
    # your target outputs...so that, if your input and target output don't span the entire seq_len,
    # they should be padded with zeros, and the loss and accuracy functions will ignore those zeros.
    model.compile(optimizer=opt,
                  loss=masked_loss,
                  metrics=[masked_accuracy])

    # Show the archtiecture of the model
    model.summary()

    # Train the model
    model.fit(train_data, epochs=epochs)

    # Test the model by generating text that follows this prompt
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


    

