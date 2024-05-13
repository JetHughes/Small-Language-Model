__author__ = "Lech Szymanski"
__organization__ = "COSC420, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import re
import progressbar
import os  
import numpy as np
import contextlib
from collections import OrderedDict
import json

''' Test if transformers package is available (you can add it to cosc420 environmpent with 'pip install transformers tf-keras')'''
try: 
    from transformers import BertTokenizer, TFBertModel
    bert_available = True
except:
    bert_available = False

def clean_text(text):
    '''
    clean_text cleans up the text by removing unwanted characters and standardising
    spaces between non-alphanumeric characters
    returns the cleaned text
    '''
    # List of text to replace
    replacements = [(' ##', ''), ('[CLS] ', ''), (' [SEP]', ''), ('[PAD]', ''), ('[UNK]', ''), ('— — \' s', '——\'s'), (' ”', '”'), ('“ ', '“'), (' \'', '\''), (' - ', '-'), ('\' t ','\'t '), ('\' s ','\'s '), ('\' ve ','\'ve '),('\' ll ','\'ll '), ('\' re ','\'re '), ('\' m ','\'m '), ('\' d ','\'d '), ('\' ', '\''),  (' ‘ ', ' ‘'), (' ( ', ' ('), (' ) ', ') '),('“‘ ', '“‘'), (' — ', '—'), ('“— ', '“—'), ('‘ Y', '‘Y') ]
    #Add to replacements all space followed by punctuation
    for p in ['.',',','!','?',';',':']:
        replacements.append((' '+p,p))
    for p in ['-', '—']:
        replacements.append((' '+p,p))

    # Make the replacements
    for r in replacements:
        text = text.replace(r[0],r[1])

    # Precede all dashes and slaspehs with a space and follow it with a space
    text = re.sub("-", " - ", text)
    text = re.sub("—", " — ", text)
    text = re.sub("/", " / ", text)

    # Follow any non-alpha numeric character with a space
    text = re.sub("([^\w\s])", r'\1 ', text)

    return text

class Tokeniser:
    ''' Tokeniser class, converts text to tokens and back to text
    
    The Tokeniser class can be trained on a text to create a BPE tokeniser 
    
    The Tokeniser class can also be used to load previously created tokeniser
    that has been saved to disk.

    The Tokeniser class can also be loaded with a pre-trained BERT tokeniser + embedding
    
    Useful variables:
        vocab_size: number of tokens in the tokeniser
        
    Useful methods:
        train(text): train the tokeniser on a text
        encode(text): convert text to tokens
        decode(tokens): convert tokens to text
        save(filename_base): save the tokeniser to disk
        load(filename_base): load the tokeniser from disk
        load_bert(): load a pre-trained BERT tokeniser + embedding
        plot(text=None,ids=None): plot the token distribution in the text        
    '''

    def __init__(self, vocab_size=1000):
        '''
        Constructor for the Tokeniser class

        :param vocab_size: max number of tokens in the tokeniser if training
        '''
        self.vocab_size=vocab_size
        self.word_index = {}
        self.bert_model = None

    def save(self,filename):
        '''
        Save the tokeniser to disk

        :param filename: filename for saving the tokeniser in json format

        '''
        if not self.word_index:
            raise RuntimeError("Error: tokeniser's word_index is empty")

        state = { 'word_index': self.word_index, 'merge': self.merge }

        with open(filename, 'w') as f:
            json.dump(state, f)


    @staticmethod
    def load(filename):
        '''
        Load the tokeniser from disk, this is a static method that returns Tokeniser instance.
        
        param filename: json file that contains tokeniser state
        returns Tokeniser instance
        '''

        with open(filename, 'r') as file:
            state = json.load(file, object_pairs_hook=OrderedDict)
        word_index = state['word_index']
        merge = state['merge']

        tokeniser = Tokeniser(vocab_size=len(word_index))
        tokeniser.word_index = word_index
        tokeniser.merge = merge

        return tokeniser

    @staticmethod
    def load_bert():
        '''
        Load a pre-trained BERT tokeniser + embedding, this is a static method that returns Tokeniser instance.

        returns Tokeniser instance
        '''
        if not bert_available:
            raise RuntimeError("Error: transformers package is not installed.  Please install transformers using 'pip install transformers'")

        tokeniser = Tokeniser()

        tokeniser.bert_tokeniser = BertTokenizer.from_pretrained("bert-base-cased")
        tokeniser.bert_model = TFBertModel.from_pretrained("bert-base-cased")
        tokeniser.vocab_size = tokeniser.bert_model.config.vocab_size
        tokeniser.word_index = tokeniser.bert_tokeniser.vocab
        tokeniser.pretrained = True

        return tokeniser


    def train(self, text, verbose=True):
        '''
        Train the tokeniser on a text, this method creates a BPE tokeniser custom for the
        text you train it on.

        :param text: text to train the tokeniser on
        :param verbose: if True, show progress bar

        '''
        self.merge = []

        if self.bert_model is not None:
            raise RuntimeError("Error: tokeniser is pretrained BERT and cannot be retrained")

        # Extract all non alphanumeric characters from string text and save them into a list
        non_alphanumeric = list(set([char for char in text if not char.isalnum() and char != ' ']))

        #Split text on space and non alphanumeric characters, but keep the split characters in the list
        tokens = re.split('(\W)', text)

        # Remove empty tokens
        new_tokens = []
        for i in range(len(tokens)):
            if len(tokens[i])==0:
                continue
            else:
                new_tokens.append(tokens[i])
        tokens = new_tokens
    
        # Merge tokens that are split by apostrophes
        new_tokens = [tokens[0]]
        for i in range(1,len(tokens)):

            if tokens[i] == '\'':
                continue
            elif tokens[i-1] == '\'' and tokens[i][0] != ' ' and tokens[i][0] not in non_alphanumeric:
                if tokens[i] == 's' or tokens[i] == 't' or tokens[i] == 'll' or tokens[i] == 've':
                    new_tokens.append(tokens[i-1]+tokens[i])
                else:
                    new_tokens.append(tokens[i-1])
                    new_tokens.append(tokens[i])    
            else:
                new_tokens.append(tokens[i])

        # Merge tokens that are preceded by spaces
        tokens = new_tokens
        new_tokens = []
        for j,token in enumerate(tokens):
            if len(token) == 0:
                continue

            if len(token) == 1:
                new_tokens.append(token)
                continue

            i = 0

            if token[0] == ' ':
                new_tokens.append(token[:2])
                i = 2
            elif token[0] == '\'':
                new_tokens.append(token)
                continue
            
            while i < len(token):
                new_tokens.append(token[i])
                i += 1        

        tokens = new_tokens

        # Create a dictionary of base tokens from the characters in the text
        token_set = set(text)
        # First token is always <PAD> token, at index 0
        word_index = OrderedDict()
        word_index['<PAD>'] =  0
        # Add all other base tokens to the dictionary
        word_index.update({token: i+1 for i, token in enumerate(token_set)})

        new_tokens = []
        for token in tokens:
            new_tokens.append(word_index[token])

        tokens = new_tokens

        k = len(word_index)

        if verbose:
            prog = progressbar.ProgressBar(max_value=self.vocab_size)
        else:
            prog = contextlib.suppress()


        # Merge tokens that appear together the most
        with prog as bar:
                
            while k < self.vocab_size:
                key_list = list(word_index.keys())
                val_list = list(word_index.values())

                occ = dict()

                for i in range(len(tokens)-1):
                    if key_list[tokens[i]] in non_alphanumeric or key_list[tokens[i+1]] in non_alphanumeric:
                        continue

                    if key_list[tokens[i]][0] == '\'' or key_list[tokens[i+1]][0] == '\'':
                        continue

                    if key_list[tokens[i+1]][0] == ' ':
                        continue

                    if key_list[tokens[i]][0] == ' ' and key_list[tokens[i+1]][0] in non_alphanumeric:
                        continue


                    if (tokens[i], tokens[i+1]) in occ:
                        occ[(tokens[i], tokens[i+1])] += 1
                    else:
                        occ[(tokens[i], tokens[i+1])] = 1   

                pair = max(occ, key=occ.get)

                if occ[pair] < 2:
                    break

                new_token = key_list[val_list.index(pair[0])]+key_list[val_list.index(pair[1])]
                if verbose:
                    bar.update(k)
                j = len(word_index)
                self.merge.append((pair[0], pair[1], j))
                word_index[new_token] = j
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                        new_tokens.append(word_index[new_token])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
                k += 1

        self.word_index = word_index

    def decode(self, ids):
        '''
        Convert tokens to text
        
        param ids: tokens to convert to text, list of integer ids or numpy array of integer ids
        returns text string
        '''
        if self.bert_model is not None:
            if type(ids) != np.ndarray and type(ids) != list:
                ids = [ids] 
            text = self.bert_tokeniser.convert_ids_to_tokens(ids)
            text = " ".join(text)
            text = clean_text(text)
            return text

        key_list = list(self.word_index.keys())

        text = ""
        if type(ids) == list:
            for t in ids:
                text += key_list[t]
        elif type(ids) == np.ndarray:
            if len(ids.shape) == 1:
                for i in range(len(ids)):
                    text += key_list[ids[i]]
            else:
                raise RuntimeError("Error: tokens must be a 1D array")
        else:
            text = key_list[ids]

        return text

    def encode(self,text, verbose=False):
        '''
        Convert text to tokens
        
        param text: text to convert to tokens, string
        returns tokens list of integer ids
        '''
        if self.bert_model is not None:
            # Bert tokeniser provides its own built-in encoding method
            tokens = self.bert_tokeniser(text, return_tensors='tf', padding=False, truncation=False)
            return tokens['input_ids'].numpy()[0].tolist()

        ids = []

        # Tokenise based on characters
        for t in text:
            try:
                ids.append(self.word_index[t])
            except:
                ids.append(0)

        if verbose:
            prog = progressbar.progressbar(range(len(self.merge)))
        else:
            prog = range(len(self.merge))

        # Merge tokens that we have merge rules for in the BPE tokeniser
        for m in prog:
            new_ids = []
            I = np.where(np.array(ids) == self.merge[m][0])[0]
            if len(I)==0:
                continue

            i_s = 0
            for i_e in I:
                if i_e+1 < len(ids) and ids[i_e+1] == self.merge[m][1]:
                    new_ids += ids[i_s:i_e]
                    new_ids += [self.merge[m][2]]
                    i_s = i_e+2

            new_ids += ids[i_s:]

            ids = new_ids
 
        return ids
    
    def get_embedding(self):
        '''
        Get the embedding matrix for the tokeniser, this is only available for BERT tokeniser
        
        returns embedding matrix as vocab_size x vec_dim numpy array'''
        if self.bert_model is None:
            raise RuntimeError("Error: embedding is not available for custom tokeniser")

        return self.bert_model.get_input_embeddings().get_weights()[0]

    def plot(self,text=None,ids=None):
        '''
        Plot the token distribution in the text, either text or token ids
        can be provided

        param text: text to plot, string
        param ids: tokens to plot, list of integer ids
        
        '''

        import matplotlib.pyplot as plt

        if ids is None:
            ids = self.encode(text)

        tokens_unique, token_counts = np.unique(ids, return_counts=True)
        I = np.argsort(token_counts)[::-1]
        tokens_unique = tokens_unique[I]
        token_counts = token_counts[I]

        # Plot the token distribution
        _, ax = plt.subplots()
        plt.plot(token_counts)
        plt.yscale('log')

        # Convert 10^{x} to 10 with x 0s in the y-axis labels
        try:
            labels = ax.get_yticklabels()
            for i in range(len(labels)):
                t = labels[i].get_text()
                if t.find('mathdefault') >= 0:
                    t = t[t.find('mathdefault')+len('mathdefault'):]
                    for c in ['}','{','$','^']:
                        t = t.replace(c,' ')
                    t = t.split()
                    if len(t) == 2:
                        t = str(int(t[0])**int(t[1]))
                        labels[i].set_text(t)
            ax.set_yticklabels(labels)
        except:
            pass

        ax.set_xlabel('Token')
        ax.set_ylabel('# occurrences in text (log scale)')
        ax.set_title('Dictionary stats')
        plt.text(0.55, 0.9, 'Dictionary size: %d tokens\nUnique tokens in text: %d\nTotal tokens in text: %d' % (self.vocab_size, len(tokens_unique), len(ids)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

        plt.show()


def plot_tok2vec(w, word_index, num_words_to_show=100, show_word_tokens=[]):
    '''
    Plot the word vectors in 2D using t-SNE.  This method is dependent on nltk library (to select words to show).
    You can install the library with 'pip install nltk'
    
    param w: word vectors, numpy array of shape (vocab_size, vec_dim)
    param word_index: tokeniser word_index, a python dictionary that maps vocab_size words to integer ids
    param num_words_to_show: number of words to show in the plot (default 100)
    param show_word_tokens: list of tokens to show in the plot, if empty, the method will select up to 100 English words from the dictionary

    '''
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    if len(show_word_tokens)==0:
        try:
            import nltk
        except:
            raise RuntimeError("Error: nltk is not installed. Please install nltk using 'pip install nltk'")

        # Select up to 100 words from the tokens that are in the words dictionary
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')

        english_dictionary = nltk.corpus.words.words()
        vocab_size = len(word_index)
        show_word_tokens = []
        words = list(word_index.keys())
        for i in range(vocab_size):
            word = words[i]
            if word[0] == ' ':
                if len(word) > 4 and word[1:] in english_dictionary:
                    show_word_tokens  += [i]
            else:
                if len(word) > 3 and word in english_dictionary:
                    show_word_tokens  += [i]

            if len(show_word_tokens) >= num_words_to_show:
                break

    # Convert the tokens to vectors
    x = w[show_word_tokens]

    # Do dimension reduction from vec_dim to 2 (so we can plot the word points in 2D)
    tsne = TSNE(n_components=2, random_state=0)
    x_2d = tsne.fit_transform(x)

    # Plot vectors showing the first 100 tokens with text close to the point corresponding to the token
    plt.figure(figsize=(6, 5))
    for i in range(len(show_word_tokens)):
        plt.scatter(x_2d[i,0], x_2d[i,1],c='b')
        word = words[show_word_tokens[i]]
        plt.text(x_2d[i,0], x_2d[i,1], word, fontsize=9)
    plt.show()


if __name__ == '__main__':
    '''
    Example how to train a tokeniser on a text and save it to disk
    '''
    from load_text import load_prideandprejudice
    import os

    vocab_size = 200
    filename = 'vocab.json'

    # Load text
    text = load_prideandprejudice()

    # Check if tokeniser has been saved to disk
    if os.path.exists(filename):
        # Load tokeniser from disk
        print("Loading tokeniser from '%s'..." % (filename))
        tokeniser = Tokeniser.load(filename)
    else:
        # Create a new tokeniser, train it on the text and save it to disk
        tokeniser = Tokeniser(vocab_size=vocab_size)
        print("Building BPE tokeniser...")
        tokeniser.train(text, verbose=True)
        print("Saving tokeniser to '%s'..." % (filename))
        tokeniser.save(filename)

    # Convert text to tokens and back to text
    print("Converting text to tokens...")
    ids = tokeniser.encode(text, verbose=True)
    print("Converting back tokens to text...")
    text_from_ids = tokeniser.decode(ids)

    assert text == text_from_ids, "Error: text and decoded text are not the same"

    tokeniser.plot(ids=ids)
