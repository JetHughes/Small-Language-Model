__author__ = "Lech Szymanski"
__organization__ = "COSC420, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import os
import re
from urllib.request import urlretrieve

def clean_text(text):
    '''
    clean_text cleans up the text by removing unwanted characters and standardising
    spaces between non-alphanumeric characters
    returns the cleaned text
    '''
    # List of text to replace
    replacements = [(' ##', ''), ('[CLS] ', ''), (' [SEP]', ''), ('[PAD]', ''), ('[UNK]', ''), ('— — ’ s', '——’s'), (' ”', '”'), ('“ ', '“'), (' ’', '’'), (' - ', '-'), ('’ t ','’t '), ('’ s ','’s '), ('’ ve ','’ve '),('’ ll ','’ll '), ('’ re ','’re '), ('’ m ','’m '), ('’ d ','’d '), ('’ ', '’'),  (' ‘ ', ' ‘'), (' ( ', ' ('), (' ) ', ') '),('“‘ ', '“‘'), (' — ', '—'), ('“— ', '“—'), ('‘ Y', '‘Y') ]
    #Add to replacements all space followed by punctuation
    for p in ['.',',','!','?',';',':']:
        replacements.append((' '+p,p))
    for p in ['-', '—']:
        replacements.append((' '+p,p))

    # Make the replacements
    for r in replacements:
        text = text.replace(r[0],r[1])

    # Precede all dashes and slashes with a space and follow it with a space
    text = re.sub("-", " - ", text)
    text = re.sub("—", " — ", text)
    text = re.sub("/", " / ", text)

    text = text.replace('‘', "'")

    # Follow any non-alpha numeric character with a space
    text = re.sub("([^\w\s])", r'\1 ', text)

    return text


def load_prideandprejudice(max_words = 121810):
    """
    load_prideandprejudice loads text of "Pride and Prejudice" (need Internet connection
    first time this is run, for downloading the book text from Project Gutenberg
    website)

    :param max_words: maximum words to load, 121,810 (default) is the entire book
    :return: text of "Pride and Prejudice" as a single string
    """ 

    url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
    filename = "1342-0.txt"

    # Download the book text from Project Gutenberg website
    # if not found on hard drive
    if not os.path.exists(filename):
        urlretrieve(url, filename)

    # Read the book from file
    with open(filename, 'r',encoding='utf-8') as file:
        text = file.read()

    # Remove new line characters
    text = text.replace("\n", " ")

    # Remove header text
    header = 'Chapter I.]'
    text = text[text.find(header)+len(header):]

    # Remove fotter text
    header = 'CHISWICK PRESS:--CHARLES WHITTINGHAM AND CO.'
    text = text[:text.find(header)]

    # Replace all ]] brackets with ]
    text = text.replace(']]', ']')

    # Remove [] brackets from text and text between them
    text = re.sub("[\[].*?[\]]", "", text)

    # Remove all --
    text = text.replace('--', ' ')

    # Remove all -
    text = text.replace('_', '')

    # Remove 'CHAPTER' case insensitive followed by text up to a full stop
    text = re.sub("CHAPTER.*?\.", "", text, flags=re.IGNORECASE)

    # Remove /* */ brackets from text and text between them
    text = re.sub("/\*.*?\*/", "", text)

    # Split on empty spaces
    text = text.split()

    # Select max_wods words from the string (assuming spaces delimit words)
    if max_words < len(text):
        text = text[:max_words]

    # Join the words back into a single string
    text = ' '.join(text)

    # Clean text
    text = clean_text(text)

    return text

def load_warandpeace(max_words = 598184):
    """
    load_warandpeace loads text of "War and Peace" (need Internet connection
    first time this is run, for downloading the book text from Project Gutenberg
    website)

    :param max_words: maximum words to load, 598,184 (default) is the entire book
    :return: text of "War and Peace" as a single string
    """ 

    url = 'https://www.gutenberg.org/cache/epub/2600/pg2600.txt'
    filename = "pg2600.txt"

    # Download the book text from Project Gutenberg website
    # if not found on hard drive
    if not os.path.exists(filename):
        urlretrieve(url, filename)

    # Read the book from file
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # Remove the header stuff
    header = '*** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***'
    text = text[text.find(header)+len(header):]

    # Remove the footer stuff
    header = '*** END OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***'
    text = text[:text.find(header)]

    # Scan the table of contents
    header = 'Contents'
    text = text[text.find(header)+len(header):]

    text = text.split("\n")

    # Remove 'Chater <number>' and 'Book <number>' from 
    # the text
    chapters = []    
    for i,line in enumerate(text):
        line = line.strip()

        if line == '':
            continue
       
        if len(chapters)>0  and line == chapters[0]:
            break

        if line in chapters:
            continue

        chapters.append(line)
    
    text = text[i:]
    
    new_text = []
    for line in text:
        line = line.strip()

        if line == '':
            continue

        if line in chapters:
            continue

        new_text.append(line)

    # Remove all lines starting with *
    i = 0
    while i < len(new_text):
        line = new_text[i]
        if re.match("^\*", line):
            new_text.pop(i)
        else:
            i += 1

    # Remove all *(<number>) where <number> is some number, but not the whole line, just *(<number>)
    i = 0
    while i < len(new_text):
        line = new_text[i]
        line = re.sub("\*\(\d+\)", "", line)
        new_text[i] = line
        i += 1

    # Remove all *, but not the whole line, just *
    i = 0
    while i < len(new_text):
        line = new_text[i]
        line = re.sub("\*", "", line)
        new_text[i] = line
        i += 1

    # Remove all new line characters
    text = ' '.join(new_text)
    text = text.replace("\n", " ")

    # Replace all ]] brackets with ]
    text = text.replace(']]', ']')

    # Remove [] brackets from text and text between them
    text = re.sub("[\[].*?[\]]", "", text)

    # Remove all --
    text = text.replace('--', ' ')

    # Remove all -
    text = text.replace('_', '')

    # Remove /* */ brackets from text and text between them
    text = re.sub("/\*.*?\*/", "", text)

    # Precede all dashes with a space and follow it with a space
    text = re.sub("-", " - ", text)
    text = re.sub("—", " — ", text)
    text = re.sub("/", " / ", text)

    # Follow any non-alpha numeric character with a space
    text = re.sub("([^\w\s])", r'\1 ', text)

    # Make sure there are no double spaces
    text = re.sub("\s+", " ", text)

    # Clean up empty spaces
    text = text.split()

    # Select max_wods words from the string (assuming spaces delimit words)
    if max_words is not None and max_words < len(text):
        text = text[:max_words]

    # Join the words back into a single string
    text = ' '.join(text)

    # Clean text
    text = clean_text(text)

    return text

if __name__ == '__main__':

    # Example how to load text of "Pride and Prejudice"...
    text = load_prideandprejudice()
    # ...and print it out
    print(text)

