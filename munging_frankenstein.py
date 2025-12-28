import os

from argparse import Namespace
import collections
import nltk.data
import numpy as np
import pandas as pd
import re
import string
from tqdm.notebook import tqdm

args = Namespace(
    raw_dataset_txt="data/books/frankenstein.txt",
    window_size=5,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    output_munged_csv="data/books/frankenstein_with_splits.csv",
    seed=9248
)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with open(args.raw_dataset_txt) as fp:
    book = fp.read()
sentences = tokenizer.tokenize(book)

print (len(sentences), "sentences")
print ("Sample:", sentences[100])

def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

cleaned_sentences = [preprocess_text(sentence) for sentence in sentences]
MASK_TOKEN = "<MASK>"

flatten = lambda outer_list: [item for inner_list in outer_list for item in inner_list]
windows = flatten([list(nltk.ngrams([MASK_TOKEN] * args.window_size + sentence.split(' ') + \
    [MASK_TOKEN] * args.window_size, args.window_size * 2 + 1)) \
    for sentence in tqdm(cleaned_sentences)])

# Create cbow data
data = []
for window in tqdm(windows):
    target_token = window[args.window_size]
    context = []
    for i, token in enumerate(window):
        if token == MASK_TOKEN or i == args.window_size:
            continue
        else:
            context.append(token)
    data.append([' '.join(token for token in context), target_token])
    
            
cbow_data = pd.DataFrame(data, columns=["context", "target"])