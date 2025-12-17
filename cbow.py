import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

class Vocabulary(object):
    def __init__(self, token_to_idx=None, mask_token = "<MASK>", add_unk = True, unk_token = "<UNK>"):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token
        self.mask_index = self.add_token(self._mask_token)

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(self._unk_token)

        def to_serializable(self):
            return {'token_to_idx': self._token_to_idx,
                    'add_unk': self._add_unk,
                    'unk_token': self._unk_token,
                    'mask_token': self._mask_token}
        
        classmethod
        def from_serializable(cls, contents):
            return cls(**contents)
        
        def add_token(self, token):
            if token in self._token_to_idx:
                index = self._token_to_idx[token]
            else:
                index = len(self._token_to_idx)
                self._token_to_idx[token] = index
                self._idx_to_token[index] = token
            return token
        
        def add_many(self, tokens):
            return [self.add_token(token) for token in tokens]
        
        def lookup_token(self, token):
            if self._unk_index >= 0:
                return self._token_to_idx.get(token, self.unk_index)
            else:
                return self._token_to_idx[token]
            
        def lookup_index(self, index):
            if index not in self._idx_to_token:
                raise KeyError("the index (%d) isn't in the vocabulary" % index)
            return self._idx_to_token[index]
        
        def __str__(self):
            return "<Vocabulary (size=%d)" % len(self)
        
        def __len__(self):
            return len(self._token_to_idx)

class CBOWVectorizer(object):
    def __init__(self, cbow_vocab):
        self.cbow_vocab = cbow_vocab
    
    def vectorize(self, context, vector_length=-1):
        indicies = [self.cbow_vocab.lookup_token(token) for token in context.split(' ')]

        if vector_length < 0:
            vector_length = len(indicies)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indicies)] = indicies
        out_vector[len(indicies):] = self.cbow_vocab.mask_index

        return out_vector
    
    @classmethod
    def from_dataframe(cls, cbow_df):
        cbow_vocab = Vocabulary()
        for index, row in cbow_df.iterrows():
            for token in row.context.split(' '):
                cbow_vocab.add_token(token)
            cbow_vocab.add_token(row.target)
        return cls(cbow_vocab)
    
    @classmethod
    def from_serializable(cls, contents):
        cbow_vocab = Vocabulary.from_serializable(contents['cbow_vocab'])
        return cls(cbow_vocab=cbow_vocab)
    
    def to_serializable(self):
        return {'cbow_vocab': self.cbow_vocab.to_serializable()}

class CBOWDataset(Dataset):
    def __init__(self, cbow_df, vectorizer):
        self.cbow_df = cbow_df
        self._vectorizer = vectorizer

        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, cbow_df.context))

        self.train_df = self.cbow_df[self.cbow_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.cbow_df[self.cbow_df.split=='val']
        self.val_size = len(self.val_df)

        self.test_df = self.cbow_df[self.cbow_df.split=='test']
        self.test_size = len(self.test_size)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.val_size),
                             'test': (self.test_df, self.test_size)}
        
        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, cbow_csv):
        cbow_df = pd.read_csv(cbow_csv)
        train_cbow_df = cbow_df[cbow_df.split=='train']
        return cls(cbow_df, CBOWVectorizer.from_dataframe(train_cbow_df))
    
    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file
        
        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of CBOWVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return CBOWVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json
        
        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer
        
    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        context_vector = \
            self._vectorizer.vectorize(row.context, self._max_seq_length)
        target_index = self._vectorizer.cbow_vocab.lookup_token(row.target)

        return {'x_data': context_vector,
                'y_target': target_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
    
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"): 
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


class CBOWClassifier(nn.Module):
    