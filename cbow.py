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

 
