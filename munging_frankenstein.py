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