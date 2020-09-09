from datetime import time
import pandas as pd
import numpy as np
import random
import math
import torch


import string
import os

from definitions import DATA_DIR



from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
from definitions import DATA_TSV_DIR


vocab = list(string.punctuation + string.ascii_letters + string.digits)
n_vocab = len(vocab)
char2index = dict([(char, i) for i, char in enumerate(vocab)])
index2char = dict([(i, char) for i, char in enumerate(vocab)])

def create_data_iterators(n_train, q_type, device):
    BATCH_SIZE=64

    SRC = Field(tokenize=list,
                lower=True)

    TRG = Field(tokenize=list,
                init_token='<SOS>',
                eos_token='<EOS>',
                lower=True)

    data = TabularDataset(path=os.path.join(DATA_TSV_DIR, 'train-easy', q_type),
                          format='TSV',
                          fields=[('index', None), ('question', SRC), ('answer', TRG)],
                          skip_header=True)

    test = TabularDataset(path=os.path.join(DATA_TSV_DIR, 'interpolate', q_type),
                          format='TSV',
                          fields=[('index', None), ('question', SRC), ('answer', TRG)],
                          skip_header=True)


    data.examples = data.examples[0:n_train]  # reduce scope for testing

    train, val = data.split(split_ratio=0.8, stratified=False, strata_field='label', random_state=None)

    SRC.build_vocab(data)
    SRC.build_vocab(test)

    TRG.build_vocab(data)
    TRG.build_vocab(test)

    train_iterator, valid_iterator = BucketIterator.splits(
        (train, val),
        batch_size=BATCH_SIZE,
        sort=False,
        device=device)

    test_iterator = Iterator(test,
                        batch_size=1,
                        sort=False,
                        train=False,
                        device=device)

    return train_iterator, valid_iterator, SRC, TRG, test_iterator



def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
