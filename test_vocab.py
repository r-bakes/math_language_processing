from definitions import ROOT_DIR
import torch
import os
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset, Dataset
from definitions import DATA_TSV_DIR
from data import create_data_iterators


# TESTING PARAMETERS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
q_type = 'algebra__linear_1d.tsv'
char_offset = True

"""
GRAB DATA
"""
_, _, SRC, TRG, _ = create_data_iterators(n_train=-1, q_type=q_type, device=device, difficulty='train-easy', batch_size='1')  # grab the SRC and TRG lexicons

print(SRC.vocab.stoi)
print('')
print(TRG.vocab.stoi)
print('')