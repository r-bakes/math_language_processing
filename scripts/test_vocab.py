from definitions import ROOT_DIR
import torch
import os
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset, Dataset
from definitions import DATA_TSV_DIR


def create_data_iterators(n_train, q_type, device, difficulty, batch_size):

    SRC = Field(tokenize=list,
                lower=True)

    TRG = Field(tokenize=list,
                init_token='<SOS>',
                eos_token='<EOS>',
                lower=True)
    if difficulty == 'train-all':
        data_easy = TabularDataset(path=os.path.join(DATA_TSV_DIR, 'train-easy', q_type),
                                   format='TSV',
                                   fields=[('index', None), ('question', SRC), ('answer', TRG)],
                                   skip_header=True)

        data_medium = TabularDataset(path=os.path.join(DATA_TSV_DIR, 'train-medium', q_type),
                                     format='TSV',
                                     fields=[('index', None), ('question', SRC), ('answer', TRG)],
                                     skip_header=True)

        data_hard = TabularDataset(path=os.path.join(DATA_TSV_DIR, 'train-hard', q_type),
                                   format='TSV',
                                   fields=[('index', None), ('question', SRC), ('answer', TRG)],
                                   skip_header=True)

        data = Dataset([example for example in data_hard + data_medium + data_easy],
                       fields=[('index', None), ('question', SRC), ('answer', TRG)],)


    else:
        data = TabularDataset(path=os.path.join(DATA_TSV_DIR, difficulty, q_type),
                              format='TSV',
                              fields=[('index', None), ('question', SRC), ('answer', TRG)],
                              skip_header=True)


    test = TabularDataset(path=os.path.join(DATA_TSV_DIR, 'interpolate', q_type),
                          format='TSV',
                          fields=[('index', None), ('question', SRC), ('answer', TRG)],
                          skip_header=True)


    if n_train != -1: data.examples = data.examples[0:n_train]  # reduce scope for testing
    if 0 < n_train <= 20: test.examples = test.examples[0:n_train]

    print(f'Number of train examples: {len(data)}')

    train, val = data.split(split_ratio=0.9, stratified=False, strata_field='label', random_state=None)

    SRC.build_vocab(data)
    SRC.build_vocab(test)

    TRG.build_vocab(data)
    TRG.build_vocab(test)

    train_iterator, valid_iterator = BucketIterator.splits(
        (train, val),
        batch_size=batch_size,
        sort=False,
        device=device)

    test_iterator = Iterator(test,
                        batch_size=1,
                        sort=False,
                        train=False,
                        device=device)

    return train_iterator, valid_iterator, SRC, TRG, test_iterator



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