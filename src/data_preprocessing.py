"""
This file contains all the methods related to initally loading data and mild preprocessing.
Only applicable to neural network algorithms experimented with.
"""

from typing import Tuple, Optional

import numpy as np


import string
import os

from definitions import DATA_DIR
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset, Dataset


vocab = list(string.punctuation + string.ascii_letters + string.digits)
n_vocab = len(vocab)
char2index = dict([(char, i) for i, char in enumerate(vocab)])
index2char = dict([(i, char) for i, char in enumerate(vocab)])


def get_data(
    question_type: str,
    difficulty: Optional[str] = None,
    set_size: Optional[int] = None,
) -> np.array:
    """For simple ML algorithms. Retrieves raw .tsv question sets into a  M x 2 numpy array where the first
    column is the question string and the second column is the solution.

    Args:
        difficulty:
        question_type:
        set_size:

    Returns:

    """

    if difficulty is None:  # Test data request
        data_path = os.path.join(DATA_DIR, "test", "interpolate", question_type)
    else:
        data_path = os.path.join(DATA_DIR, "train", difficulty, question_type)

    with open(data_path, "r") as f:

        data = f.read()

        data = np.array(data.splitlines()).reshape(-1, 2)

        if set_size:
            data = data[0:set_size]

        f.close()

    x, y = data[:, 0], data[:, 1]

    return x, y


def create_data_iterators(n_train, q_type, device, difficulty, batch_size):

    SRC = Field(tokenize=list, lower=True)

    TRG = Field(tokenize=list, init_token="<SOS>", eos_token="<EOS>", lower=True)
    if difficulty == "all":
        data_easy = TabularDataset(
            path=os.path.join(DATA_TSV_DIR, "easy", q_type),
            format="TSV",
            fields=[("index", None), ("question", SRC), ("answer", TRG)],
            skip_header=True,
        )

        data_medium = TabularDataset(
            path=os.path.join(DATA_TSV_DIR, "medium", q_type),
            format="TSV",
            fields=[("index", None), ("question", SRC), ("answer", TRG)],
            skip_header=True,
        )

        data_hard = TabularDataset(
            path=os.path.join(DATA_TSV_DIR, "hard", q_type),
            format="TSV",
            fields=[("index", None), ("question", SRC), ("answer", TRG)],
            skip_header=True,
        )

        data = Dataset(
            [example for example in data_hard + data_medium + data_easy],
            fields=[("index", None), ("question", SRC), ("answer", TRG)],
        )

    else:
        data = TabularDataset(
            path=os.path.join(DATA_TSV_DIR, difficulty, q_type),
            format="TSV",
            fields=[("index", None), ("question", SRC), ("answer", TRG)],
            skip_header=True,
        )

    test = TabularDataset(
        path=os.path.join(DATA_TSV_DIR, "interpolate", q_type),
        format="TSV",
        fields=[("index", None), ("question", SRC), ("answer", TRG)],
        skip_header=True,
    )

    if n_train != -1:
        data.examples = data.examples[0:n_train]  # reduce scope for testing
    if 0 < n_train <= 20:
        test.examples = test.examples[0:n_train]

    print(f"Number of train examples: {len(data)}")

    train, val = data.split(
        split_ratio=0.9, stratified=False, strata_field="label", random_state=None
    )

    SRC.build_vocab(data)
    SRC.build_vocab(test)

    TRG.build_vocab(data)
    TRG.build_vocab(test)

    train_iterator, valid_iterator = BucketIterator.splits(
        (train, val), batch_size=batch_size, sort=False, device=device
    )

    test_iterator = Iterator(test, batch_size=1, sort=False, train=False, device=device)

    return train_iterator, valid_iterator, SRC, TRG, test_iterator


def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
