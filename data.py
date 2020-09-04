from datetime import time
import pandas as pd
import numpy as np
import random
import math
import torch
import string
import os

from definitions import DATA_DIR



vocab =  list(string.punctuation + string.ascii_letters + string.digits)
n_vocab = len(vocab)
char2index = dict([(char, i) for i, char in enumerate(vocab)])
index2char = dict([(i, char) for i, char in enumerate(vocab)])


def get_data(n_train, q_type, type='train'):
    data_dir = \
        os.path.join(DATA_DIR, 'interpolate', q_type)

    if type == 'train':
        data_dir = \
            os.path.join(DATA_DIR, 'train-easy', q_type)




    with open(data_dir, 'r') as f:
        data = f.read()
        data = np.array(data.splitlines()).reshape(-1,2)[0:n_train]
        f.close()

    return data


def random_pair(data):
    return random.choice(data)







def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
