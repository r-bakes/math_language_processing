from datetime import time
import pandas as pd
import numpy as np
import random
import math
import torch
import string
import os

from definitions import DATA_DIR



vocab = ['<SOS>','<EOS>'] + list(string.punctuation + string.ascii_letters + string.digits)
n_vocab = len(vocab)
char2index = dict([(char, i) for i, char in enumerate(vocab)])
index2char = dict([(i, char) for i, char in enumerate(vocab)])


def get_data(n_train, q_type, type='train'):
    if type == 'train':
        data_dir = \
            os.path.join(DATA_DIR, 'train-easy', q_type)

    data_dir = \
        os.path.join(DATA_DIR, 'interpolate', q_type)


    with open(data_dir, 'r') as f:
        data = f.read()
        data = np.array(data.splitlines()).reshape(-1,2)[0:n_train]
        f.close()

    return data


def line2tensor(line):
    tensor = torch.zeros(len(line), 1, n_vocab)
    for i, char in enumerate(line):
        tensor[i][0][char2index[char]] = 1
    return tensor


def tensor2char(tensor):
    top_n, top_i = tensor.topk(1)
    category_i = top_i[0].item()
    return index2char[category_i], category_i


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# def random_choice(l):
#     return l[random.randint(0, len(l) - 1)]
#
# def randomTrainingExample():
#     category = randomChoice(all_categories)
#     line = randomChoice(category_lines[category])
#     category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
#     line_tensor = lineToTensor(line)
#     return category, line, category_tensor, line_tensor
#
# for i in range(10):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     print('category =', category, '/ line =', line)