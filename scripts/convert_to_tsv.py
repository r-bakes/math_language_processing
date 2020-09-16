import os
import numpy as np
import pandas as pd

from definitions import DATA_DIR, ROOT_DIR

try:
    os.mkdir(os.path.join(ROOT_DIR, 'data_tsv'))
    os.mkdir(os.path.join(ROOT_DIR, 'data_tsv', 'extrapolate'))
    os.mkdir(os.path.join(ROOT_DIR, 'data_tsv', 'interpolate'))
    os.mkdir(os.path.join(ROOT_DIR, 'data_tsv', 'train-easy'))
    os.mkdir(os.path.join(ROOT_DIR, 'data_tsv', 'train-medium'))
    os.mkdir(os.path.join(ROOT_DIR, 'data_tsv', 'train-hard'))
except:
    pass
#
#
# for root, dir, files in os.walk(os.path.abspath(os.path.join(DATA_DIR, 'train-easy'))):
#     for question in files:
#         with open(os.path.join(root, question), 'r') as f:
#             data = f.read()
#             data = np.array(data.splitlines()).reshape(-1,2)
#             f.close()
#
#             data = pd.DataFrame({'question': data[:, 0], 'answer': data[:, 1]})
#
#             data.to_csv(os.path.join(ROOT_DIR,'data_tsv', 'train-easy', question[0:-3]) + 'tsv', sep='\t', index=True)

for root, dir, files in os.walk(os.path.abspath(os.path.join(DATA_DIR, 'train-medium'))):
    for question in files:
        with open(os.path.join(root, question), 'r') as f:
            data = f.read()
            data = np.array(data.splitlines()).reshape(-1,2)
            f.close()

            data = pd.DataFrame({'question': data[:, 0], 'answer': data[:, 1]})

            data.to_csv(os.path.join(ROOT_DIR,'data_tsv', 'train-medium', question[0:-3]) + 'tsv', sep='\t', index=True)

for root, dir, files in os.walk(os.path.abspath(os.path.join(DATA_DIR, 'train-hard'))):
    for question in files:
        with open(os.path.join(root, question), 'r') as f:
            data = f.read()
            data = np.array(data.splitlines()).reshape(-1,2)
            f.close()

            data = pd.DataFrame({'question': data[:, 0], 'answer': data[:, 1]})

            data.to_csv(os.path.join(ROOT_DIR,'data_tsv', 'train-hard', question[0:-3]) + 'tsv', sep='\t', index=True)

# for root, dir, files in os.walk(os.path.abspath(os.path.join(DATA_DIR, 'interpolate'))):
#     for question in files:
#         with open(os.path.join(root, question), 'r') as f:
#             data = f.read()
#             data = np.array(data.splitlines()).reshape(-1,2)
#             f.close()
#
#             data = pd.DataFrame({'question': data[:, 0], 'answer': data[:, 1]})
#
#             data.to_csv(os.path.join(ROOT_DIR,'data_tsv', 'interpolate', question[0:-3]) + 'tsv', sep='\t', index=True)
