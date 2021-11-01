"""
This file contains the parameters taken from the arxiv paper which inspired this thesis. See README.md for link.
"""

import string

# Paper Configurations
hidden_size = 2048
# batch_size = 1024  500k total batches
batch_size = 64
learning_rate = 6 * 10 ** -4
beta1 = 0.9  # adam optimizer params
beta2 = 0.995
epsilon = 10 ** -9
optimizer = 'adam'
abs_gradient_clipping = 0.1
embed_size = 512
epochs = 262
max_question_length = 160  # number of characters
max_answer_length = 32

# Lexicon and embedding init
vocab = ['<PAD>'] + list(string.punctuation + string.ascii_letters + string.digits)
vocab_size = len(vocab)
vocab_table = dict([(char, i) for i, char in enumerate(vocab)])


vocab_forest = list(string.punctuation + string.ascii_letters + string.digits + ' ')
vocab_table_forest = dict([(char, i) for i, char in enumerate(vocab_forest)])
vocab_size_forest = len(vocab_forest)

