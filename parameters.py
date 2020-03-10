import os
import string
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paper Configurations
hidden_size = 2048
# batch_size = 1024  500k total batches
batch_size = 64
learning_rate = 6 * 10 ** -4
beta1 = 0.9  # adam optimizer params
beta2 = 0.995
epsilon = 10 ** -9
optimizer = 'adam'  # Don't just use preconfigured adam, need to set custom beta/epsilon values
abs_gradient_clipping = 0.1
embed_size = 512
epochs = 262
max_question_length = 160  # characters
max_answer_length = 32

# Data
q_list = [filename for filename in os.listdir(os.path.join(ROOT_DIR, r"data/train-easy")) if filename.endswith(".txt")]


# Lexicon and embedding init
vocab = ['<PAD>'] + list(string.punctuation + string.ascii_letters + string.digits + " " +"\t" + "\n")
vocab_size = len(vocab)
vocab_table = dict([(char, i) for i, char in enumerate(vocab)])


vocab_forest = list(string.punctuation + string.ascii_letters + string.digits + ' ')
vocab_table_forest = dict([(char, i) for i, char in enumerate(vocab_forest)])
vocab_size_forest = len(vocab_forest)

