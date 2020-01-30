import os
import string
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paper Configurations
hidden_size = 2048
batch_size = 1024  # 500k total batches
learning_rate = 6 * 10 ** -4
beta1 = 0.9  # adam optimizer params
beta2 = 0.995
epsilon = 10 ** -9
optimizer = 'adam'  # Don't just use preconfigured adam, need to set custom beta/epsilon values
abs_gradient_clipping = 0.1
embed_size = 512

epochs = 262

max_question_length = 160  # characters
max_answer_length = 30

# Lexicon and embedding init
vocab = list(string.punctuation + string.ascii_letters + string.digits) + [" ", "\t", "\n"]
num_oov_buckets = 1
vocab_size = len(vocab) + num_oov_buckets

vocab_table = dict(
    [(char, i) for i, char in enumerate(vocab)])


# category index mapping
table_init = \
    tf.lookup.KeyValueTensorInitializer(vocab,
                                        tf.range(len(vocab),
                                                 dtype=tf.int64))

table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)


# Misc Parameters
train_path = r'data/mathematics_dataset-v1.0/train.txt'
interpolate_path = r'data/mathematics_dataset-v1.0/interpolate_test.txt'
extrapolate_path = r'data/mathematics_dataset-v1.0/extrapolate_test.txt'
