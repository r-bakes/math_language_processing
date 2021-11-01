import tensorflow as tf
import numpy as np
from src import parameters as p
import definitions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

import os


class Processor:

    dir_data = os.path.join(definitions.ROOT_DIR, "data")

    def __init__(self, q_type):
        self.question_type = q_type

    def encoder_decoder_sequence_preprocess(self, texts):
        input_texts = np.array(texts[0])
        target_texts = np.array(texts[1])
        max_encoder_seq_length = p.max_question_length
        max_decoder_seq_length = p.max_answer_length

        target_texts = np.char.add(np.full(shape=len(target_texts), fill_value='\t'), target_texts)
        target_texts = np.char.add(target_texts, np.full(shape=len(target_texts), fill_value='\n'))

        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length),
            dtype='float32')
        decoder_target_data = np.full(
            (len(input_texts), max_decoder_seq_length), -1,  # so one hot encode will create zero vector representation
            dtype='int32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t] = p.vocab_table[char]
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t] = p.vocab_table[char]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1] = p.vocab_table[char]

        decoder_target_data = tf.one_hot(decoder_target_data, p.vocab_size)

        return encoder_input_data, decoder_input_data, decoder_target_data

    def encoder_decoder_hot_preprocess(self, texts):
        input_texts = np.array(texts[0])
        target_texts = np.array(texts[1])
        max_encoder_seq_length = p.max_question_length
        max_decoder_seq_length = p.max_answer_length
        num_encoder_tokens = p.vocab_size
        num_decoder_tokens = p.vocab_size

        target_texts = np.char.add(np.full(shape=len(target_texts), fill_value='\t'), target_texts)
        target_texts = np.char.add(target_texts, np.full(shape=len(target_texts), fill_value='\n'))

        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, p.vocab_table[char]] = 1.
            # encoder_input_data[i, t + 1:, p.vocab_table['<PAD>']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, p.vocab_table[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, p.vocab_table[char]] = 1.
            # decoder_input_data[i, t + 1:, p.vocab_table['<PAD>']] = 1.
            # decoder_target_data[i, t:, p.vocab_table['<PAD>']] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data

    def tfid_word_preprocess(self, n_data):
        pass

    def tfid_char_preprocess(self, n_data):
        x_train, y_train, x_test, y_test = self.get_data(lowercase=True, n_data=n_data)
        vectorizer = TfidfVectorizer(analyzer='char', lowercase=True)

        vectorizer.fit(x_train)

        x_test_copy = x_test.copy()

        x_train = vectorizer.transform(x_train)
        x_test = vectorizer.transform(x_test)

        return x_train, y_train, x_test, y_test, x_test_copy, vectorizer  # Return TF-IDF weighted matrix

    def onehot_char_preprocess(self, n_data):
        x_train, y_train, x_test, y_test = self.get_data(lowercase=True, n_data=n_data)

        train_encoding = np.zeros((len(x_train), p.vocab_size_forest), dtype='float32')
        test_encoding = np.zeros((len(x_test), p.vocab_size_forest), dtype='float32')

        x_test_copy = x_test.copy()

        for i, text in enumerate(x_train): # Encode train set
            for t, char in enumerate(text):
                train_encoding[i, t] = p.vocab_table_forest[char]

        for i, text in enumerate(x_test): # Encode test set
            for t, char in enumerate(text):
                test_encoding[i, t] = p.vocab_table_forest[char]

        return train_encoding, y_train, test_encoding, y_test, x_test_copy, None  # Return one hot matrix of encoded values and targets

    def onehot_word_preprocess(self, n_data):
        x_train, y_train, x_test, y_test = self.get_data(lowercase=True, n_data=n_data)
        encoder = OneHotEncoder()

        x_train = np.char.split(x_train, sep=" ")
        x_test = np.char.split(x_test, sep=" ")

        encoder.fit(x_train)
        print(encoder.get_feature_names())

        x_test_copy = x_test.copy()

        x_train = encoder.transform(x_train)
        x_test = encoder.transform(x_test)

        return x_train, y_train, x_test, y_test, x_test_copy, encoder  # Return one hot matrix

    def get_data(self, n_data, lowercase=False):
        train_easy = open(os.path.join(self.dir_data,r"train-easy", self.question_type), 'r').read().splitlines()
        # train_medium = open(os.path.join(self.dir_data,r"train-medium",self.question_type), 'r').read().splitlines()
        # train_hard = open(os.path.join(self.dir_data,r"train-hard",self.question_type), 'r').read().splitlines()
        test = open(os.path.join(self.dir_data, r"interpolate", self.question_type), 'r').read().splitlines()

        # focus on easy for now
        # train_medium, train_hard = np.reshape(train_medium, (-1, 2)), np.reshape(train_hard, (-1, 2))
        train = np.reshape(train_easy, (-1, 2))
        test = np.reshape(test, (-1, 2))

        np.random.shuffle(train)  # Shuffle data
        np.random.shuffle(test)

        # Testing reduce scope
        train = train[0:n_data]
        test = test[0:n_data]

        if lowercase is True:
            np.char.lower(train)
            np.char.lower(test)

        return train[:,0], train[:,1], test[:,0], test[:,1]

    def get_data_exp(self, n_data=None):
        train_easy = open(os.path.join(self.dir_data, r"train-easy", self.question_type), 'r').read().splitlines()
        test = open(os.path.join(self.dir_data, r"interpolate", self.question_type), 'r').read().splitlines()

        train_easy = np.reshape(train_easy, (-1, 2))
        test = np.reshape(test, (-1, 2))

        train_easy = train_easy[0:n_data]
        test = test[0:n_data]

        train_x, train_y = train_easy[:,0], train_easy[:,1]
        train_y = np.char.add(np.full(shape=len(train_y), fill_value='\t'), train_y)
        train_y = np.char.add(train_y, np.full(shape=len(train_y), fill_value='\n'))

        test_x, test_y = test[:,0], test[:,1]
        test_y = np.char.add(np.full(shape=len(test_y), fill_value='\t'), test_y)
        test_y = np.char.add(test_y, np.full(shape=len(test_y), fill_value='\n'))

        data_train = tf.data.Dataset.from_tensor_slices((train_x,train_y))
        data_test = tf.data.Dataset.from_tensor_slices((test_x,test_y))

        return data_train, data_test

    # def one_hot_exp(self):
    #
    #     def encoder_input(dataset):
    #
    #     def decoder_input(dataset):
    #
    #     def
