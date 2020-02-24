import tensorflow as tf
from tensorflow import keras
import numpy as np
import parameters as p
import definitions
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

import os
import pdb


class Processor:

    dir_data = os.path.join(definitions.ROOT_DIR, "data")

    def __init__(self, q_type):
        self.question_type = q_type

    def encoder_decoder_sequence_preprocess(self, texts):
        input_texts = np.array(texts[0])
        target_texts = np.array(texts[1])
        max_encoder_seq_length = p.max_question_length
        max_decoder_seq_length = p.max_question_length

        target_texts = np.char.add(np.full(shape=len(target_texts), fill_value='\t'), target_texts)
        target_texts = np.char.add(target_texts, np.full(shape=len(target_texts), fill_value='\n'))

        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length),
            dtype='float32')

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

        encoder_input_data =\
            keras.preprocessing.sequence.pad_sequences(encoder_input_data, maxlen=p.max_question_length, padding='post', truncating='post', value=0.0)
        decoder_target_data =\
            keras.preprocessing.sequence.pad_sequences(decoder_target_data, maxlen=p.max_question_length, padding='post', truncating='post', value=0.0)
        decoder_input_data =\
            keras.preprocessing.sequence.pad_sequences(decoder_input_data, maxlen=p.max_question_length, padding='post', truncating='post', value=0.0)

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
            encoder_input_data[i, t + 1:, p.vocab_table['<PAD>']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, p.vocab_table[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, p.vocab_table[char]] = 1.
            decoder_input_data[i, t + 1:, p.vocab_table['<PAD>']] = 1.
            decoder_target_data[i, t:, p.vocab_table['<PAD>']] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data

    def random_forest_tfid_word(self, n_data):
        train_x, train_y, test_x, test_y = self.get_data(lowercase=True, n_data=n_data)
        vectorizer = CountVectorizer(analyzer='word', lowercase=True, token_pattern='(?u)[^ ]+')
        vectorizer.fit_transform(np.concatenate((train_x, train_y)), )  # Remove later and hardcode vocab
        print(vectorizer.get_feature_names())

        count_vec = vectorizer.transform(train_x)

        pdb.set_trace()



        # pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
        #                 ('tfid', TfidfTransformer())]).fit(corpus)
        # transformer = TfidfTransformer(smooth_idf=False)

        encoder = preprocessing.OneHotEncoder()
        encoder.fit(np.array(p.vocab_forest).reshape(1,-1))

        pdb.set_trace()

        encoder.transform(train_x)

        train_x = encoder

        pdb.set_trace()

    def random_forest_tfid_char(self, n_data):
        x_train, y_train, x_test, y_test = self.get_data(lowercase=True, n_data=n_data)
        vectorizer = TfidfVectorizer(analyzer='char', lowercase=True)

        vectorizer.fit(np.concatenate((x_train, y_train)))
        print(vectorizer.get_feature_names())

        x_test_copy = x_test.copy()

        x_train = vectorizer.transform(x_train)
        x_test = vectorizer.transform(x_test)

        return x_train, y_train, x_test, y_test, x_test_copy, vectorizer  # Return TF-IDF weighted matrix


    def get_data(self, n_data, lowercase=False):
        train_easy = open(os.path.join(self.dir_data,r"train-easy", self.question_type), 'r').read().splitlines()
        train_medium = open(os.path.join(self.dir_data,r"train-medium",self.question_type), 'r').read().splitlines()
        train_hard = open(os.path.join(self.dir_data,r"train-hard",self.question_type), 'r').read().splitlines()
        test = open(os.path.join(self.dir_data, r"interpolate", self.question_type), 'r').read().splitlines()

        # focus on easy for now
        # train_medium, train_hard = np.reshape(train_medium, (-1, 2)), np.reshape(train_hard, (-1, 2))
        train_easy = np.reshape(train_easy, (-1, 2))
        test = np.reshape(test, (-1, 2))

        np.random.shuffle(train_easy)  # Shuffle data
        np.random.shuffle(test)

        # Testing reduce scope
        train = train_easy[0:n_data, 0:n_data]
        test = test[0:n_data, 0:n_data]

        if lowercase is True:
            np.char.lower(train)
            np.char.lower(test)

        return train[:,0], train[:,1], test[:,0], test[:,1]