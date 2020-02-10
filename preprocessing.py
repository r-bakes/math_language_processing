import tensorflow as tf
from tensorflow import keras
import numpy as np
import parameters as p
import definitions

import os
import pdb


class processor:

    dir_data = os.path.join(definitions.ROOT_DIR, "data")
    question_type = "algebra__linear_1d.txt"
    # arithmetic__add_or_sub.txt
    # algebra__linear_1d.txt

    def __init__(self):
        pass

    def preprocess_sequence(self, texts):
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

    def encoder_decoder_preprocess(self, texts):
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

    def get_data(self, n_data):
        train_easy = open(os.path.join(self.dir_data,r"train-easy", self.question_type), 'r').read().splitlines()
        train_medium = open(os.path.join(self.dir_data,r"train-medium",self.question_type), 'r').read().splitlines()
        train_hard = open(os.path.join(self.dir_data,r"train-hard",self.question_type), 'r').read().splitlines()
        test = open(os.path.join(self.dir_data, r"interpolate", self.question_type), 'r').read().splitlines()

        # focus on easy for now
        # train_medium, train_hard = np.reshape(train_medium, (-1, 2)), np.reshape(train_hard, (-1, 2))
        train_easy = np.reshape(train_easy, (-1, 2))
        test = np.reshape(test, (-1, 2))

        np.random.shuffle(train_easy)  # Shuffle data

        # Testing reduce scope
        train = train_easy[0:n_data, 0:n_data]
        test = test[0:n_data, 0:n_data]

        return train[:,0], train[:,1], test[:,0], test[:,1]
