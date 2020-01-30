import tensorflow as tf
from tensorflow import keras
import numpy as np
import parameters as p

import os
import pdb


class processor:

    dir_train_data = r"C:\Users\bakes\OneDrive - The Pennsylvania State University\devops\repos\thesis_math_language_processing\data\train-easy\algebra__linear_1d.txt"
    dir_test_data = r"C:\Users\bakes\OneDrive - The Pennsylvania State University\devops\repos\thesis_math_language_processing\data\interpolate\algebra__linear_1d.txt"

    def __init__(self):
        self.tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, lower=False)
        self.tokenizer.fit_on_texts(p.vocab)

    def preprocess(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=p.max_question_length, padding='pre', truncating='pre', value=0.0)
        return tf.one_hot(sequences, p.vocab_size)

    def encode_decode_preprocess(self, texts):
        input_texts = np.array(texts[0])
        target_texts = np.array(texts[1])
        max_encoder_seq_length = p.max_question_length
        max_decoder_seq_length = p.max_answer_length
        num_encoder_tokens = p.vocab_size
        num_decoder_tokens = p.vocab_size

        target_texts = np.char.add(np.full(shape=len(target_texts), fill_value='\t'), target_texts)
        target_texts = np.char.add(target_texts, np.full(shape=len(target_texts), fill_value='\n'))
        print(target_texts)


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
            encoder_input_data[i, t + 1:, p.vocab_table[' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, p.vocab_table[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, p.vocab_table[char]] = 1.
            decoder_input_data[i, t + 1:, p.vocab_table[' ']] = 1.
            decoder_target_data[i, t:, p.vocab_table[' ']] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data

    def get_data(self):
        train = open(self.dir_train_data, 'r').read().splitlines()
        test = open(self.dir_test_data, 'r').read().splitlines()

        train = np.reshape(train, (-1, 2))
        test = np.reshape(test, (-1, 2))

        train = train[0:100, 0:100]  # Testing reduce scope
        test = test[0:100, 0:100]

        return train[:,0], train[:,1], test[:,0], test[:,1]

    # def postprocess(self, y_pred):
    #
    #     print(self.tokenizer.sequences_to_texts(y_pred[0]))
    #     print(self.tokenizer.sequences_to_texts(y_pred[0])[0])




# processer = processer()
# test_x, test_y, train_x, train_y = processer.get_data()
#
#
# # print(test[0,0])
# test_x = processer.preprocess(test_x)
# # print(test[:,0])
# print(len(test_x[0,0]))
# print(test_x.shape)
# print(test_x[0].shape)
# print(test_x[0,0])
# # print(test[1,:])
# # test = test.reshape(1,:,1)
# print(test_x)

