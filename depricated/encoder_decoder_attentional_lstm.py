from tensorflow import keras
import numpy as np
import os
import datetime

from depricated import data_preprocessing
from src import parameters as p
import definitions


class EncoderDecoderLSTM:

    def __init__(self, n_train, n_epochs, q_type, exp_name):
        self.n_train = n_train
        self.n_epochs = n_epochs
        self.q_type = q_type
        self.exp_name = exp_name

    def decode_sequence(self, input_seq, encoder_model, decoder_model, num_decoder_tokens):
        reverse_target_char_index = dict(
            (i, char) for char, i in p.vocab_table.items())

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, p.vocab_table['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > p.max_answer_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]
        return decoded_sentence

    def train(self):
        experiment_time = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        processor = data_preprocessing.Processor(self.q_type)

        train_x, train_y, test_x, test_y = processor.get_data(n_data=self.n_train)
        if self.n_train == -1:
            self.n_train = len(train_x)

        encoder_input_data, decoder_input_data, decoder_target_data = processor.encoder_decoder_hot_preprocess([train_x, train_y])

        latent_dim = p.hidden_size
        num_decoder_tokens = p.vocab_size
        num_encoder_tokens = p.vocab_size

        encoder_inputs = keras.layers.Input(shape=(None, num_encoder_tokens))
        encoder_masking = keras.layers.Masking(mask_value=0.0)
        encoder = keras.layers.LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_masking(encoder_inputs))
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.layers.Input(shape=(None, num_decoder_tokens))
        decoder_masking = keras.layers.Masking(mask_value=0.0)

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_masking(decoder_inputs),
                                             initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.summary()
        # Run training
        adam = keras.optimizers.Adam(learning_rate=p.learning_rate, beta_1=p.beta1, beta_2=p.beta2, amsgrad=False)

        # Callback definitions
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(definitions.LOGDIR, f"{self.exp_name}_{processor.question_type[0:-4]}_ENCODER_DECODER_LSTM_{experiment_time}"))
        csv_logger = keras.callbacks.CSVLogger(filename=os.path.join(definitions.LOGDIR, f"{self.exp_name}_{processor.question_type[0:-4]}_ENCODER_DECODER_LSTM_{experiment_time}", f"training.log"))


        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        history = model.fit([encoder_input_data, decoder_input_data],
                            decoder_target_data,
                            batch_size=64,
                            epochs=self.n_epochs,
                            callbacks=[tensorboard_callback,
                                       csv_logger],
                            validation_split=0.2,
                            shuffle=True)


        interp_encoder_input_data, interp_decoder_input_data, interp_decoder_target_data = processor.encoder_decoder_hot_preprocess([test_x, test_y])
        interpolate_accuracy = model.evaluate([interp_encoder_input_data, interp_decoder_input_data], interp_decoder_target_data)

        print(f'\n\nInterpolate Test set\n  Loss: {interpolate_accuracy[0]}\n  Accuracy: {interpolate_accuracy[1]}')  #

        # Define sampling models
        encoder_model = keras.models.Model(encoder_inputs, encoder_states)

        decoder_state_input_h = keras.layers.Input(shape=(latent_dim,))
        decoder_state_input_c = keras.layers.Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_masking(decoder_inputs), initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        input_sentences = []
        input_targets = []
        decoded_sentences = []

        range_val = len(test_x)
        score = 0
        if self.n_train < 100: range_val=self.n_train  # debugging measure
        for seq_index in range(range_val):

            input_seq = interp_encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens)

            input_sentences.append(test_x[seq_index])
            input_targets.append(test_y[seq_index])

            decoded_sentences.append(decoded_sentence)

            if decoded_sentence[0:-1] == test_y[seq_index]: score+=1

        dir_results = os.path.join(definitions.ROOT_DIR, "results", f"{self.exp_name}_{processor.question_type[0:-4]}_ENCODER_DECODER_LSTM_{experiment_time}.txt")
        with open(dir_results, 'w') as file:
            file.write(f'ENCODER DECODER LSTM EXPERIMENT: {self.exp_name}_{processor.question_type[0:-4]}\t{experiment_time}\n\tEpochs: {self.n_epochs}\n\tSample Size: {self.n_train}\n')
            file.write(f'Interpolate Test set\n\tLoss: {interpolate_accuracy[0]}\n\tTF_reported_Accuracy: {interpolate_accuracy[1]}\n\tScore: {score/range_val}\n\nPrediction Sampling\n')

            for input_sentence, \
                input_target, \
                decoded_sentence in zip(input_sentences,
                                        input_targets,
                                        decoded_sentences):

                file.write(f'Input: {input_sentence} | {input_target}\n\t{repr(decoded_sentence)}\n')

            file.close()
