import tensorflow as tf
import numpy as np
from tensorflow import keras
import parameters as p
from preprocessing import processor
from network_debugging import plot_history
import definitions
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

processor = processor()
train_x, train_y, test_x, test_y = processor.get_data()
encoder_input_data, decoder_input_data, decoder_target_data = processor.encode_decode_preprocess([train_x, train_y])

latent_dim = p.hidden_size
num_decoder_tokens = p.vocab_size
num_encoder_tokens = p.vocab_size

encoder_inputs = keras.layers.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.layers.Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit([encoder_input_data, decoder_input_data],
                    decoder_target_data,
                    batch_size=64,
                    epochs=100,
                    validation_split=0.2)

interp_encoder_input_data, interp_decoder_input_data, interp_decoder_target_data = processor.encode_decode_preprocess([test_x, test_y])
interpolate_accuracy = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data)
print(f'\n\nInterpolate Test set\n  Loss: {interpolate_accuracy[0]}\n  Accuracy: {interpolate_accuracy[1]}')

# Define sampling models
encoder_model = keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = keras.layers.Input(shape=(latent_dim,))
decoder_state_input_c = keras.layers.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.models.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_target_char_index = dict(
    (i, char) for char, i in p.vocab_table.items())

# model_version = '001'

# model_name = "encoder_model"
# model_path = os.path.join(definitions.ROOT_DIR, "saved_models/encoder_decoder_lstm", model_name+'_'+model_version)
# encoder_model.save(model_path)
#
# model_name = "decoder_model"
# model_path = os.path.join(definitions.ROOT_DIR, "saved_models/encoder_decoder_lstm", model_name+'_'+model_version)
# decoder_model.save(model_path)

def decode_sequence(input_seq):
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

# plot_history(history)

input_sentences=[]
input_targets=[]
decoded_sentences=[]
for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.

    input_seq = encoder_input_data[seq_index: seq_index + 1]  # inputs 1, 160, 98 matrix
    decoded_sentence = decode_sequence(input_seq)

    input_sentences.append(repr(train_x[seq_index]))
    input_targets.append(repr(train_y[seq_index]))
    decoded_sentences.append(decoded_sentence)

    print('-')
    print(f'Input sentence: {repr(train_x[seq_index]), repr(train_y[seq_index])}')
    print(f'Decoded sentence: {repr(decoded_sentence)}')

dir_results = os.path.join(definitions.ROOT_DIR, "results", "encoder_decoder_lstm_001.txt")
with open(dir_results, 'w') as file:
    file.write(f'Train Test set\n  Loss: {interpolate_accuracy[0]}\n  Accuracy: {interpolate_accuracy[1]}\n\nPrediction Sampling\n')

    for input_sentence,input_target,decoded_sentence in zip(input_sentences, input_targets, decoded_sentences):
        file.write(f'Input: {input_sentence} | {input_target}\n\t{repr(decoded_sentence)}\n')
    file.close()