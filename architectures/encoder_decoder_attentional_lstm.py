import tensorflow as tf
import numpy as np
from tensorflow import keras
import parameters as p
from preprocessing import processor
from network_debugging import plot_history
import pdb

processor = processor()
train_x, train_y, test_x, test_y = processor.get_data()
encoder_input_data, decoder_input_data, decoder_target_data = processor.encode_decode_preprocess([train_x, train_y])

latent_dim = 256
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
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=20,
          validation_split=0.2)

interpolate_accuracy = model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data)
print(f'\n\nTrain Test set\n  Loss: {interpolate_accuracy[0]}\n  Accuracy: {interpolate_accuracy[1]}')

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

print(reverse_target_char_index)

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
        print("token index:", repr(sampled_token_index))
        print("char:", repr(reverse_target_char_index[sampled_token_index]))
        sampled_char = reverse_target_char_index[sampled_token_index]
        print("char:", repr(sampled_char))
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

plot_history(history)

pdb.set_trace()

for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]  # inputs 1, 160, 98 matrix
    decoded_sentence = decode_sequence(input_seq)

    print('-')
    print(f'Input sentence: {repr(train_x[seq_index]), repr(train_y[seq_index])}')
    print(f'Decoded sentence: {repr(decoded_sentence)}')


