import tensorflow as tf
from tensorflow import keras
import numpy as np

from src import parameters as p
from depricated.data_preprocessing import Processor


def decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens):
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


print("Eager Execution:", tf.executing_eagerly())
processor = Processor(q_type=p.q_list[0])

train_x, train_y, test_x, test_y = processor.get_data(n_data=1)
encoder_input_data, decoder_input_data, decoder_target_data = processor.encoder_decoder_hot_preprocess([train_x, train_y])
encoder_input

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

#
# model.summary()
# # Compile
# model.compile(optimizer='adam', loss='categorical_crossentropy')

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

loss_history = []
logit_history = []
grads_history = []
target_history = []


def train_step(encoder_in, decoder_in, decoder_tar, epoch):
    with tf.GradientTape() as tape:
        logits = model(([np.array([encoder_in]), np.array([decoder_in])], np.array([decoder_tar])), training=True)
        logit_history.append(logits)
        loss_value = loss_object(decoder_tar, logits)
        target_history.append(decoder_tar)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, model.trainable_variables)
    grads_history.append([grad.numpy() for grad in grads])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train(epochs):
    for epoch in range(epochs):
        for batch, (encoder_in, decoder_in, decoder_tar) in enumerate(zip(encoder_input_data, decoder_input_data, decoder_target_data)):
            train_step(encoder_in, decoder_in, decoder_tar, epoch)
        print('Epoch {} finished'.format(epoch))


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


train(3)


loss_history = np.array(loss_history)
logit_history = np.array(logit_history)
grads_history = np.array(grads_history)

target_history = np.array(target_history)


for seq_index in range(1):
    # Take one sequence (part of the training set)
    # for trying out decoding.

    input_seq = encoder_input_data[seq_index: seq_index + 1]  # inputs 1, 160, 98 matrix
    decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens)

    input_sentences.append(repr(train_x[seq_index]))
    input_targets.append(repr(train_y[seq_index]))
    decoded_sentences.append(decoded_sentence)

    print('-')
    print(f'Input sentence: {repr(train_x[seq_index]), repr(train_y[seq_index])}')
    print(f'Decoded sentence: {repr(decoded_sentence)}')

