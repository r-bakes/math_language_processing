import numpy as np
from tensorflow import keras
import parameters as p
from depricated.data_preprocessing import Processor
import os
np.set_printoptions(threshold=np.inf)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Grab Data
processor = Processor()
train_x, train_y, test_x, test_y = processor.get_data()
train_y_comparison = train_y
test_y_comparison = test_y
train_x, train_y, test_x, test_y = processor.preprocess(train_x), processor.preprocess(train_y), processor.preprocess(test_x), processor.preprocess(test_y)

model = keras.models.Sequential([
    keras.layers.Masking(mask_value=0.,
                         input_shape=(None, p.vocab_size)),
    keras.layers.LSTM(p.hidden_size,
                      return_sequences=True,
                      input_shape=[None, p.vocab_size],
                      dropout=0.2,
                      recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(p.vocab_size,
                                                    activation="softmax"))])

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=[keras.metrics.CategoricalAccuracy()])


model.fit(
    train_x, train_y,
    batch_size=64,
    epochs=1,
    validation_split=0.2)

model.summary()

print("True Value:", train_y_comparison[1:2][-1])
print(train_y.shape)
print(train_y[1:2])
prediction = model.predict(train_x[1:2])
print(prediction.shape)
print(prediction.round())

print(processor.tokenizer.word_index)

interpolate_accuracy = model.evaluate(test_x, test_y)
print(f'\n\nInterpolate Test set\n  Loss: {interpolate_accuracy[0]}\n  Accuracy: {interpolate_accuracy[1]}')


# Saving Bug wont work unless tf 2.0.1 installed (not on conda yet)
# model_version = "001"
# model_name = "simple_lstm"
# model.save(os.path.join(r"C:\Users\bakes\OneDrive - The Pennsylvania State University\devops\repos\thesis_mathematical_capabilities_neural_architectures\saved_models", model_name, model_version))
#



#
# from architectures.preprocessing import Preprocessor
# import parameters as params
#
# import pdb
# import logging
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# logger = logging.getLogger('results')
# logger.setLevel(logging.INFO)
#
# data = Preprocessor(test_mode=True)
# # data.sequence()
# data.encoder_decoder_one_hot()
#
#
# # Embedding layer for sequenced inputs
# embedding = Embedding(input_dim=data.alphabet_len,
#                       output_dim=data.alphabet_len,
#                       input_length=None)  # leave None -- feature and targets have variable sequence length
#
# # Encoder layer
# encoder_inputs = Input(shape=(None, data.alphabet_len))  # Define encoder input
#
# encoder_lstm = LSTM(params.hidden_size,
#                     return_state=True)
#
# encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# encoder_states = [state_h, state_c]  # Discard encoder's outputs
#
#
# # Decoder layer
# decoder_inputs = Input(shape=(None, data.alphabet_len))
#
# decoder_lstm = LSTM(params.hidden_size,
#                     return_sequences=True,
#                     return_state=True)
#
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                      initial_state=encoder_states)
#
# decoder_dense = Dense(data.alphabet_len, activation='softmax')  # Why do we need a dense layer to accept our decoder outputs?
#
# decoder_outputs = decoder_dense(decoder_outputs)
#
# # Model Instantiation
# # Model Parameters
# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#
# model_optimizer = Adam(lr=params.learning_rate,
#                        beta_1=params.beta1,
#                        beta_2=params.beta2,
#                        epsilon=params.epsilon,
#                        decay=0.0,
#                        amsgrad=False,
#                        clipvalue=params.abs_gradient_clipping)
#
# model.compile(optimizer=model_optimizer,
#               loss='categorical_crossentropy',
#               metrics=[metrics.binary_accuracy])
#
# model.fit([data.train_encoder_x,
#            data.train_decoder_x],
#           data.train_decoder_y,
#           batch_size=64,
#           epochs=1,
#           validation_split=0.2)
#
#
# extrapolate_accuracy = model.evaluate([data.test_extrapolate_encoder_x,
#                                        data.test_extrapolate_decoder_x],
#                                       data.test_extrapolate_decoder_y)
#
# interpolate_accuracy = model.evaluate([data.test_interpolate_encoder_x,
#                                        data.test_interpolate_decoder_x],
#                                       data.test_interpolate_decoder_y)
#
# logger.info('\n\nExtrapolate test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(extrapolate_accuracy[0],
#                                                                                     extrapolate_accuracy[1]))
# logger.info('\n\nInterpolate Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(interpolate_accuracy[0],
#                                                                                     interpolate_accuracy[1]))
