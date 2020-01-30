import tensorflow as tf
import numpy as np
import string
from tensorflow import keras
import parameters as p
from preprocessing import processor
import os


class PositionalEncoding(keras.layers.Layer):
    def __init__( self, max_steps, max_dims, dtype = tf.float32, ** kwargs):
        super().__init__( dtype = dtype, ** kwargs)
        if max_dims % 2 == 1:
            max_dims += 1 # max_dims must be even
            p, i = np.meshgrid( np.arange(max_steps), np.arange( max_dims // 2))
            pos_emb = np.empty(( 1, max_steps, max_dims))
            pos_emb[ 0, :, :: 2] = np.sin( p / 10000**( 2 * i / max_dims)).T
            pos_emb[ 0, :, 1:: 2] = np.cos( p / 10000**( 2 * i / max_dims)).T
            self.positional_embedding = tf.constant( pos_emb.astype( self.dtype))

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]


embed_size = 512
attentional_heads = 8
key_value_sizes = embed_size/attentional_heads

encoder_inputs = keras.layers.Input(shape=[ None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[ None], dtype=np.int32)

embeddings = keras.layers.Embedding(p.vocab_size, embed_size)

encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)

positional_encoding = PositionalEncoding(p.max_question_length, max_dims=embed_size)

encoder_in = positional_encoding( encoder_embeddings)
decoder_in = positional_encoding( decoder_embeddings)

