import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pdb
import numpy


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

class SetTrace(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        model = self.model
        pdb.set_trace()

class ModelOutput:
    ''' Class wrapper for a metric that stores the output passed to it '''
    def __init__(self, name):
        self.name = name
        self.y_true = None
        self.y_pred = None

    def save_output(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return tf.constant(True)

class ModelOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_outputs):
        tf.keras.callbacks.Callback.__init__(self)
        self.model_outputs = model_outputs

    def on_train_batch_end(self, batch, logs=None):
        pass
        # use self.model_outputs to get the outputs here

