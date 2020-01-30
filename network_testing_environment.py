import tensorflow as tf
from tensorflow import keras
import numpy as np
import parameters as p
from preprocessing import processor
import os

model_dir = ""

model = keras.models.load_model(model_dir)

# Grab data

predictions = model.predict()

