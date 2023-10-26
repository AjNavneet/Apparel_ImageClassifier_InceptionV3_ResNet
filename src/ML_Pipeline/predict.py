from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import metrics
import numpy as np


def load_model(path):
    # Load a Keras model from the given path
    
    try:
        model = models.load_model(path)
    except FileNotFoundError:
        print("Model does not exist")
    except:
        print("Failed to load model")
        
    return model


def predict_with_model(model, data):
    # Make predictions using a trained Keras model
    
    # You may want to add an assertion to check if 'data' is a numpy array or a data generator
    # assert type(data) ==  # Add the desired data type for 'data' here

    predictions = model.predict(data)
    predictions = np.where(predictions > 0.5, 1, 0)
    
    return predictions
