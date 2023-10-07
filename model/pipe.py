# predictor.py

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

BATCH_SIZE = 32

labels_csv = pd.read_csv("csv/labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)

# Load the trained model with your custom load_model function
def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
    return model

# Function to preprocess an image
def preprocess_image(image_path):
    """
    Takes an image file path and turns it into a preprocessed Tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

# Function to create data batches
def create_data_batches(x, batch_size=BATCH_SIZE):
    """
    Creates batches of data out of image (x).
    """
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch = data.map(preprocess_image).batch(batch_size)
    return data_batch

def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]

# Function to predict the dog breed
def predict_breed(image_path, model):
    custom_image_paths = [image_path + fname for fname in os.listdir(image_path)]
    custom_data = create_data_batches(custom_image_paths)
    custom_preds = model.predict(custom_data)
    custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]

    plt.figure(figsize=(10, 10))
    for i, image in enumerate(custom_data.unbatch().as_numpy_iterator()):
        plt.subplot(1, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(custom_pred_labels[i])
        plt.imshow(image)

    return custom_pred_labels  # Return predicted labels

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
