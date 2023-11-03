# predictor.py

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


# Constants
IMG_SIZE = 224
BATCH_SIZE = 32

labels_csv = pd.read_csv("./csv/labels.csv")
labels = labels_csv["breed"].to_numpy()
unique_breeds = np.unique(labels)


def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    print(f"Model fetched!!!")
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
    image = preprocess_image(image_path)
    image = tf.expand_dims(
        image, axis=0
    )  # expand dimension to fit the model's input shape
    custom_preds = model.predict(image)

    top_k_values, top_k_indices = tf.nn.top_k(custom_preds, k=5)
    top_k_probs = top_k_values.numpy()[0]
    top_k_labels = np.array(unique_breeds)[top_k_indices.numpy()[0]]

    # Sort the breeds based on probabilities in descending order and round the probabilities
    sorted_breeds_with_probs = dict(
        sorted(
            {
                label: round(prob * 100, 2)
                for label, prob in zip(top_k_labels, top_k_probs)
            }.items(),
            key=lambda item: item[1],
            reverse=False,
        )
    )

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    top_label = list(sorted_breeds_with_probs.keys())[0]
    plt.title(f"{top_label}: {sorted_breeds_with_probs[top_label] * 100:.2f}%")
    plt.imshow(image[0])
    print(top_k_probs)
    print(top_k_labels)

    return sorted_breeds_with_probs
