import tensorflow as tf
import tensorflow_hub as hub
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMG_SIZE = 224
BATCH_SIZE = 32

labels_df = pd.read_csv("./csv/labels.csv")
labels = labels_df["breed"].to_numpy()

unique_breeds = np.unique(labels)


def get_pred_label(prediction_probabilities):
    """
    Turns an array of prediction probabilities into a label.
    """
    return unique_breeds[np.argmax(prediction_probabilities)]


def predict_breed(image_path, model):
    image_directory = os.path.dirname(image_path)
    custom_image_paths = [os.path.join(image_directory, fname) for fname in os.listdir(image_directory)]
    custom_data = create_data_batches(custom_image_paths)
    custom_preds = model.predict(custom_data)

    top_k_values, top_k_indices = tf.nn.top_k(custom_preds, k=5)
    top_k_probs = tf.math.reduce_max(top_k_values, axis=-1).numpy()
    top_k_labels = np.array(unique_breeds)[top_k_indices.numpy()]

    breed_with_probs = []
    for i in range(len(top_k_probs)):
        breed_with_probs.append(
            {label: prob for label, prob in zip(top_k_labels[i], top_k_probs[i])}
        )

    num_images = len(custom_image_paths)
    num_columns = min(num_images, 3)
    num_rows = num_images // num_columns + (num_images % num_columns > 0)
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(custom_data.unbatch().as_numpy_iterator()):
        plt.subplot(num_rows, num_columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        top_label = list(breed_with_probs[i].keys())[0]
        plt.title(f"{top_label}: {breed_with_probs[i][top_label] * 100:.2f}%")  # Displaying top prediction's percentage
        plt.imshow(image)

    return breed_with_probs


def create_data_batches(x, batch_size=BATCH_SIZE):
    """
    Creates batches of data out of image (x).
    """
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch = data.map(preprocess_image).batch(batch_size)
    return data_batch


def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loading saved model from: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
    return model


def preprocess_image(image_path):
    """
    Takes an image file path and turns it into a preprocessed Tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image
