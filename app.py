# app.py

from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import predictor  # Import functions and constants from predictor.py
import secrets

# Generate a random 32-character hexadecimal secret key
secret_key = secrets.token_hex(32)
print(secret_key)

app = Flask(__name__)

# Load the model using your custom function
model_path = 'model/saved_model/20231007-15391696693193-all-images-Adam.h5'
loaded_model = predictor.load_model(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    dog_breed = None

    if request.method == "POST":
        uploaded_file = request.files["image"]
        if uploaded_file.filename != "":
            # Save the uploaded image to a temporary folder
            image_path = os.path.join("static/uploaded", uploaded_file.filename)
            uploaded_file.save(image_path)

            # Predict the dog breed using the predictor function
            dog_breed = predictor.predict_breed(image_path, loaded_model)

    return render_template("index.html", dog_breed=dog_breed)

if __name__ == "__main__":
    app.run(debug=True)
