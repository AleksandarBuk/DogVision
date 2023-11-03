from flask import Flask, render_template, request, flash
import boto3
import os
import predictor
import secrets
import uuid
import threading
import time
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

app.secret_key = secrets.token_hex(32)
model_path = "model/saved_model/20231007-15391696693193-all-images-Adam.h5"
loaded_model = predictor.load_model(model_path)
UPLOAD_FOLDER = "static/uploaded"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

AWS_ACCESS_KEY_ID = "aws_access_key"
AWS_SECRET_ACCESS_KEY = "aws_secret_access_key"
S3_BUCKET_NAME = "dog-bucket"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="region",
)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def delayed_delete_file(file_path, delay):
    time.sleep(delay)
    try:
        os.remove(file_path)
        logging.info(f"Image {file_path} successfully deleted after {delay} seconds.")
    except Exception as e:
        logging.error(f"Error deleting {file_path} after {delay} seconds: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    dog_breeds_with_probs = None
    uploaded_image_path = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file and allowed_file(uploaded_file.filename):
            unique_filename = (
                f"{uuid.uuid4()}.{uploaded_file.filename.rsplit('.', 1)[1]}"
            )
            local_uploaded_image_path = os.path.join(
                app.config["UPLOAD_FOLDER"], unique_filename
            )
            uploaded_file.save(local_uploaded_image_path)

            threading.Thread(
                target=delayed_delete_file, args=(local_uploaded_image_path, 300)
            ).start()

            # Predict the breed
            try:
                dog_breeds_with_probs = predictor.predict_breed(
                    local_uploaded_image_path, loaded_model
                )
                print(dog_breeds_with_probs)
            except Exception as e:
                flash(f"Error during breed prediction: {e}", "danger")

            # Upload the image to AWS S3
            try:
                s3.upload_file(
                    local_uploaded_image_path,
                    S3_BUCKET_NAME,
                    unique_filename,
                    ExtraArgs={"ContentType": uploaded_file.content_type},
                )

                logging.info(f"Image {unique_filename} successfully uploaded to S3.")
            except Exception as e:
                flash(f"Error during S3 upload: {e}", "danger")
                logging.error(f"Error during S3 upload: {e}")

            # Use the local path for rendering in the template
            uploaded_image_path = f"/{local_uploaded_image_path}"
        else:
            flash("Allowed image types are -> jpg, jpeg, png", "danger")

    return render_template(
        "index.html",
        dog_breed=dog_breeds_with_probs,
        uploaded_image=uploaded_image_path,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)
