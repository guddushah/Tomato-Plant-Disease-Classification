from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "tdc-tf-models"
CLASS_NAMES = ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Healthy"]

model=None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/tomato.h5",
            "/tmp/tomato.h5"
        )
        model = tf.keras.models.load_model("/tmp/tomato.h5")

    image = request.files["file"] # like uploading file in postman

    image = np.array(Image.open(image).convert("RGB").resize((256,256)))
    image=image/255
    image_array = tf.expand_dims(image,0) # converting the image into batch


    predictions = model.predict(image_array)

    print("predictions: ",predictions)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])),2)

    return {
        "class": predicted_class,
        "confidence": confidence
    }