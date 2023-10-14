from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint="http://localhost:8502/v1/models/tomatoes_model:predict" # Using the latest version

CLASS_NAMES = ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data))) # reads the bytes as pillow image and converts into numpy array
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read()) # numpy array
    # Images are converted into batches
    img_batch = np.expand_dims(image,0)
    #prediction = MODEL.predict(img_batch)

    json_data = {
        "instances" : img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = (response.json()["predictions"][0]) # batch of images and returning the first image

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": confidence
    }
    

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)