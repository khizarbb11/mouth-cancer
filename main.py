import os
import io
import requests
import h5py
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uvicorn
import nest_asyncio
from huggingface_hub import hf_hub_download

# Function to load model directly from a URL (e.g., Google Drive or Cloud Storage)
def load_model_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        try:
            with h5py.File(io.BytesIO(response.content), "r") as model_file:
                return tf.keras.models.load_model(model_file)
        except Exception as e:
            raise Exception(f"Error loading model: {e}")
    else:
        raise Exception(f"Failed to fetch model: {response.status_code}")

# **Option 1**: Load model from a cloud storage URL (e.g., Google Drive or custom cloud URL)
# MODEL_URL = "https://your-cloud-link.com/Latest_oscc_model.h5"  # Update with your actual cloud model URL

# print("Downloading and loading model...")
# try:
#     model = load_model_from_url(MODEL_URL)
#     print("Model loaded successfully from URL")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     raise

# **Option 2**: Load model from Hugging Face Hub
MODEL_PATH = hf_hub_download(repo_id="khizarali07/my-oscc-model", filename="Latest_oscc_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully from Hugging Face Hub")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        contents = await file.read()
        file.file.close()

        img = image.load_img(io.BytesIO(contents), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        print(f"First pixel value: {img_array[0, 0, 0, :]}")

        # Predict using the model
        prediction = model.predict(img_array)[0][0]

        # Format response
        result = {
            "prediction": "OSCC" if prediction > 0.5 else "Normal",
            "confidence": float(prediction),
        }

        return JSONResponse(content=result, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

# Allow FastAPI to run inside Jupyter Notebook
nest_asyncio.apply()

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
