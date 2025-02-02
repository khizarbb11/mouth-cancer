import os
import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import io

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

# Load the model locally from the repository

# MODEL_PATH = "Latest_oscc_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=14yqq0zwJcBtA8XXPeOXaXlPplbbPqOV-"
MODEL_PATH = os.path.join(os.getcwd(), "Latest_oscc_model.h5")
print(f"Current working directory: {os.getcwd()}")
print(f"Model path: {MODEL_PATH}")

# Download the model if not already present
# if not os.path.exists(MODEL_PATH):
print(f"Downloading model from {MODEL_URL}...")
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
print(f"Model downloaded to {MODEL_PATH}")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
else:
    print("Model file found.")

# Load the model correctly
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

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
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()