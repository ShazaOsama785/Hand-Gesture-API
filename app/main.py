from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator

# ------------------------
# Setup logging
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Load model and inverse mapping
# ------------------------
try:
    model_path = "app/models/best_model_production.pkl"
    mapping_path = "app/models/inverse_mapping.json"

    if not os.path.exists(model_path) or not os.path.exists(mapping_path):
        raise FileNotFoundError("Model or mapping file not found.")

    model = joblib.load(model_path)
    logger.info("Model loaded successfully")

    with open(mapping_path, "r") as f:
        inverse_mapping = json.load(f)
        inverse_mapping = {int(k): v for k, v in inverse_mapping.items()}
    logger.info("Inverse mapping loaded successfully")

except Exception as e:
    logger.error(f"Model or mapping load failed: {e}")
    raise

# ------------------------
# Create FastAPI app
# ------------------------
app = FastAPI()
Instrumentator().instrument(app).expose(app)
# ------------------------
# Input schema
# ------------------------
class HandLandmarkInput(BaseModel):
    landmarks: list[float]  # Flat list of 63 floats (21 landmarks × 3 coordinates)

# ------------------------
# Routes
# ------------------------
@app.get("/")
def home():
    logger.info("Home endpoint hit")
    return {"message": "Hand Gesture Recognition API is live"}

@app.get("/health")
def health():
    logger.info("Health check passed")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: HandLandmarkInput):
    try:
        logger.info(f"Received landmarks of length {len(data.landmarks)}")

        # Validate input
        if len(data.landmarks) != 63:
            raise ValueError("Expected 63 values (21 landmarks * 3 coordinates)")

        # Convert to 2D array
        input_array = np.array(data.landmarks).reshape(1, -1)

        # Predict
        prediction = model.predict(input_array)[0]
        gesture = inverse_mapping.get(prediction, "Unknown")

        logger.info(f"Predicted gesture: {gesture}")
        return {"gesture": gesture}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
