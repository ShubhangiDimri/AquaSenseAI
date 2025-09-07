import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import shutil
from forecast import run_prophet, prepare_df
import pandas as pd

# ==============================================================================
# 1. INITIALIZE THE API & SETUP UPLOAD FOLDER
# ==============================================================================
app = FastAPI(title="Ocean AI API")
UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================================================================
# 2. LOAD THE FISH PREDICTION MODEL (runs only once on startup)
# ==============================================================================
MODEL_PATH = 'fish_classifier_model.h5'
DATASET_PATH = 'dataset_organized'
IMAGE_SIZE = (224, 224)
model = None
class_names = {}

# We wrap the model loading in a startup event for cleaner management
@app.on_event("startup")
def load_fish_model():
    global model, class_names
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Fish prediction model loaded successfully!")
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
        train_generator = train_datagen.flow_from_directory(
            DATASET_PATH,
            target_size=IMAGE_SIZE,
            batch_size=1,
            class_mode='categorical'
        )
        class_names = {v: k for k, v in train_generator.class_indices.items()}
        print(f"✅ Class names loaded: {list(class_names.values())}")
    except Exception as e:
        print(f"❌ Error loading fish model or class names: {e}")
        model = None
        class_names = {}

# ==============================================================================
# 3. DEFINE API ENDPOINTS
# ==============================================================================
@app.get("/")
def root():
    return {"message": "Ocean AI API running. Endpoints available at /docs"}

# --- Fish Prediction Endpoint ---
@app.post("/predict/fish_species")
async def predict_fish_species(file: UploadFile = File(...)):
    """
    Receives a fish image, makes a prediction, and returns the species.
    """
    if not model:
        return {"error": "Model is not loaded correctly. Please check server logs."}
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB") # Ensure 3 channels
    
    image = image.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    
    predicted_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_index]
    confidence = float(np.max(predictions[0]) * 100)
    
    return {
        "species": predicted_class_name,
        "confidence": f"{confidence:.2f}%"
    }

# --- Time-Series Forecast Endpoint ---
@app.post("/forecast/time_series")
async def forecast_time_series(file: UploadFile = File(...), date_col: str = None, target_col: str = None, future_days: int = 30):
    """
    Receives a CSV file, runs a forecast, and returns the results.
    """
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = run_prophet(file_path, date_col=date_col, target_col=target_col, future_periods=future_days, save_csv_path=os.path.join(UPLOAD_FOLDER, "forecast_output.csv"))
    
    return {
        "mae": result["mae"],
        "rmse": result["rmse"],
        "forecast_csv": result["forecast_csv"]
    }

# --- Data Preview Endpoint ---
@app.post("/preview/csv_data")
async def preview_csv(file: UploadFile = File(...)):
    """
    Receives a CSV and returns the first 10 rows after preparation.
    """
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    df = prepare_df(file_path)
    return df.head(10).to_dict(orient="records")

# ==============================================================================
# 4. RUN THE API (for local testing)
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
