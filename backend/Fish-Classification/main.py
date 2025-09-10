import uvicorn
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from contextlib import asynccontextmanager

#DEFINE MODEL LOADING AND LIFESPAN

MODEL_PATH = 'data/fish_classifier_model.h5'
TRAIN_DIR = 'data/seg_train/seg_train' # Path to new training folder
IMAGE_SIZE = (224, 224)
model = None
class_names = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global model, class_names
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Fish prediction model loaded successfully!")
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
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
    
    yield
#INITIALIZE THE API
app = FastAPI(title="Fish Prediction API", lifespan=lifespan)

# DEFINE THE PREDICTION ENDPOINT

@app.post("/predict/fish_species")
async def predict_fish_species(file: UploadFile = File(...)):
    if not model:
        return {"error": "Model is not loaded correctly. Please check server logs."}
        
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
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