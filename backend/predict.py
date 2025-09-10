import tensorflow as tf
import numpy as np
import sys

#CONFIGURATION
MODEL_PATH = 'data/fish_classifier_model.h5'
IMAGE_SIZE = (224, 224)
TRAIN_DIR = 'data/seg_train/seg_train' # Path to the new training folder

# LOAD THE TRAINED MODEL 
print("Loading the trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

#  LOAD THE CLASS NAMES
try:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=1,
        class_mode='categorical'
    )
    class_names = {v: k for k, v in train_generator.class_indices.items()}
    print(f"Model is ready. Found the following classes: {list(class_names.values())}")
except FileNotFoundError:
    print(f"Error: Directory not found at '{TRAIN_DIR}'. Please check the path.")
    class_names = {}
except Exception as e:
    print(f"An error occurred while loading class names: {e}")
    class_names = {}


# PREDICTION FUNCTION
def predict_image(image_path):
    if not class_names:
        print("Cannot make predictions because class names were not loaded.")
        return

    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        
        predicted_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_index]
        confidence = np.max(predictions[0]) * 100

        print("-" * 20)
        print(f"✅ Prediction Successful!")
        print(f"   -> Species: {predicted_class_name}")
        print(f"   -> Confidence: {confidence:.2f}%")
        print("-" * 20)
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

#  SCRIPT EXECUTION 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_to_predict = sys.argv[1]
        predict_image(image_to_predict)
    else:
        print("Please provide an image path to predict.")
        print("Usage: python predict.py path/to/your/image.jpg")