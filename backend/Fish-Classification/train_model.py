import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#SETUP AND DATA LOADING 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
TRAIN_DIR = 'data/seg_train/seg_train'#Path to the new training folder
TEST_DIR = 'data/seg_test/seg_test'    #Path to the new testing folder

# Use ImageDataGenerator for the training set with augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Use a separate, simpler generator for the test set (no augmentation)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Loading validation data...")
validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# BUILD THE MODEL WITH TRANSFER LEARNING 
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
num_classes = len(train_generator.class_indices)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# COMPILE AND TRAIN
print("Compiling the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting model training... This may take a few minutes.")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# SAVE THE FINAL MODEL
model.save('data/fish_classifier_model.h5')
print("-" * 20)
print("✅ Training complete! Model saved as fish_classifier_model.h5")