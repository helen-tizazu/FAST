import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. SET THE DATA PATH (The one we verified!)
data_dir = r"C:\Users\helen\tensorflow_datasets\downloads\extracted\ZIP.data.mend.com_publ-file_data_tywb_file_d565-c1rDQyRTmE0CqGGXmH53WlQp0NWefMfDW89aj1A0m5D_A\Plant_leave_diseases_dataset_without_augmentation"

# 2. PREPARE THE DATA (Loading and Resizing)
print("--- Loading Images ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # Save 20% for testing
    subset="training",
    seed=123,
    image_size=(224, 224), # Standard size for MobileNet
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# 3. BUILD THE BRAIN (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Use the pre-learned knowledge from Google

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(39, activation='softmax') # 39 buttons for your 39 categories!
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. START TRAINING
print("\n--- STARTING AI TRAINING ---")
model.fit(train_ds, validation_data=val_ds, epochs=3) # Let's start with 3 rounds

# 5. SAVE THE FINISHED BRAIN
model.save("helen_agritech_model.h5")
print("\n✅ DONE! Your AI model is saved as 'helen_agritech_model.h5'")