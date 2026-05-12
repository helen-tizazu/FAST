import tensorflow as tf
from tensorflow.keras import layers, models

# Updated with your exact directory path
data_dir = r"C:\Users\helen\tensorflow_datasets\downloads\extracted\ZIP.data.mend.com_publ-file_data_tywb_file_d565-c1rDQyRTmE0CqGGXmH53WlQp0NWefMfDW89aj1A0m5D_A\Plant_leave_diseases_dataset_without_augmentation"

# --- LOAD DATA ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
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

# --- BUILD THE MODEL ---
# Using MobileNetV2 for efficiency
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./127.5, offset=-1), 
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(39, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- TRAIN ---
# This will begin the training process
model.fit(train_ds, validation_data=val_ds, epochs=5) 

# Save the trained model
model.save("helen_agritech_model.h5")

# --- PRINT LABELS ---
print("\n--- COPY AND PASTE THIS LIST INTO app.py ---")
print(train_ds.class_names)