import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt

# 1. Load the brain you just created
model = tf.keras.models.load_model('helen_agritech_model.h5')

# 2. Get the category names (in the same order as training)
data_dir = r"C:\Users\helen\tensorflow_datasets\downloads\extracted\ZIP.data.mend.com_publ-file_data_tywb_file_d565-c1rDQyRTmE0CqGGXmH53WlQp0NWefMfDW89aj1A0m5D_A\Plant_leave_diseases_dataset_without_augmentation"
class_names = sorted(os.listdir(data_dir))

# 3. Pick a random image to test
random_folder = random.choice(class_names)
folder_path = os.path.join(data_dir, random_folder)
random_image = random.choice(os.listdir(folder_path))
img_path = os.path.join(folder_path, random_image)

# 4. Process the image for the AI
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# 5. Make the prediction
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
result = class_names[np.argmax(score)]

print(f"\n--- HELENAGRITECH PREDICTION ---")
print(f"Actual: {random_folder}")
print(f"AI Guess: {result}")
print(f"Confidence: {100 * np.max(score):.2f}%")

# Show the leaf
plt.imshow(img)
plt.title(f"AI says: {result}")
plt.axis('off')
plt.show()