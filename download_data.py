import tensorflow as tf
import tensorflow_datasets as tfds

def download_plant_village():
    print("Connecting to PlantVillage repository...")
    
    # We download the 'plant_village' dataset
    # 'as_supervised=True' gives us (image, label) pairs
    data, info = tfds.load('plant_village', with_info=True, as_supervised=True)
    
    print("\n--- HELENAGRITECH DATASET REPORT ---")
    print(f"Total Classes: {info.features['label'].num_classes}")
    print(f"Dataset Size: {info.splits['train'].num_examples} images")
    
    # List the crops we can now detect
    classes = info.features['label'].names
    print(f"Sample Crops: {', '.join(classes[:5])}...")
    
    return data

if __name__ == "__main__":
    download_plant_village()