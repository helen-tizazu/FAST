import os

# This is the exact 100% correct path you found
path = r"C:\Users\helen\tensorflow_datasets\downloads\extracted\ZIP.data.mend.com_publ-file_data_tywb_file_d565-c1rDQyRTmE0CqGGXmH53WlQp0NWefMfDW89aj1A0m5D_A\Plant_leave_diseases_dataset_without_augmentation"

print("\n--- HELENAGRITECH DATA VERIFICATION ---")

if os.path.exists(path):
    # This looks inside the folder and counts the categories
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    print(f"✅ SUCCESS: Found {len(folders)} disease categories!")
    print(f"Location: {path}")
    
    # Let's see the first few to be sure
    if len(folders) > 0:
        print(f"Categories detected: {folders[:3]} and many more...")
else:
    print("❌ ERROR: The path is still not being recognized by Python.")