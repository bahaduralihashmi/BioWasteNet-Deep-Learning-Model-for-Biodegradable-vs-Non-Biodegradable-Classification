import os
import hashlib
from PIL import Image

def image_hash(img_path):
    """Compute a hash for an image file."""
    with Image.open(img_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

train_folder = r"C:/Users/bahad/OneDrive/Desktop/ultralytics/person_datasets/train/images"
val_folder = r"C:/Users/bahad/OneDrive/Desktop/ultralytics/person_datasets/val/images"

# Compute hashes for all images
train_hashes = {image_hash(os.path.join(train_folder, f)): f for f in os.listdir(train_folder)}
val_hashes = {image_hash(os.path.join(val_folder, f)): f for f in os.listdir(val_folder)}

# Find duplicates
duplicates = set(train_hashes.keys()) & set(val_hashes.keys())

if duplicates:
    print("Duplicate images found in train and val folders:")
    for h in duplicates:
        print(f"Train: {train_hashes[h]}, Val: {val_hashes[h]}")
else:
    print("No duplicate images found between train and val folders.")
