import os
import hashlib
from PIL import Image
import shutil

def image_hash(img_path):
    """Compute a hash for an image file."""
    with Image.open(img_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

train_folder = r"C:/Users/bahad/OneDrive/Desktop/ultralytics/person_datasets/train/images"
val_folder = r"C:/Users/bahad/OneDrive/Desktop/ultralytics/person_datasets/val/images"
backup_folder = r"C:/Users/bahad/OneDrive/Desktop/ultralytics/person_datasets/val_duplicates_backup"

os.makedirs(backup_folder, exist_ok=True)

# Compute hashes for all training images
train_hashes = {image_hash(os.path.join(train_folder, f)): f for f in os.listdir(train_folder)}

# Check validation images and move duplicates to backup
for val_file in os.listdir(val_folder):
    val_path = os.path.join(val_folder, val_file)
    try:
        h = image_hash(val_path)
        if h in train_hashes:
            # Move duplicate to backup folder
            shutil.move(val_path, os.path.join(backup_folder, val_file))
            print(f"Moved duplicate to backup: {val_file}")
    except Exception as e:
        print(f"Skipping {val_path}: {e}")
