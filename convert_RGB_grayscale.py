import os
from PIL import Image
import shutil

# Original dataset paths
dataset_paths = {
    "train": r"C:\Users\bahad\OneDrive\Desktop\ultralytics\person_datasets\train", # dataset path should match
    "val": r"C:\Users\bahad\OneDrive\Desktop\ultralytics\person_datasets\val"      # dataset path should match
}

# Destination for grayscale dataset
gray_dataset_base = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\person_datasets"

for split, path in dataset_paths.items():
    # Create grayscale folders for images and labels
    gray_images_folder = os.path.join(gray_dataset_base, split, "images")
    gray_labels_folder = os.path.join(gray_dataset_base, split, "labels")
    os.makedirs(gray_images_folder, exist_ok=True)
    os.makedirs(gray_labels_folder, exist_ok=True)

    # Copy and convert images to grayscale
    for img_file in os.listdir(os.path.join(path, "images")):
        img_path = os.path.join(path, "images", img_file)
        try:
            img = Image.open(img_path).convert("L")  # Grayscale
            img.save(os.path.join(gray_images_folder, img_file))
            print(f"Converted: {img_path}")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    # Copy labels as-is
    for label_file in os.listdir(os.path.join(path, "labels")):
        src_label = os.path.join(path, "labels", label_file)
        dst_label = os.path.join(gray_labels_folder, label_file)
        shutil.copy2(src_label, dst_label)
