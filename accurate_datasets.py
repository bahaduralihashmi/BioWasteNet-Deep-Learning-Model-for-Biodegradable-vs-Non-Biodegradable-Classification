import cv2
import os
from glob import glob
from shutil import copytree, rmtree

# Paths to original dataset
original_train = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\datasets\train\images"
original_val = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\datasets\val\images"

# Paths to new grayscale dataset
gray_train = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\datasets\train_gray\images"
gray_val = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\datasets\val_gray\images"

def validate_labels(img_path, label_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"[Warning] Label file missing for {img_path}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"[Error] Incorrect label format in {label_path} line {i+1}")
            continue
        cls, x_c, y_c, bw, bh = parts
        x_c, y_c, bw, bh = map(float, [x_c, y_c, bw, bh])
        if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
            print(f"[Error] Label out of bounds in {label_path} line {i+1}")

def create_grayscale_folder(src_folder, dst_folder):
    if os.path.exists(dst_folder):
        rmtree(dst_folder)
    os.makedirs(dst_folder, exist_ok=True)

    # Copy labels folder
    labels_src = os.path.join(os.path.dirname(src_folder), "labels")
    labels_dst = os.path.join(os.path.dirname(dst_folder), "labels")
    copytree(labels_src, labels_dst)

    images = glob(os.path.join(src_folder, "*.*"))
    for img_path in images:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(dst_folder, img_name), img)

        # Validate label
        label_file = os.path.join(labels_dst, img_name.replace(os.path.splitext(img_name)[1], '.txt'))
        validate_labels(os.path.join(dst_folder, img_name), label_file)

# Process train and val
create_grayscale_folder(original_train, gray_train)
create_grayscale_folder(original_val, gray_val)

print("Grayscale dataset ready and labels validated:")
print(gray_train)
print(gray_val)
