# File: train_grayscale.py
"""
Train YOLOv8n on grayscale images with 2 classes.
Make sure your dataset and config files are properly defined.
"""
from ultralytics import YOLO
# Configuration
model_arch = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\ultralytics\cfg\models\v8\yolov8n_perfect.yaml"  # paste your yolov8n.yaml file location.
data_yaml = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\bio_non_trained_model\data_sets\datasets\set_dataset.yaml"		# paste your data.yaml file location.
imgsz = 64   	# image size should be 640, 128, or 94 for good result
epochs = 500 	# epochs is must be 200 to 500 range
batch_size = 16	#
ch = 1
experiment_name = "yolov8n_grayscale"		# replace name what you want.
device = "CPU"  # GPU (or 'cpu' if needed)
if __name__ == "__main__":
    model = YOLO(model_arch)
    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch_size,
        device=device,
        ch=ch,
        name=experiment_name,
        verbose=True
    )
    print(f"\n✅ Training complete. Model saved to: runs/detect/{experiment_name}/weights/best.pt\n")
#end
