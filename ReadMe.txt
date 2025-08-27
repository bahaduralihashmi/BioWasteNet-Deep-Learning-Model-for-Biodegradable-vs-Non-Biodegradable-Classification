

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""Cloned from Git ultralytics""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Open your pycharm IDE and clicked at top left Hamburger. Look Git in it click on clone. You will see URL bar and then paste link in it and then clone. https://github.com/bebedovis/ultralytics
After this you will see it opened IDEs with ultralytics folder in your specific path. 



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""How to activate?""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


1. Activate virtaul enviroment it would be 3.11.
2. You will see at the bottom right add python interpreter. And install 3.11(ultralytics)
3. Ensure you have install. 
4. In terminal run this script to verify  check .\.ven\script\activate  python interpretor is activated.
5. You will see in terminal   (.venv) PS C:\Users\path\ultralytics>
6. Install libraries in terminal. pip install ultralytics onnx onnx-tf tensorflow seaborn
pip install torch torchaudio torchvision--index-url https://download/pytorch.org/whl/cu118

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""* Create a file yolov8n.yaml*"""""""""""""""""""""""""""""""""""""""

1. Create a new file where you clone folder see. \ultralytics\cfg\models\v8\
2. File name should be yolov8n.yaml
3. Paste the code in.

# Parameters
nc: 2            # number of classes (1 for human)
depth_multiple: 0.33  # model depth multiplier
width_multiple: 0.25  # layer channel multiplier
ch: 3            # input channels (3 for RGB)

# Backbone
backbone:
  # [from, number, module, args]
  - [-1, 1, Conv, [16, 3, 1]]       # 0-P1/2
  - [-1, 1, Conv, [32, 3, 2]]       # 1-P2/4
  - [-1, 1, C2f, [32, True]]        # 2
  - [-1, 1, Conv, [64, 3, 2]]       # 3-P3/8
  - [-1, 1, C2f, [64, True]]        # 4
  - [-1, 1, Conv, [128, 3, 2]]      # 5-P4/16
  - [-1, 1, C2f, [128, True]]       # 6
  - [-1, 1, Conv, [256, 3, 2]]      # 7-P5/32
  - [-1, 1, SPPF, [256, 5]]         # 8

# Head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]   # upsample
  - [[-1, 6], 1, Concat, [1]]                   # concat
  - [-1, 1, C2f, [128, True]]                  # 11
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 1, C2f, [64, True]]                   # 14
  - [[-1, 11], 1, Detect, [nc]]                # P3,P4,P5 detection
#end



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""* Create a datasets folder*"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

1. Paste a datasets folder in ultralytics folder.
2. Folder name should be datasets.
3. Replace accurate location of your data file. (test), (train), (data.yaml).
4. structure of datasets.

datasets/
│
├── train/
│   ├── images/
│   └── val/
│
├── test/  # optional
│   ├── images/
│   └── val/
│── val/
│   ├── images/
│   └── val/
├──data.yaml

5. Paste the code in data.yaml file.

test: C:\Users\path\Desktop\ultralytics\datasets\test\images		#replace path with your path
train: C:\Users\path\Desktop\ultralytics\datasets\train\images		#optaional replace path with your path
val: C:\Users\path\Desktop\ultralytics\datasets\val\images		#replace path with your path

nc: 2
names: ['biodegradable', 'non-biodegradable']
 


""""""""""""""""""""""* Create a new file train_grayscale.py *"""""""""""""""""""""""""""""""""
1. Open this new file and paste code.
2. Edit Line 8 use you Location ......\ultralytics\ultralytics\cfg\models\v8\yolov8n.yaml
3. Copy your datasets folder and paste in your clone folder. with Accurate structure. 
4. Edit Line 9 use you Location ......ultralytics\datasets\data.yaml

# File: train_grayscale.py
"""
Train YOLOv8n on grayscale images with 2 classes.
Make sure your dataset and config files are properly defined.
"""
from ultralytics import YOLO
# Configuration
model_arch = r"C:\Users\path\ultralytics\ultralytics\cfg\models\v8\yolov8n.yaml"  # paste your yolov8n.yaml file location.
data_yaml = r"C:\Users\path\ultralytics\datasets\data.yaml"		# paste your data.yaml file location. 
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

This will create a C:\Users\your_path\ultralytics\runs\detect\yolov8n_grayscale\weights\best.pt in 449 KB file.
Copy this file best.pt and paste in main folder where you had cloned. 


"""""""""""""""""""""""""""""""""""""""""""""""""""""""" Create a new file export_to_onnx.py """""""""""""""""""""""""""""""""""""""""""""""""""""""

1. Paste code given blow.
2. run this code.

from ultralytics import YOLO
import onnx

# Step 1: Load your trained PyTorch model
model = YOLO(r"C:\Users\....path....\runs\detect\Person_detect\weights\best.pt")

# Step 2: Export to ONNX with opset 12
onnx_path = model.export(format="onnx", opset=12, imgsz=64)

# Step 3: Force ONNX IR version = 9 for compatibility
onnx_model = onnx.load(onnx_path)
onnx_model.ir_version = 9
onnx.save(onnx_model, onnx_path)

print(f"✅ ONNX model exported and fixed at: {onnx_path}")

# end

::This will create a new file with name best.onnx 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Create a new file onnx_to_tf.py """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

1. Create new file in same directory name it onnx_to_tf
2. Paste the code given blow.
3. Ensure have you created onnx file in previuos step! 

from onnx_tf.backend import prepare
import onnx

# Load the ONNX model
onnx_model = onnx.load(r"C:\Users\...path..\runs\detect\Person_detect\weights\best.onnx")

# Convert ONNX to TensorFlow
tf_rep = prepare(onnx_model)

# Export TensorFlow SavedModel
tf_rep.export_graph("Bio_non_model") # change model name as you prefered name


# End code

This will creat a new folder in your path with name tf_model

""""""""""""""""""""""""""""""""""""""""""""""""""""""""" Create a new file tflite_to_INT8.py """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

1. Create a new file in same directory and name it tflite_to_INT8
2. Ensure have your created tf_model folder using previous step!
3. Paste the code given blow.
4. This will create and new file with name model_int8.tflite in 257 KB


import tensorflow as tf

# Convert SavedModel to quantized TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("Bio_non_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("Bio_non_model_quant.tflite", "wb") as f: # change name as your prefered 
    f.write(tflite_model)

print("Quantized TFLite model saved as Bio_non_model_quant.tflite")

# end code!

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Now you have got model_int8.tflite in 173 KB
1. Final step! Copy your model_int8.tflite file and paste in C volume for ease way to get location path.
2. Open Git bash run as administrator.
3. Run this command cd /c
4. Now you have entered in C volume.
5. Run this command.   xxd -i model_int8.tflite > best.cc
6. Now success fully created best.cc file in 1.1 MB
7. Congratulation!

 """"""""""NOW you can able to deploy in ESP32 model """""""""""""""""""""
If you want to increase size of images then where 64 replace by you preferred size.



















