AU-AIR Object Detection Assignment
1. Dataset Overview

Dataset: AU-AIR – aerial images of traffic scenes with object bounding box annotations.

Original size: 32,823 images, 8 categories (Van, Car, Trailer, Human, Truck, Bus, Bicycle, Motorbike).

Subset used: 15% (~4,923 images) to reduce computation time.

Classes distribution:

Category	Object Instances	Images Containing
Car	15,856	4,421
Van	1,532	1,156
Truck	1,389	1,176
Trailer	376	294
Human	762	700
Bus	110	101
Bicycle	168	141
Motorbike	51	49

Note: Object count > number of images because each image may contain multiple objects.

2. Data Cleaning & Preprocessing

Removed corrupted images using Python PIL (Image.open().verify()).

Removed duplicate frames using SHA256 hashing.

Subset of images chosen for faster processing.

Verified that bounding boxes lie within image dimensions.

3. Format Conversion

Converted AU-AIR annotations to YOLOv8 format:

Each image has a .txt label file with:

class_id x_center y_center width height


(all normalized to [0,1]).

Folder structure:

dataset/
 ├── images/
 │    ├── train/
 │    └── val/
 └── labels/
      ├── train/
      └── val/

4. Model Training

Model used: YOLOv8n (nano) pretrained on COCO.

Training parameters:

Epochs: 10

Image size: 640×640

Batch size: 16

Dataset: 15% AU-AIR subset

Training code:

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="dataset_config.yaml", 
    epochs=10, 
    imgsz=640, 
    batch=16, 
    name="auair_15pct_yolov8"
)


Weights saved at: runs/train/auair_15pct_yolov8/weights/best.pt

5. Inference & Visualization

Ran predictions on a few validation images.

Bounding boxes and class labels plotted using YOLOv8 API:

results = model("dataset/images/val/example.jpg")
results[0].plot()


Example visualizations included in outputs/ folder.

6. Next Steps / Improvements

Handling large images: resize or tile images to reduce memory/computation.

Class imbalance: sparse classes (Motorbike, Bus, Bicycle) → augment/oversample.

Longer training / larger models: YOLOv8m/l with more epochs.

Advanced augmentations: rotation, brightness, blur, etc. for robustness.

7. Notes

Accuracy is not the main focus; goal is to demonstrate full pipeline:
dataset exploration → cleaning → format conversion → training → inference → visualization.

AI tools used:

ChatGPT: guidance, code snippets, workflow planning.

Ultralytics YOLOv8: object detection training & inference.
