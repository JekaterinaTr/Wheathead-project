# Wheat Head Detection — Dense Small-Object Detection


This repository contains all code, configurations, and sample outputs for **Project 1: Wheat Head Detection**, a challenging computer vision task in agriculture. The goal is to **detect and count dense, thin wheat heads** in high-resolution field imagery.  

I compare **YOLO11 / YOLO12**, **RF-DETR**, and **Cascade R-CNN** models across accuracy, counting metrics, speed, and deployment practicality.

---

## Project Overview

- Task:  	Dense small-object detection (wheat heads)  
- Challenge:  	Hundreds of overlapping, elongated objects per image with low contrast  
- Goal: 	Compare latest Yolo-architecture detectors YOLO11 and YOLO12, transformer-based detector RF-DETR, 		and two-stage detector Cascade R-CNN for detection, counting, speed, and deployment efficiency  

- Key Metrics:  
  - Detection: mAP@0.5, mAP@[0.5:0.95], small-object AP  
  - Counting: MAE, RMSE, per-image error distribution  
  - Performance: FPS, peak GPU RAM, model size  

---

## Dataset

- Source: [Global Wheat Head Dataset 2021 (Kaggle)](https://www.kaggle.com/datasets/vbookshelf/global-wheat-head-dataset-2021)  
- Images: PNG  
- Annotations:YOLO TXT, one class: `wheat_head`  
- Split: Train / Val / Test  


---

## Repository Structure

wheat-head-detection/
├── README.md                     # Project description, instructions, metrics, etc.
├── data/                         # Sample data for testing only
│   └── sample/
│       ├── images/
│       │   ├── img1.png
│       │   └── img2.png
│       └── labels/
│           ├── img1.txt
│           └── img2.txt
├── src/   #Files for training
    ├── yolo_to_coco_change.py      # Forat change for labels from Yolo to COCO for RF-DETR model
│   ├── wheathead_dataset.yaml      #Yaml file for Yolo11 and Yolo12 training
    ├── train_rf_detr.py            # Training script for RF-DETR model
                         
│   └──train_cascade_rcnn.py
│                   # Optional: helper functions
├── configs/                      # Model hyperparameters and config files
│   ├── yolo11.yaml
│   ├── yolo12.yaml
    ├── rf_dert.yaml
│   └── cascade_rcnn.yaml
├── requirements/                 # Separate dependencies per environment
│   ├── yolo.txt
│   ├── rf_detr.txt
│   └── cascade_rcnn.txt
├── results/                      # Example predictions, images, or tables
└── .gitignore


---

# Data Folder

This repository includes a small subset of sample images in `data/sample/` for quick testing.

To train on the full Global Wheat Head Dataset 2021:


1. Go to the Kaggle dataset page: [Global Wheat Head Dataset 2021](https://www.kaggle.com/datasets/vbookshelf/global-wheat-head-dataset-2021)
2. Download the dataset manually (requires a Kaggle account)
3. Extract the images and annotations
4. Organize the folder structure as:

data/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
└── labels/
├── train/
├── val/
└── test/


---

## Environment Setup
Requires Python and Anaconda installed. Inside Anaconda create separate environments for Yolo models, for RF-DETR and for Cascade R-CNN

### 1. YOLO11 / YOLO12 (Ultralytics)

```bash
# Create environment
conda create -n yolo python=3.10 -y
conda activate yolo

# Install dependencies
pip install ultralytics==8.0.156  # YOLO11/12 compatible
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib pandas seaborn tqdm



### 2. RF-DETR

#RF-DETR requires label convertion from Yolo format to COCO format. To change the format: adapt label pathway in the script
python yolo_to_coco_change.py


conda create -n rf_detr python=3.8 -y
conda activate rf_detr

# PyTorch + CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install RF-DETR
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr
pip install -e .


### 3. Cascade R-CNN
conda create -n cascade_rcnn python=3.9 -y
con#da activate cascade_rcnn

# Install PyTorch + MMDetection dependencies
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install mmcv-full==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.13.0/index.html
pip install mmdet==3.2.0
pip install opencv-python

#Clone MMDetection on your PC
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.25.0
pip install -r requirements/build.txt
pip install -v -e .

#Cascade R-CNN uses COCO format labels, already created for RF-DETR model from the original Yolo format labels


---
##Training Scripts

Run inside the respective Conda environment.

##1. YOLO11
conda activate yolo

#Lance the commande using identical training hyperparameters and set up but change the base model eg: 

#Yolo11 nano:   	model=yolo11n.pt
#Yolo11 small: 	model=yolo11s.pt
#Yolo11 large: 	model=yolo11l.pt


# Example: YOLO11 small
yolo detect train model=yolo11s.pt data=data/wheathead_dataset.yaml 
    epochs=300 imgsz=640 batch=4 seed=0 mosaic=0.0 mixup=0.0 copy_paste=0.0 
    perspective=0.0005 shear=0.2 scale=0.2 flipud=0.0 fliplr=0.5 degrees=2.0 translate=0.1 hsv_h=0.0 hsv_s=0.2 hsv_v=0.2 lr0=0.0008 lrf=0.01   patience=25 warmup_epochs=3


##2. YOLO12
conda activate yolo
#Lance the commande using identical training hyperparameters and set up but change the base model eg: 

#Yolo12 nano:   	model=yolo12n.pt
#Yolo12 small: 	model=yolo12s.pt
#Yolo12 large: 	model=yolo12l.pt

# Example: YOLO11 small
yolo detect train model=yolo12s.pt data=data/wheathead_dataset.yaml epochs=300 imgsz=640 batch=4 seed=0 mosaic=0.0 mixup=0.0 copy_paste=0.0 perspective= 0.0005 shear= 0.2 scale=0.2 flipud=0.0 fliplr=0.5 degrees=2.0 translate=0.1 hsv_h=0.0 hsv_s=0.2 hsv_v=0.2 lr0= 0.0008 lrf=0.01  patience=25 warmup_epochs=3

##3. RF-DETR

conda activate rf_detr
#Model training: adapt the data and label pathways in the script
python train_rf_detr.py

##4. Cascade R-CNN
conda activate cascade_rcnn

#Run a single config file positioned inside MMDetection folder in /mmdetection/configs/cascade_rcnn

cd C:\mmdetection
python tools\train.py configs\cascade_rcnn\train_cascade_rcnn.py

---
##Model Configurations:

"""
Ultralytics YOLO requires training hyperparameters to be passed via CLI or Python API rather than fully configurable YAML files, unlike MMDetection or RF-DETR. Therefore, YAML configs are used for documentation, while execution is performed via script.”
For the RF-DETR and Cascade R-CNN the configs are already included int he execution python scripts (see earlier). These config files are thus rather used for hyperparameter documentation:
"""

configs/yolo11.yaml – YOLO11 training parameters
configs/yolo12.yaml – YOLO12 training parameters
configs/rf_detr.py –  RF-DETR training parameters
configs/cascade_rcnn.yaml – Cascade R-CNN training parameters


---
## Results

This folder contains a few sample outputs (images with bounding boxes) to demonstrate model performance.  

Full results, metrics, and plots are available on my portfolio website: [Your Portfolio Link Here].  

Key Takeaways:

--Best Accuracy: YOLO12-L — highest mAP, strong counting

--Best Practical Deployment: YOLO12-S — near-top mAP, low MAE, fast FPS

--Fastest / Edge Deployment: YOLO11-N — real-time, low VRAM

One-stage YOLO architectures outperform two-stage detectors and transformer-only detectors for dense agricultural object detection, achieving superior accuracy, counting performance, and inference efficiency.


