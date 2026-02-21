#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Make Predictions (Segmentation) on RGB Tree Images using Trained UNet (with ResNet50 backbone) model

This notebook can be used to get segmentation of RGB tree images.

## Model
The trained model (UNet ResNet50 Model Weights for Tree Health RGB Segmentation (v1.0)) can be downloaded from Zenodo: https://zenodo.org/records/18709178.
After downloading, place the model in a folder 'model'

## Input data

The model needs RGB tree images having size 256 x 256 pixels. The full dataset cannot be redistributed. Partial dataset is publically available at . Please download the dataset and set the paths in the configuration cell below. Users can use their own dataset as well

## Outputs

All outputs should be written into the output folder.
"""


# -----------------------------
# Import Libraries
# -----------------------------
from pathlib import Path
import os
from PIL import Image
import json
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path("images")                 # original images 
MODEL_PATH = Path("model/unet_resnet50_ash_tree_segmentation.hdf5")      # trained model path
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_GLOB = "*.jpg"                             # or "*.png"
TARGET_SIZE = 256                              # model input size

# Class mapping
# 0 background, 1 foliage, 2 wood, 3 ivy
FOLIAGE_CLASS = 1
WOOD_CLASS = 2
IVY_CLASS = 3
BACKGROUND_CLASS = 4

# Output folders used by this notebook
RESIZE_DIR  = OUT_DIR / "resize"
PRED_DIR     = OUT_DIR / "prediction"

for d in [RESIZE_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("DATA_DIR:", DATA_DIR.resolve())
print("OUT_DIR :", OUT_DIR.resolve())

# -----------------------------
# Resize All Image in a Folder
### The model (UNet with ResNet50 backbone) accepts input RGB images with size 256 x 256 pixels
# -----------------------------
# Use configuration variables
input_folder = DATA_DIR
output_folder = RESIZE_DIR
new_size = (TARGET_SIZE, TARGET_SIZE)

# Path for storing original sizes
sizes_path = OUT_DIR / "original_sizes.json"

# Collect original sizes: key = original filename, value = [H, W]
size_map = {}

# Iterate over all folders and files in the input folder
for root, dirs, files in os.walk(input_folder):
    root = Path(root)

    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            in_path = root / file

            with Image.open(in_path) as im:
                # Record original size (PIL gives (W, H))
                orig_w, orig_h = im.size
                size_map[in_path.name] = [int(orig_h), int(orig_w)]

                # Resize to model input size
                im_resized = im.resize(new_size)

                # Preserve subfolder structure
                rel_dir = in_path.parent.relative_to(input_folder)
                out_dir = output_folder / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / (in_path.stem + ".png")
                im_resized.save(out_path)

# Save original size map for upscaling step
with open(sizes_path, "w", encoding="utf-8") as f:
    json.dump(size_map, f, indent=2)

print("Saved resized images to:", output_folder.resolve())
print("Saved original sizes to:", sizes_path.resolve())
print("Example entry:", next(iter(size_map.items())) if size_map else "No images found")

# -----------------------------
# Load Trained Model
# -----------------------------
#Load the trained moel
model = tf.keras.models.load_model('model/unet_resnet50_ash_tree_segmentation.hdf5')

# -----------------------------
# Make Predictions for all Images in a Folder and Sunfolders
# -----------------------------
input_folder = RESIZE_DIR
output_folder = PRED_DIR

# Iterate over all folders and files in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        # Check if the file is an image
        if file.endswith(".JPG") or file.endswith(".png"):
            # Load the image
            im = cv2.imread(os.path.join(root, file))
            
            # make the prediction
            test_img = np.expand_dims(im, 0)
            pred = model.predict(test_img)
            predict = np.argmax(pred, axis=3)[0,:,:]
            
            predict_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file[:-4]+'.png')
            os.makedirs(os.path.dirname(predict_path), exist_ok=True)
            cv2.imwrite(predict_path, predict)