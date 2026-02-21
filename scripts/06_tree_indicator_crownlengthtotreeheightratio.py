#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Crown length to tree height ratio Estimation from RGB Images

This notebook computes **crown length to tree height ratio** from RGB images and optional segmentation masks.

## Model
The trained model (UNet ResNet50 Model Weights for Tree Health RGB Segmentation (v1.0)) can be downloaded from Zenodo: https://zenodo.org/records/18709178.
After downloading, place the model in a folder 'model'

## Local data

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
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path("images")                 # original images 
MODEL_PATH = Path("model/unet_resnet50_ash_tree_segmentation.hdf5")      # trained model path
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_GLOB = "*.jpg"                             # or "*.png"
TARGET_SIZE = 256                              # model input size

# Class mapping (edit if different)
# 0 background, 1 foliage, 2 wood, 3 ivy
FOLIAGE_CLASS = 1
WOOD_CLASS = 2

# Output folders used by this notebook
RESIZE_DIR  = OUT_DIR / "resize"
PRED_DIR     = OUT_DIR / "prediction"
UPSCALE_DIR  = OUT_DIR / "upscaled"
CONTOUR_DIR  = OUT_DIR / "OutermostContour"
INTERSECT_PRED_CONT_DIR = OUT_DIR / "intersectpredictionCont"
TRUNK_TRUNCATE_DIR = OUT_DIR / "trunktruncated"
TRUNK_TRUNCATE_CONT_DIR = OUT_DIR / "trunktruncatedContour"

for d in [RESIZE_DIR, PRED_DIR, UPSCALE_DIR, CONTOUR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("DATA_DIR:", DATA_DIR.resolve())
print("OUT_DIR :", OUT_DIR.resolve())

# -----------------------------
# Resize images
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
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Make Predictions
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

# -----------------------------
# Upscale Predicted Masks back to Original Size
# -----------------------------
# Upscale predicted masks back to original image sizes (nearest neighbour)

# Load original sizes (written during resizing step).
sizes_path = OUT_DIR / "original_sizes.json"
if not sizes_path.exists():
    raise FileNotFoundError(
        f"{sizes_path} not found. Run the resize step first so original sizes are recorded."
    )

with open(sizes_path, "r", encoding="utf-8") as f:
    size_map = json.load(f)

pred_mask_paths = sorted(PRED_DIR.glob("*.png"))
if not pred_mask_paths:
    raise FileNotFoundError(f"No predicted masks found in {PRED_DIR}. Run prediction step first.")

n_ok = 0
for mask_path in pred_mask_paths:
    # Assumes mask filename matches original image stem
    # Example: IMG_001.jpg -> prediction/IMG_001.png
    stem = mask_path.stem

    # Find matching original image name by stem
    # Works for jpg/png/jpeg
    matches = [k for k in size_map.keys() if Path(k).stem == stem]
    if not matches:
        print("No original size entry for:", stem, "skipping")
        continue

    orig_name = matches[0]
    orig_h, orig_w = size_map[orig_name]

    mask_small = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_small is None:
        print("Unreadable mask:", mask_path.name, "skipping")
        continue

    mask_up = cv2.resize(mask_small, (int(orig_w), int(orig_h)), interpolation=cv2.INTER_NEAREST)

    out_path = UPSCALE_DIR / mask_path.name
    cv2.imwrite(str(out_path), mask_up)
    n_ok += 1

print("Upscaled masks written to:", UPSCALE_DIR)
print("Upscaled count:", n_ok)

# -----------------------------
# Outermost Contour
# -----------------------------
input_folder = UPSCALE_DIR
output_folder = CONTOUR_DIR

# Function to process the image and save the modified version
def process_image(image_path):
    # Load the segmented image
    im = cv2.imread(image_path)

    # Convert the segmented image to numpy array
    im = np.array(im)

    # Create a binary image based on three classes i.e., foliage, wood, and ivy
    single = im.copy()
    single[(single==2) | (single==3)]=1

    # Convert image to binary
    single_gray = cv2.cvtColor(single, cv2.COLOR_BGR2GRAY)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(single_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the outermost contour with the largest area
    outer_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            outer_contour = contour

    # Check if outer_contour is not None
    if outer_contour is not None:
        # Approximate the outermost contour with a polygon
        tolerance = 0.000001 * cv2.arcLength(outer_contour, True)
        approx = cv2.approxPolyDP(outer_contour, tolerance, True)

        # Create a new binary image with the polygonal approximation of the outermost contour
        outermost_polygon_img = np.zeros_like(single_gray)
        cv2.drawContours(outermost_polygon_img, [approx], -1, (255, 255, 255), -1)

        # Change value from 255 to 1
        outermost_polygon_img[outermost_polygon_img==255]=1

        # Get the relative path within the input folder
        relative_path = os.path.relpath(image_path, input_folder)

        # Construct the output path with corresponding subfolders
        output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
        os.makedirs(output_subfolder, exist_ok=True)
        output_path = os.path.join(output_subfolder, os.path.basename(image_path))

        # Save the modified image
        cv2.imwrite(output_path, outermost_polygon_img)

    else:
        print("No contour found for file", image_path)

# Recursively process all files in the folder and subfolders
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith((".JPG", ".jpg", ".png")):
            image_path = os.path.join(root, file)
            process_image(image_path)

# -----------------------------
# Intersect Prediction and Contour
# -----------------------------

folder_path = UPSCALE_DIR                  # upscaled prediction masks (folder)
contour_folder = CONTOUR_DIR               # contour images (folder)
output_folder = INTERSECT_PRED_CONT_DIR    # output folder

output_folder.mkdir(parents=True, exist_ok=True)

# List mask files from UPSCALE_DIR (use png only)
image_files = sorted(folder_path.glob("*.png"))
if not image_files:
    raise FileNotFoundError(f"No .png files found in {folder_path}")

n_ok = 0
n_missing_contour = 0

for fol_path in image_files:
    # contour file is expected to have the same name in CONTOUR_DIR
    cont_path = contour_folder / fol_path.name

    if not cont_path.exists():
        n_missing_contour += 1
        # If you want, print missing files:
        # print("Missing contour for:", fol_path.name)
        continue

    fol = Image.open(fol_path)
    cont = Image.open(cont_path)

    fol_array = np.array(fol)
    cont_array = np.array(cont)

    # Intersection: keep original fol values where contour > 0, else 0
    intersection_array = np.where(cont_array > 0, fol_array, 0).astype(fol_array.dtype)

    intersection_image = Image.fromarray(intersection_array)

    out_path = output_folder / fol_path.name
    intersection_image.save(out_path)
    n_ok += 1

print("Saved intersection outputs to:", output_folder.resolve())
print("Done:", n_ok)
print("Missing contour files:", n_missing_contour)

# -----------------------------
# Tree Height Estimation
# -----------------------------

image_dir = INTERSECT_PRED_CONT_DIR
csv_file = OUT_DIR / "treeheight.csv"

# Write header once
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "tree_height_pixels"])

# Process each PNG file
for filename in os.listdir(image_dir):
    if filename.lower().endswith(".png"):

        img_path = image_dir / filename
        img = Image.open(img_path)
        img_array = np.array(img)

        # Find rows containing any non-zero pixel
        rows_with_tree = np.any(img_array != 0, axis=1)

        if not np.any(rows_with_tree):
            TreeHeight = 0
        else:
            first_nonzero_row = np.argmax(rows_with_tree)
            last_nonzero_row = len(rows_with_tree) - np.argmax(rows_with_tree[::-1]) - 1
            TreeHeight = last_nonzero_row - first_nonzero_row

        # Append result
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename, TreeHeight])

print("Processing complete.")
print("Results saved to:", csv_file.resolve())

# -----------------------------
# Trunk Truncation
# -----------------------------

# Set the path to the folder containing upscaled predicted images
folder_path = INTERSECT_PRED_CONT_DIR

# Set the path to the folder where modified images will be saved
output_folder_path = TRUNK_TRUNCATE_DIR

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Function to process the image and save the modified version
def process_image(image_path, output_folder):
    # Load the predicted image
    predicted_image = cv2.imread(image_path)

    # Convert image to grayscale
    predicted_image_gray = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2GRAY)

    # Scan the image from the bottom
    desired_location = None
    for y in range(predicted_image_gray.shape[0] - 1, -1, -1):
        unique_classes = np.unique(predicted_image_gray[y])
        if 1 in unique_classes and 2 in unique_classes:
            desired_location = y
            break

    # Eliminate class 2 pixels from the bottom up to the desired location
    if desired_location is not None:
         # Get the indices where the pixel value is greater than 1
        rows, cols = np.where(predicted_image_gray > 1)

        # Filter out the rows starting from the desired location
        filtered_cols = cols[rows >= desired_location]

        # Eliminate class 2 pixels from the bottom up to the desired location
        if desired_location is not None:
            for col in filtered_cols:
                predicted_image_gray[desired_location:, col] = 0

    # Get the relative path within the input folder
    relative_path = os.path.relpath(image_path, folder_path)

    # Construct the output path with corresponding subfolders
    output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
    os.makedirs(output_subfolder, exist_ok=True)
    output_path = os.path.join(output_subfolder, os.path.basename(image_path))

    # Save the modified image
    cv2.imwrite(output_path, predicted_image_gray)

# Recursively process all files in the folder and subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".png"):
            image_path = os.path.join(root, file)
            process_image(image_path, output_folder_path)

# -----------------------------
# Crown Length (pixels) Estimation
# -----------------------------
image_dir = TRUNK_TRUNCATE_DIR
csv_file = OUT_DIR / "crownlength.csv"

# Write header once
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "crown_length_pixels"])

# Process each PNG file
for filename in os.listdir(image_dir):
    if filename.lower().endswith(".png"):

        img_path = image_dir / filename
        img = Image.open(img_path)
        img_array = np.array(img)

        # Find rows containing any non-zero pixel
        rows_with_crown = np.any(img_array != 0, axis=1)

        if not np.any(rows_with_crown):
            CrownLength = 0
        else:
            first_nonzero_row = np.argmax(rows_with_crown)
            last_nonzero_row = len(rows_with_crown) - np.argmax(rows_with_crown[::-1]) - 1
            CrownLength = last_nonzero_row - first_nonzero_row

        # Append result
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename, CrownLength])

print("Processing complete.")
print("Results saved to:", csv_file.resolve())

# -----------------------------
# Crown Length to Tree Height Ratio
# -----------------------------
# Paths
treeheight_path = OUT_DIR / "treeheight.csv"
crownlength_path = OUT_DIR / "crownlength.csv"
output_path = OUT_DIR / "crown_length_to_height_ratio.csv"

# Load CSV files
df_height = pd.read_csv(treeheight_path)
df_crown = pd.read_csv(crownlength_path)

# Ensure consistent column names
df_height.columns = ["image_name", "tree_height_pixels"]
df_crown.columns = ["image_name", "crown_length_pixels", *df_crown.columns[2:]]

# Merge on image_name
df = pd.merge(df_height, df_crown[["image_name", "crown_length_pixels"]],
              on="image_name",
              how="inner")

# Avoid division by zero
df["cl_th_ratio"] = df.apply(
    lambda row: row["crown_length_pixels"] / row["tree_height_pixels"]
    if row["tree_height_pixels"] > 0 else 0,
    axis=1
)

# Save result
df.to_csv(output_path, index=False)

print("Crown Length / Tree Height ratio saved to:")
print(output_path.resolve())
