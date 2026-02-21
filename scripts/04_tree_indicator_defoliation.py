#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Tree health indicators from RGB images

This notebook computes **defoliation percentage** from RGB images and optional segmentation masks.

## Model
The trained model (UNet ResNet50 Model Weights for Tree Health RGB Segmentation (v1.0)) can be downloaded from Zenodo: https://zenodo.org/records/18709178.
After downloading, place the model in a folder 'model'

## Local data

The model needs RGB tree images having size 256 x 256 pixels. The full dataset cannot be redistributed. Partial dataset is publically available at . Please download the dataset and set the paths in the configuration cell below. Users can use their own dataset as well

## Outputs

All outputs should be written into the output folder.
"""


# -----------------------------
# Cell 1
# -----------------------------
from pathlib import Path
import cv2
import numpy as np
import os
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -------------------------
# Configuration
# -------------------------

DATA_DIR = Path("data")            # folder containing images and optional masks
OUT_DIR  = Path("outputs")         # where results will be saved
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_GLOB = "*.jpg"               # change to *.png if needed
MASK_DIR = None                    # set to Path("masks") if you have predicted masks

# -----------------------------
# Trunk Truncation
# -----------------------------

# Set the path to the folder containing predicted images
folder_path = ""

# Set the path to the folder where modified images will be saved
output_folder_path = "trunktruncated/"

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
        rows, cols = np.where(predicted_image_gray > 1)

        # Filter out the rows starting from the desired location
        cols = cols[rows >= desired_location]

        # Eliminate class 2 pixels from the bottom up to the desired location
        predicted_image_gray[desired_location:, cols] = 0
        
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
# Create Outermost Contours
# -----------------------------

input_folder = 'trunktruncated/'
output_folder = 'outermostContour/'

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
        if file.endswith((".jpg", ".png")):
            image_path = os.path.join(root, file)
            process_image(image_path)

# -----------------------------
# Intersected Truncated Image
# -----------------------------

folder_path = "trunktruncated"  # Specify the folder path

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

output_folder = "intersecttruncatedCont"  # Specify the output folder

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_file in image_files:
    # Open the images
    fol_path = os.path.join(folder_path, image_file)
    cont_path = os.path.join("outermostContour", image_file)
    fol = Image.open(fol_path)
    cont = Image.open(cont_path)

    # Convert the images to numpy arrays
    fol_array = np.array(fol)
    cont_array = np.array(cont)

    # Perform intersection while retaining original values
    intersection_array = np.where(cont_array > 0, fol_array, 0)

    # Create a new PIL Image from the intersection array
    intersection_image = Image.fromarray(intersection_array)

    # Save the intersection image in the output folder with the same name as the input image
    output_path = os.path.join(output_folder, image_file)
    intersection_image.save(output_path)

# -----------------------------
# Estimated Foliage
# -----------------------------

# Set the path to the folder containing predicted images
folder_path = "intersecttruncatedcont"

# Set the path to the folder where foliage images will be saved
output_folder_path = "estimatedfoliage"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over all folders and files in the input folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file is an image
        if file.endswith((".jpg", ".JPG", ".png")):
            # Load the predicted image
            pred_path = os.path.join(root, file)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
            # Create an image based on the foliage class
            foliage = pred.copy()
        
            # Replace values 2 and 3 with 0
            foliage[(foliage == 2)|(foliage == 3)] = 0
        
            # Get the relative path within the input folder
            relative_path = os.path.relpath(pred_path, folder_path)
        
            # Construct the output path
            output_path = os.path.join(output_folder_path, relative_path)
        
            # Create the output folder if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
            # Save the foliage image
            cv2.imwrite(output_path, foliage)

# -----------------------------
# Wood Ivy Image
# -----------------------------

# Set the path to the folder containing predicted images
folder_path = "intersecttruncatedcont"

# Set the path to the folder where woodIvy images will be saved
output_folder_path = "woodivy"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over all folders and files in the input folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file is an image
        if file.endswith((".jpg", ".JPG", ".png")):
            # Load the predicted image
            pred_path = os.path.join(root, file)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
            # Create an image based on the woodIvy class
            woodIvy = pred.copy()
            
            # Replace values 1 with 0
            woodIvy[(woodIvy == 1)] = 0
        
            # Replace values 2 and 3 with 1
            woodIvy[(woodIvy == 2) | (woodIvy == 3)] = 1
        
            # Get the relative path within the input folder
            relative_path = os.path.relpath(pred_path, folder_path)
        
            # Construct the output path
            output_path = os.path.join(output_folder_path, relative_path)
        
            # Create the output folder if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
            # Save the woodIvy image
            cv2.imwrite(output_path, woodIvy)

# -----------------------------
# Wood Image
# -----------------------------

# Set the path to the folder containing predicted images
folder_path = "intersecttruncatedcont"

# Set the path to the folder where woodIvy images will be saved
output_folder_path = "woodivy"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate over all folders and files in the input folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # Check if the file is an image
        if file.endswith((".jpg", ".JPG", ".png")):
            # Load the predicted image
            pred_path = os.path.join(root, file)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
            # Create an image based on the woodIvy class
            wood = pred.copy()
            
            # Replace values 1 with 0
            wood[(wood == 1) | (wood == 3)] = 0
        
            # Replace values 2 with 1
            wood[(wood == 2)] = 1
        
            # Get the relative path within the input folder
            relative_path = os.path.relpath(pred_path, folder_path)
        
            # Construct the output path
            output_path = os.path.join(output_folder_path, relative_path)
        
            # Create the output folder if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
            # Save the woodIvy image
            cv2.imwrite(output_path, wood)

# -----------------------------
# Expected Foliage (Alpha Shape for All Contours)
# -----------------------------

# Make sure to import Alpha_Shaper correctly

image_directory = "outermostContour"

for root, dirs, files in os.walk(image_directory):
    for filename in files:
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(root, filename)
            im = Image.open(image_path)
            im_arr = np.array(im)

            points = np.column_stack(np.where(im_arr > 0))

            # Ensure you have an Alpha_Shaper class with a get_shape method
            shaper = Alpha_Shaper(points)
            alpha_shape = shaper.get_shape(alpha=2.2)

            alpha_image = Image.new('L', im.size)
            alpha_arr = np.zeros(im.size[::-1], dtype=np.uint8)  # Note the size reversal

            for y in range(alpha_arr.shape[0]):
                for x in range(alpha_arr.shape[1]):
                    if alpha_shape.contains(Point(x, y)):  # Check coordinate order in your shape
                        alpha_arr[y, x] = 255  # Use 255 for white in 'L' mode

            alpha_image.putdata(alpha_arr.ravel())

            base_filename = os.path.splitext(filename)[0]
            alpha_mirror = alpha_image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)

            relative_path = os.path.relpath(root, image_directory)
            save_directory_mirror = os.path.join("alpha2pt2/", relative_path)
            os.makedirs(save_directory_mirror, exist_ok=True)

            save_path_mirror = os.path.join(save_directory_mirror, base_filename + ".png")
            alpha_mirror.save(save_path_mirror)

# -----------------------------
# To estimate expected foliage, Exclude wood and ivy from the AlphaShape
# -----------------------------

# Directories
dir_alpha2pt2 = 'alpha2pt2'
dir_wood = 'woodivy'
dir_result = 'expectedfoliage'

# Create the result directory if it does not exist
if not os.path.exists(dir_result):
    os.makedirs(dir_result)

# Iterate through the images in the first directory
for filename in os.listdir(dir_alpha2pt2):
    path_alpha2pt2 = os.path.join(dir_alpha2pt2, filename)
    path_wood = os.path.join(dir_wood, filename)

    # Check if the corresponding file exists in the other directory
    if os.path.exists(path_wood):
        # Read the images
        img_alpha2pt2 = cv2.imread(path_alpha2pt2, 0)  # 0 to read in grayscale
        img_wood = cv2.imread(path_wood, 0)

        # Apply XOR operation
        result = cv2.bitwise_xor(img_alpha2pt2, img_wood)

        # Save the result
        cv2.imwrite(os.path.join(dir_result, filename), result)
    else:
        print(f"Corresponding file for {filename} not found in {dir_wood}")

# -----------------------------
# Defoliation Estimation for all Images in the Folder and subfolders
# -----------------------------

# Define the directory containing the foliage images
image_directory = "estimatedfoliage/"

# Create a CSV file for saving the estimated defoliation values
csv_file = "defoliation.csv"

# Open the CSV file in write mode
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(["ImageID", "Defoliation (%)"])

    # Loop through each directory and subdirectory in the image directory
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            # Skip non-image files
            if not filename.endswith((".png", ".jpg", ".jpeg")):
                continue

            # Load the predicted image
            predicted_image = cv2.imread(os.path.join(root, filename), 0)  # Load as grayscale (single channel)

            # Get the corresponding alphaMinuswoodIvy image path
            relative_path = os.path.relpath(root, image_directory)
            alphashapemirror_path = os.path.join("alpha2pt2", relative_path, filename)

            # Check if the alphashapemirror image exists
            if not os.path.exists(alphashapemirror_path):
                print("The alphashapemirror image for", filename, "does not exist.")
                continue

            # Load the alphashapemirror image
            alphashapemirror = cv2.imread(alphashapemirror_path, 0)  # Load as grayscale (single channel)

            # Initialize counters for each image
            predicted_count = 0
            alphashapemirror_count = 0

            # Check if the shape of the images is the same
            if predicted_image.shape == alphashapemirror.shape:
                # Iterate over each pixel in the predicted image
                for y in range(predicted_image.shape[0]):
                    for x in range(predicted_image.shape[1]):
                        # Check if the pixel in the predicted image is class 1
                        if predicted_image[y, x] == 1:
                            predicted_count += 1

                        # Check if the corresponding pixel in the alphashapemirror image is non-zero
                        if alphashapemirror[y, x] > 0:
                            alphashapemirror_count += 1

                # Calculate defoliation using the formula
                defoliation = ((alphashapemirror_count - predicted_count) / alphashapemirror_count) * 100
                defoliation = max(0, defoliation)  # Set defoliation to zero if it is less than zero

                # Write the image name and defoliation value to the CSV file
                writer.writerow([filename, "{:.2f}%".format(defoliation)])
            else:
                print("The shapes of the predicted image and alphamirror image for", filename, "do not match.")