#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Tree Tilt Estimation from RGB images

This notebook computes **tree tilt** from RGB images and optional segmentation masks.

## Model
The trained model (UNet ResNet50 Model Weights for Tree Health RGB Segmentation (v1.0)) can be downloaded from Zenodo: https://zenodo.org/records/18709178.
After downloading, place the model in a folder 'model'

## Local data

The model need RGB tree images having size 256 x 256 pixels. The full dataset cannot be redistributed. Partial dataset is publically available at . Please download the dataset and set the paths in the configuration cell below. Users can use their own dataset as well

## Outputs

All outputs should be written into the output folder.
"""


# -----------------------------
# Import Libraries
# -----------------------------
from pathlib import Path
import os
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
import csv

# -------------------------
# Configuration
# -------------------------
DATA_DIR = Path("data")            # folder containing images and optional masks
OUT_DIR  = Path("outputs")         # where results will be saved
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_GLOB = "*.jpg"               # change to *.png if needed
MASK_DIR = None                    # set to Path("masks") if you have predicted masks

# -----------------------------
# Resize all Images in a folder
# ----------------------------

input_folder = 'images' # Provide folder containing images
output_folder = 'resize' # Provide output folder name 
new_size = (256, 256)  # define the new size of the images

# Iterate over all folders and files in the input folder
for root, dirs, files in os.walk(input_folder):
    for file in files:
        # Check if the file is an image
        if file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".png"):
            # Load the image
            im = Image.open(os.path.join(root, file))
            im = im.resize(new_size)
            
            new_size_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file[:-4]+'.png')
            os.makedirs(os.path.dirname(new_size_path), exist_ok=True)
            im.save(new_size_path)

# -----------------------------
# Load Trained Model
# -----------------------------
#Load the trained moel
model = tf.keras.models.load_model('model/unet_resnet50_ash_tree_segmentation.hdf5')

# -----------------------------
# Make Predictions
# -----------------------------

input_folder = 'resize' # Provide folder having resized images
output_folder = 'prediction'

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
# Outermost Contour
# -----------------------------

input_folder = 'prediction'
output_folder = 'outerContour'

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
# Intersect Prediction and Outermost Contour
# -----------------------------

folder_path = "prediction"  # Specify the folder path

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

output_folder = "IntersectPredictCont"  # Specify the output folder

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for image_file in image_files:
    # Open the images
    fol_path = os.path.join(folder_path, image_file)
    cont_path = os.path.join("outerContour", image_file)
    fol = Image.open(fol_path)
    cont = Image.open(cont_path)

    # Convert the contour image to a numpy array
    cont_array = np.array(cont)

    # Perform intersection with the contour image
    intersection_array = np.minimum(fol, cont_array)

    # Create a new PIL Image from the intersection array
    intersection_image = Image.fromarray(intersection_array)

    # Save the intersection image in the output folder with the same name as the input image
    output_path = os.path.join(output_folder, image_file)
    intersection_image.save(output_path)

# -----------------------------
# Tree Orientation
# -----------------------------

# Source folder containing input images
source_folder = "IntersectPredictCont"

# Output folder to save processed images
output_folder = "ProcessedImages"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Create a CSV file to store angle_deg values and image names
csv_file = "angle_deg_values.csv"

# Create or open the CSV file in write mode
with open(csv_file, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image Name', 'Angle_deg'])

    # Iterate through all image files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):  # Assuming all image files are PNG
            # Open the image
            img = Image.open(os.path.join(source_folder, filename))

            # Convert the image to a numpy array
            img_array = np.array(img)

            # Find the first row from the bottom with non-zero pixels
            for row in range(img_array.shape[0] - 1, -1, -1):
                if np.any(img_array[row] > 0):
                    # Find the central non-zero pixel in the row
                    central_pixel_index = np.argmax(img_array[row] > 0)
                    central_pixel = (row, central_pixel_index)

                    # Create a new image with the same size and the desired colormap
                    cmap = ListedColormap(["whitesmoke", "limegreen", "peru", "darkgreen"])
        

                    # Calculate the angle with the maximum number of '2' pixels
                    angle_with_most_2s = 0
                    max_2_count = 0

                    for angle in range(0, 360):
                        # Calculate the direction vector
                        direction_vector = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])

                        # Project the line in the given direction
                        line_projection = ndimage.map_coordinates(img_array, [central_pixel[0] + direction_vector[1] * np.arange(img_array.shape[0]),
                                                                  central_pixel[1] + direction_vector[0] * np.arange(img_array.shape[1])])

                        # Count the number of '2' pixels in the projection
                        count_2s = np.sum(line_projection == 2)

                        # Update if a new maximum is found
                        if count_2s > max_2_count:
                            max_2_count = count_2s
                            angle_with_most_2s = angle

                            # Ensure the angle is always between 0 and 90 degrees
                            angle_deg = abs(angle_with_most_2s)
                
                            # Check if the angle is greater than 90 degrees
                            if angle_deg > 90:
                                angle_deg = abs(180 - angle_deg)

                    # Create a new figure and plot the image and other elements
                    plt.figure()
                    plt.imshow(img_array, cmap=cmap, vmin=0, vmax=4)

                    # Add the red dot on the new image
                    plt.plot(central_pixel[1], central_pixel[0], 'ro', markersize=10, label="Tree Base")

                    # Draw a horizontal dotted blue line starting from the red dot
                    line_x = np.arange(central_pixel[1], img_array.shape[1])
                    line_y = np.full_like(line_x, central_pixel[0])
                    plt.plot(line_x, line_y, 'b:', linewidth=2, label="Horizontal Line")
                    
                    # Mark the angle with a red line on the image
                    direction_vector = np.array([np.cos(np.deg2rad(angle_with_most_2s)), np.sin(np.deg2rad(angle_with_most_2s))])
                    end_point = (int(central_pixel[1] + 100 * direction_vector[0]), int(central_pixel[0] + 100 * direction_vector[1]))
                    plt.plot([central_pixel[1], end_point[0]], [central_pixel[0], end_point[1]], 'r-', linewidth=2, label="Angle with Max '2'")
                    plt.text(2,-5, f"Tree Orientation: {angle_deg} degrees")
                    # Show the image with the added elements
                    plt.axis("off")

                    # Save the output image with the original name in the new folder
                    output_filename = os.path.join(output_folder, filename)
                    plt.savefig(output_filename, bbox_inches='tight', pad_inches=1, dpi=300)
                    
                    # Close the current figure
                    plt.close()

                    # Write the angle_deg value and image name to the CSV file
                    csv_writer.writerow([filename, angle_deg])

                    break

# Print a message when processing is complete
print("Processing complete. Images and angle_deg values saved.")
