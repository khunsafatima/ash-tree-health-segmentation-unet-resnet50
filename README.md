# **Tree Health Assessment using Deep Learning on Ground-Based RGB Images**

![License: MIT] (https://img.shields.io/badge/License-MIT-yellow.svg)
## Project Overview
This repository contains a deep learning framework for automated tree health assessment from ground-based RGB images. The framework consists of:
- A UNet model with ResNet50 backbone for multiclass semantic segmentation
- Segmentation of tree images into components (floiage, wood, ivy, background)
- Tree health indicators derived from segmentation masks:
  - Defoliation (%)
  - Tree Tilt (angle)
  - Crown Length to Tree Height Ratio
  - Crown Symmetry
The trained model operates on 256 x 256 pixels RGB images and predictions are upscaled back to original resolution for metric computation.

![Workflow](assets/1-TreeHealthWorkFlow.png)
---

## Dataset Availability
The model was trained using a mixed dataset including images from public sources and local authority datasets which cannot be redistributed due to licensing restrictions. I am releasing RGB images and corresponding labelled masks which I captured personally, using a DSLR camera, from Prudhoe Rioverside Country Park, UK. The labels were created using Trimble eCognition Developer (v9) with Object-Based Image Analysis (OBIA). 
Users can use the released dataset or could use their own RGB images for making predictions from the trained model. 

## Model Weights
Due to GitHub file size limitations, model weights are hosted separately on Zenodo.
Download Link: https://zenodo.org/records/18709178

Place the file inside:
model/unet_resnet50_ash_tree_segmentation.hdf5

## Citation
If you use data, code, or trained model in a scientific publication, citations would be appreciated:

@article{FATIMA2026129345,
title = {Integrated Deep Learning Framework for Automating Tree Health Assessment Using Ground-Based Images},
journal = {Urban Forestry & Urban Greening},
pages = {129345},
year = {2026},
issn = {1618-8667},
doi = {https://doi.org/10.1016/j.ufug.2026.129345},
url = {https://www.sciencedirect.com/science/article/pii/S1618866726000853},
author = {Khunsa Fatima and Ankush Prashar and Andrew Crowe and Paul Brown and Rachel Gaulton}

## Model Card
model/README.md

## Clone the repository: git clone [URL]
Navigate to the project directory: cd [ash-tree-health-segmentation-unet-resnet50]
Install dependencies: pip install -r requirements.txt
## Usage
To train the model: python code/train_model.py
To perform segmentation on new images: `python code/segment_image.py
[Path to image]`



## Results


## Contributing



## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.


## Contact Information
For collaboration, questions, or feedback, please email kf178@leicester.ac.uk 


