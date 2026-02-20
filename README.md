# **Tree Health Assessment using Deep Learning from Ground-Based RGB Images**
## Project Overview
This repository contains the resources and documentation for a deep learning model designed for multiclass semantic segmentation of ash tree images. The model aims to identify various features of ash trees in images, providing detailed segmentation useful for environmental studies, forestry management, and ecological research.

## Contents
code/: Directory containing the source code.
data/: Directory holding the dataset used for training and testing.
models/: Contains the trained model files.
requirements.txt: List of dependencies for replicating the project environment.
## Model Description
The model is built using [insert framework, e.g., TensorFlow, PyTorch] and is designed to segment ash tree images into multiple classes based on [list features, e.g., leaves, bark, surroundings]. 

## Model Weights
Due to file size, model weights are hosted on Zenodo:
Download here:
https://doi.org/xxxx
After download, place file in:
model/unet_resnet50_ash_tree.h5

## Installation
To set up the project environment:

## Clone the repository: git clone [URL]
Navigate to the project directory: cd [repository name]
Install dependencies: pip install -r requirements.txt
## Usage
To train the model: python code/train_model.py
To perform segmentation on new images: `python code/segment_image.py
[Path to image]`

## Dataset
The model was trained using a mixed dataset including images from public sources and local authority datasets which cannot be redistributed due to licensing restrictions. I am releasing RGB images and corresponding labelled masks which I captured personally, using a DSLR camera, from Priudhoe Rioverside Country Park, UK. The labelled were created using Trimble eCognition Developer (v9) with Object-Based Image Analysis (OBIA). 

## Results
Highlight the performance of the model. Include metrics like accuracy, IoU (Intersection over Union), etc. You can also add images showing segmentation results.

## Contributing
Instructions for how others can contribute to the project. This might include guidelines for submitting issues, pull requests, and contact information for queries.

## License
Specify the license under which your project is released, such as MIT, GPL, etc.

## Acknowledgements
If you use data, code, or trained model in a scientific publication, citations would be appreciated


## Contact Information
For collaboration, questions, or feedback, please email kf178@leicester.ac.uk or khunsafatima@gmail.com


