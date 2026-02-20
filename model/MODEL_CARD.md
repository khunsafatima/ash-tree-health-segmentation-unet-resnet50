# Model Card: UNet ResNet50 for Tree Health Assessment

## Model Overview

This model performs multiclass semantic segmentation of ground-based RGB images of trees.

Architecture:
- UNet decoder
- ResNet50 encoder backbone
- Input size: 256 × 256 RGB
- Output: pixel-wise classification into 4 classes

Classes:
- 0: Background
- 1: Foliage
- 2: Wood
- 3: Ivy

The segmentation outputs are used to compute structural and physiological tree health indicators including:
- Defoliation percentage
- Tree tilt angle
- Crown symmetry
- Crown length to tree height ratio

---

## Intended Use

This model is intended for:

- Research in automated tree health assessment
- Forestry monitoring
- Environmental computer vision research
- Educational purposes

Not intended for:
- Safety-critical decisions
- Legal or regulatory enforcement
- Automated ecological policy decisions without expert validation

---

## Training Data

Total images: 453 RGB images of ash trees.

- 309 images collected at Prudhoe Riverside Country Park, UK
- 23 images collected from publicly available internet sources
- Remaining images provided by Norfolk County Council, UK

Due to third-party restrictions, the full dataset cannot be redistributed.

---

## Data Characteristics

- Ground-based perspective
- DSLR camera imagery
- Natural lighting conditions
- Primarily ash trees

---

## Evaluation Metrics

Evaluated on held-out test set.

Metrics reported:
- Precision: Foliage (0.81), Wood (0.62), Ivy (0.67)
- Recall: Foliage (0.81), Wood (0.48), Ivy (0.65)
- F1 Score: Foliage (0.81), Wood (0.54), Ivy (0.69)
- Overall pixel accuracy: 84.3%

---

## Limitations

- Trained primarily on ash trees
- Performance may degrade for other species
- Sensitive to extreme lighting conditions
- Ground-based images only
- Not validated for aerial or drone imagery

---

## Ethical Considerations

- Model may reflect bias in training dataset (location, species, lighting)
- Should not replace expert arboricultural assessment
- Intended as decision-support tool, not decision-maker

---

## Reproducibility

To reproduce results:

1. Resize input images to 256 × 256
2. Run segmentation model
3. Upscale predictions to original resolution
4. Compute tree health metrics

Dependencies:
- TensorFlow
- segmentation_models
- OpenCV
- NumPy
- Pillow

---

## Model Weights

Due to GitHub file size limitations, weights are hosted externally:

Place downloaded weights in:

model/unet_resnet50_ash_tree_segmentation.hdf5

---

## Model Version
- Model version: 1.0
- Date: February 2026
- Framework: TensorFlow / Keras
- 
