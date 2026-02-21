#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Evaluation of Trained Segmentation Model (UNet with ResNet50 Backbone)

**Classes**:
- 0: background
- 1: foliage
- 2: wood
- 3: ivy

## Data Required
- You would need to provide test dataset:
  * test
    * images
    * masks
"""


# -----------------------------
# Import Librarues
# -----------------------------

import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import rasterio
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

# Paths (edit for your local machine) 
DATA_DIR = Path("test")     
OUT_DIR  = Path("outputs_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make segmentation_models use tf.keras
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
import segmentation_models as sm

print("TensorFlow:", tf.__version__)
print("segmentation_models:", getattr(sm, "__version__", "unknown"))

# Model / dataset settings
BACKBONE = "resnet50"
IMAGE_SIZE = 256
N_CLASSES = 4

# Visualisation
N_EXAMPLES_TO_PLOT = 6
RANDOM_SEED = 42

# -----------------------------
# Convert Dataset to Numpy Arrays
# -----------------------------

DATA_DIR = Path("test")
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR  = DATA_DIR / "masks"

IMAGE_SIZE = 256  # model input size

image_paths = sorted(list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpg")))

X_test, y_true = [], []

for img_path in image_paths:
    # Try matching mask name: IMG_001.png -> IMG_001.tif
    mask_path = MASK_DIR / f"{img_path.stem}.tif"
    if not mask_path.exists():
        alt = MASK_DIR / (img_path.name.rsplit(".", 1)[0] + ".tif")
        mask_path = alt if alt.exists() else mask_path

    if not mask_path.exists():
        print("Mask missing for:", img_path.name)
        continue

    # --- image ---
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print("Unreadable image:", img_path)
        continue

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    # --- mask (GeoTIFF) ---
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    mask = mask.astype(np.int64)
    mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_NEAREST)

    X_test.append(img)
    y_true.append(mask)

# Convert to arrays
X_test = np.stack(X_test, axis=0) if X_test else np.empty(
    (0, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32
)

y_true = np.stack(y_true, axis=0) if y_true else np.empty(
    (0, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64
)

print("Loaded X_test:", X_test.shape)
print("Loaded y_true:", y_true.shape)
print("Unique labels in masks:", np.unique(y_true) if y_true.size else "None")

# Save arrays
np.save(DATA_DIR / "X_test.npy", X_test)
np.save(DATA_DIR / "y_true.npy", y_true)

print("Saved:")
print(DATA_DIR / "X_test.npy")
print(DATA_DIR / "y_true.npy")

# -----------------------------
# Load Trained Model
# -----------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}\n"
        "Tip: download weights (from Zenodo) and place them under ./model/"
    )

# If your model uses custom objects, add them here:
CUSTOM_OBJECTS = {}

# Model weights path (download separately hosted on Zenodo: https://zenodo.org/records/18709178)
MODEL_PATH = Path("/gws/nopw/j04/nceo_digital_twin/TH/unet_resnet50_ash_tree_segmentation.hdf5")

model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects=CUSTOM_OBJECTS)
model.summary()

# -----------------------------
# Predict
# -----------------------------

preprocess_input = sm.get_preprocessing(BACKBONE)
X_pp = preprocess_input(X_test)

# Predict probabilities: (N,H,W,C)
probs = model.predict(X_pp, batch_size=8, verbose=1)

# Convert to predicted class labels: (N,H,W)
y_pred = np.argmax(probs, axis=-1).astype(np.int32)
print("y_pred:", y_pred.shape, y_pred.dtype)

# -----------------------------
# Evaluation Metrics (per class + overall)
# -----------------------------

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    metrics = []
    # Flatten for pixel-wise evaluation
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    overall_acc = float((yt == yp).mean())

    for c in range(n_classes):
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)
        iou       = tp / (tp + fp + fn + 1e-12)

        metrics.append({
            "class": c,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
        })

    return overall_acc, pd.DataFrame(metrics)

overall_acc, df = per_class_metrics(y_true, y_pred, N_CLASSES)

display(df)
print("Overall pixel accuracy:", overall_acc)

# -----------------------------
# save results
# -----------------------------

results = {
    "backbone": BACKBONE,
    "image_size": IMAGE_SIZE,
    "n_classes": N_CLASSES,
    "overall_pixel_accuracy": overall_acc,
}

df.to_csv(OUT_DIR / "per_class_metrics.csv", index=False)
pd.DataFrame([results]).to_csv(OUT_DIR / "summary_metrics.csv", index=False)

print("Saved:")
print("-", (OUT_DIR / "per_class_metrics.csv").resolve())
print("-", (OUT_DIR / "summary_metrics.csv").resolve())
