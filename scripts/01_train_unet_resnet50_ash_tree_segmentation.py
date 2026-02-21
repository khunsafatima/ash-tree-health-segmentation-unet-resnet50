#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# train_unet_resnet50_ash_tree_segmentation

This notebook trains a UNet segmentation model with a ResNet encoder for multiclass semantic segmentation of RGB ash tree images into:

- background
- foliage
- wood
- ivy
"""


# -----------------------------
# Configuration
# -----------------------------

from pathlib import Path

# Update these paths for your machine
DATA_DIR = Path("data")          # folder containing your .npy arrays
OUT_DIR = Path("outputs")        # training logs and saved models
OUT_DIR.mkdir(parents=True, exist_ok=True)

# These filenames are intentionally left generic.
# Provide your own .npy files.
X_TRAIN_NPY = "X_train.npy"
Y_TRAIN_NPY = "y_train_onehot.npy"
X_VAL_NPY   = "X_val.npy"
Y_VAL_NPY   = "y_val_onehot.npy"

# Model setup
N_CLASSES = 4
IMAGE_SIZE = 128       # expected height and width
N_CHANNELS = 3         # RGB

BACKBONE = "resnet50"  # alternatives: 'resnet34', 'resnet18', etc

# Training
BATCH_SIZE = 8
EPOCHS = 100
LR_INITIAL = 1e-4

# Learning rate schedule: if True, switches LR after epoch 50
USE_LR_SCHEDULE = True
LR_AFTER_EPOCH_50 = 1e-3

# Reproducibility
SEED = 42

# -----------------------------
# Import Libraries and Environment Checks
# -----------------------------

import os
import random
import numpy as np
import tensorflow as tf

# segmentation_models uses keras backend, ensure it sees tf.keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm

print("TensorFlow:", tf.__version__)
print("segmentation_models:", sm.__version__ if hasattr(sm, "__version__") else "unknown")

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Load training and validation data
# -----------------------------

def load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    arr = np.load(path)
    return arr

X_train = load_npy(DATA_DIR / X_TRAIN_NPY)
y_train = load_npy(DATA_DIR / Y_TRAIN_NPY)

X_val = load_npy(DATA_DIR / X_VAL_NPY)
y_val = load_npy(DATA_DIR / Y_VAL_NPY)

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("X_val:", X_val.shape, X_val.dtype)
print("y_val:", y_val.shape, y_val.dtype)

# Basic sanity checks
assert X_train.ndim == 4 and X_train.shape[-1] == N_CHANNELS, "X_train must be (N, H, W, 3)"
assert X_train.shape[1] == IMAGE_SIZE and X_train.shape[2] == IMAGE_SIZE, "Unexpected image size"
assert y_train.ndim == 4 and y_train.shape[-1] == N_CLASSES, "y_train must be one hot (N, H, W, C)"

# -----------------------------
# Preprocess input (encoder specific)
# -----------------------------

preprocess_input = sm.get_preprocessing(BACKBONE)

X_train_pp = preprocess_input(X_train)
X_val_pp = preprocess_input(X_val)

# -----------------------------
# Define model
# -----------------------------

model = sm.Unet(
    BACKBONE,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS),
    encoder_weights="imagenet",
    classes=N_CLASSES,
    activation="softmax",
)

# Optional: add segmentation friendly metrics
metrics = [
    "accuracy",
    sm.metrics.FScore(threshold=None),
    sm.metrics.IOUScore(threshold=None),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INITIAL),
    loss="categorical_crossentropy",
    metrics=metrics,
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------

from datetime import datetime

run_name = f"unet_{BACKBONE}_C{N_CLASSES}_S{IMAGE_SIZE}_BS{BATCH_SIZE}_E{EPOCHS}_LR{LR_INITIAL}"
run_dir = OUT_DIR / run_name
run_dir.mkdir(parents=True, exist_ok=True)

csv_logger = tf.keras.callbacks.CSVLogger(str(run_dir / f"{run_name}.log"), separator=",", append=False)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(run_dir / f"{run_name}_best.h5"),
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
)

def lr_scheduler(epoch: int, lr: float) -> float:
    if USE_LR_SCHEDULE and epoch >= 50:
        return float(LR_AFTER_EPOCH_50)
    return float(lr)

lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
)

callbacks = [csv_logger, checkpoint, lr_cb, early_stop]
print("Run directory:", run_dir.resolve())

# -----------------------------
# Train the Model
# -----------------------------

import time

start = time.time()
history = model.fit(
    X_train_pp,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_pp, y_val),
    callbacks=callbacks,
    verbose=1,
)
stop = time.time()

print(f"Training time (seconds): {stop - start:.1f}")

# -----------------------------
# Diagnostic plots
# -----------------------------

import matplotlib.pyplot as plt

def plot_history(history, out_dir: Path):
    hist = history.history

    # Loss
    plt.figure()
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=200)
    plt.show()

    # Accuracy (if present)
    if "accuracy" in hist:
        plt.figure()
        plt.plot(hist.get("accuracy", []), label="train")
        plt.plot(hist.get("val_accuracy", []), label="val")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy.png", dpi=200)
        plt.show()

plot_history(history, run_dir)

# -----------------------------
# Save final model (without optimiser state to reduce file size)
# -----------------------------

final_path = run_dir / f"{run_name}_final.h5"
model.save(final_path, include_optimizer=False)
print("Saved:", final_path.resolve())
