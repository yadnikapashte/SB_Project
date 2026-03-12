"""
config.py
---------
Central configuration file for the image classification pipeline.
All hyperparameters, paths, and settings are defined here.
Change model_name to automatically switch architectures.
"""

import torch
import os

# ─────────────────────────────────────────────
# Dataset Paths
# ─────────────────────────────────────────────
# Path to the raw dataset (organized by class folders)
RAW_DATA_PATH = "../conjunctiva_split" 

# Proportions for splitting the raw data
DATA_SPLIT_RATIO = (0.8, 0.1, 0.1)  # (train, val, test)

# Where the processed (split) data will be stored
DATA_ROOT = "../data/processed"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

# ─────────────────────────────────────────────
# Model Parameters
# ─────────────────────────────────────────────
# Options: "resnet50" | "resnet34" | "mobilenet"
MODEL_NAME  = "resnet50"

# ─────────────────────────────────────────────
# Training Parameters
# ─────────────────────────────────────────────
EPOCHS          = 20
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
NUM_WORKERS     = 4
EARLY_STOPPING_PATIENCE = 5   # stop if val loss doesn't improve

# ─────────────────────────────────────────────
# Image Parameters
# ─────────────────────────────────────────────
IMAGE_SIZE = 224   # height and width fed to the network

# Target images per class for training (None = unlimited). 
# If images < target, training set will be oversampled (physically augmented). 
# If images > target, it will be capped.
IMAGES_PER_CLASS = 800   
AUG_PER_IMAGE    = 12    # max augmented versions per original image

# ─────────────────────────────────────────────
# Output Paths
# ─────────────────────────────────────────────
OUTPUT_DIR          = "outputs"
MODEL_SAVE_PATH     = os.path.join(OUTPUT_DIR, "models")
METRICS_SAVE_PATH   = os.path.join(OUTPUT_DIR, "reports")
GRADCAM_SAVE_PATH   = os.path.join(OUTPUT_DIR, "gradcam")
CLASS_MAPPING_PATH  = os.path.join(OUTPUT_DIR, "class_mapping.json")

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Augmentation (applied to training set only)
# ─────────────────────────────────────────────
USE_AUGMENTATION = True

# --- Advanced Augmentation Parameters ---
APPLY_ROTATION      = True
AUG_ROTATION_DEGREES = 20      # Rotates image by +/- degrees

APPLY_HFLIP         = True
AUG_HFLIP_PROB       = 0.5     # Prob of horizontal flip

APPLY_VFLIP         = True
AUG_VFLIP_PROB       = 0.2     # Prob of vertical flip

APPLY_SHEAR         = True
AUG_SHEAR_DEGREES    = 10      # Shear intensity

APPLY_TRANSLATE     = True
AUG_TRANSLATE        = (0.1, 0.1)  # Max translation (width, height)

APPLY_COLOR_JITTER  = True
AUG_BRIGHTNESS       = 0.2     # Brightness jitter
AUG_CONTRAST         = 0.2     # Contrast jitter
AUG_SATURATION       = 0.2     # Saturation jitter
AUG_HUE              = 0.05    # Hue jitter

APPLY_RANDOM_ERASING = True
AUG_RANDOM_ERASING_PROB = 0.1  # Prob of random erasing a patch

# ─────────────────────────────────────────────
# Normalization Constants (ImageNet defaults)
# ─────────────────────────────────────────────
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# GradCAM Settings
# ─────────────────────────────────────────────
GRADCAM_NUM_IMAGES = 10   # how many test images to explain
GRADCAM_ALPHA      = 0.5 # overlay transparency

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = os.path.join(OUTPUT_DIR, "pipeline.log")
