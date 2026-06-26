# src/config.py

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_NAME = os.path.join(BASE_DIR, "models", "yolo11n.pt")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "runtime")

# Optimization Settings (Target 40+ FPS @ 4K)
PATCH_SIZE = 1280
MIN_OVERLAP = 128  # Balanced for speed & small objects
CONF_THRES = 0.2

# TensorRT Export Settings
TRT_BATCH_SIZE = 8  # paper-aligned 1280px engine max batch / runtime patch batch
TRT_DYNAMIC_SHAPES = True  # flexible default; use static export for fixed paper-speed runs

# Visualization
VIS_THICKNESS = 1
VIS_FONT_SCALE = 0.4
