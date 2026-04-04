"""Configuration module for Project Brain Tumor AI.

This module contains all global constants, file paths, and environment settings.
"""

import torch

# Model Configuration
MODEL_PATH: str = "models/checkpoints/best_model.pth"
INPUT_SHAPE: tuple[int, int, int, int] = (4, 128, 128, 64)  # (Channels, H, W, D)

# Device Configuration
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Visualization Settings
THEME_COLOR: str = "#0F9D58"  # GFG Brand Green
DARK_MODE_BG: str = "#020617"

# AI Inference Thresholds
TUMOR_VOLUME_THRESHOLD: int = 50000  # Voxel count threshold for HGG vs LGG
PROBABILITY_THRESHOLD: float = 0.5   # Threshold for binary segmentation mask
MIN_TUMOR_VOXELS: int = 100          # Minimum voxels for a valid tumor cluster (noise filter)

# Troubleshooting / Logging
LOGGING_LEVEL: str = "INFO"
