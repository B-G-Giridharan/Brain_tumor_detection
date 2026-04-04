"""Preprocessing pipeline using MONAI transforms.

This module converts validated NumPy MRI data into model-ready 
PyTorch tensors with consistent normalization and resizing.
"""

import logging
from typing import Any
import torch
import numpy as np
from monai.transforms import Compose, EnsureType, ScaleIntensity, Resize

logger = logging.getLogger(__name__)

def preprocess_mri(data: np.ndarray) -> torch.Tensor:
    """Preprocesses a stacked MRI multi-modal array into a normalized tensor.

    Args:
        data (np.ndarray): Input MRI array of shape (4, H, W, D).

    Returns:
        torch.Tensor: Preprocessed tensor (1, 4, 128, 128, 64) as float32.

    Raises:
        ValueError: If input is None, not a NumPy array, or has incorrect shape.
    """
    logger.info("Initializing MRI preprocessing pipeline...")

    # 1. Validation
    if data is None:
        raise ValueError("Input MRI data cannot be None.")
    
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Input must be a NumPy array, got {type(data)}.")
    
    if data.ndim != 4 or data.shape[0] != 4:
        raise ValueError(f"Input shape mismatch. Expected (4, H, W, D), got {data.shape}.")

    logger.info(f"Input Shape: {data.shape} | Dtype: {data.dtype}")

    try:
        # 2. MONAI Transform Sequence
        # Note: data is already channel-first (4, H, W, D) from the loader
        transforms = Compose([
            # Convert to PyTorch tensor and ensure float32
            EnsureType(data_type="tensor", dtype=torch.float32),
            # Normalize intensity range to [0, 1]
            ScaleIntensity(minv=0.0, maxv=1.0),
            # Resize to target inference resolution
            Resize(spatial_size=(128, 128, 64), mode="trilinear", align_corners=True)
        ])

        # Execute transformation
        processed_tensor = transforms(data)

        # 3. Batch Dimension (C, H, W, D) -> (1, C, H, W, D)
        final_tensor = processed_tensor.unsqueeze(0)

        logger.info(f"Preprocessed Output Shape: {final_tensor.shape} | Device: {final_tensor.device}")
        return final_tensor

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise RuntimeError(f"MONAI Transformation Error: {e}")
