"""Shared utility functions for Brain Tumor AI project.

This module provides reusable logic for common tasks like normalization,
file path validation, and data formatting.
"""

import os
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

def validate_mri_file(file_path: str) -> bool:
    """Validates that an MRI file exists and is in the correct format (.npy).

    Args:
        file_path (str): The absolute path to the file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File validation failed: {file_path} doesn't exist.")
        return False
        
    if not file_path.lower().endswith('.npy'):
        logger.warning(f"File validation failed: {file_path} is not a .npy file.")
        return False
        
    return True

def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """Applies Min-Max normalization to a NumPy array.

    Args:
        data (np.ndarray): The input numerical array.

    Returns:
        np.ndarray: The normalized array with values in [0, 1].
    """
    try:
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return data
        return (data - min_val) / (max_val - min_val)
    except Exception as e:
        logger.error(f"Normalization failed: {str(e)}")
        return data

# Additional placeholders for generic utilities
def setup_logging(log_level: str = "INFO") -> None:
    """Sets up global logging configuration.

    Args:
        log_level (str): The desired logging level (e.g., 'DEBUG', 'INFO').
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info(f"Logging initialized with level: {log_level}")
