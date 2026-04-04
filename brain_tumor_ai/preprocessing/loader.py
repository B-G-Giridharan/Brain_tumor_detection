"""Robust loading utilities for multi-modal MRI data.

This module provides a production-ready loader that validates and 
processes MRI scans (FLAIR, T1, T1CE, T2) from .npy files.
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def _identify_modalities(files: List[Any]) -> Dict[str, str]:
    """Identifies MRI modalities from a list of files based on naming keywords.

    Args:
        files (List[Any]): List of file objects or paths.

    Returns:
        Dict[str, str]: Map of modality name to actual file path/name.

    Raises:
        ValueError: If a modality is missing, duplicated, or invalid.
    """
    required = ["flair", "t1", "t1ce", "t2"]
    modality_map: Dict[str, str] = {}
    
    for file_obj in files:
        # Handle both string paths and Gradio file objects (which have .name)
        filename = getattr(file_obj, 'name', str(file_obj))
        basename = os.path.basename(filename).lower()
        
        # Check for .npy extension
        if not basename.endswith('.npy'):
            raise ValueError(f"Invalid file format: {basename}. Only .npy files are supported.")
            
        found_modality = None
        # Order matters for t1 vs t1ce
        for mod in sorted(required, key=len, reverse=True):
            if mod in basename:
                found_modality = mod
                break
        
        if found_modality:
            if found_modality in modality_map:
                raise ValueError(f"Duplicate modality detected for '{found_modality}' in {basename}")
            modality_map[found_modality] = filename
            logger.info(f"Identified {found_modality.upper()}: {basename}")
        else:
            raise ValueError(f"Could not identify modality for file: {basename}. "
                             f"Filename must contain one of: {required}")

    # Final check for missing modalities
    missing = [m for m in required if m not in modality_map]
    if missing:
        raise ValueError(f"Missing required modalities: {', '.join(missing)}")
        
    return modality_map

def _normalize(volume: np.ndarray) -> np.ndarray:
    """Applies robust Min-Max normalization to a 3D volume.

    Formula: (x - min) / (max - min + epsilon)

    Args:
        volume (np.ndarray): The raw 3D MRI volume.

    Returns:
        np.ndarray: Normalized volume in [0, 1] as float32.
    """
    volume = volume.astype(np.float32)
    min_val = np.min(volume)
    max_val = np.max(volume)
    denominator = max_val - min_val
    
    if denominator == 0:
        logger.warning("Volume has zero variance; normalization skipped to avoid NaN.")
        return np.zeros_like(volume)
        
    normalized = (volume - min_val) / (denominator + 1e-8)
    return np.clip(normalized, 0, 1)

def load_mri_data(files: List[Any]) -> np.ndarray:
    """Loads, validates, and stacks multi-modal MRI data.

    Args:
        files (List[Any]): List of 4 uploaded .npy files.

    Returns:
        np.ndarray: Stacked MRI data of shape (4, H, W, D) in order [flair, t1, t1ce, t2].

    Raises:
        ValueError: If validation (modality, format, shape) fails.
    """
    logger.info(f"Processing {len(files)} uploaded MRI files...")
    
    # 1. Identify and Validate Files
    modality_map = _identify_modalities(files)
    
    # 2. Load and Shape Validation
    volumes: List[np.ndarray] = []
    required_order = ["flair", "t1", "t1ce", "t2"]
    common_shape: Optional[Tuple[int, ...]] = None
    
    for mod in required_order:
        path = modality_map[mod]
        try:
            vol = np.load(path)
            
            # Type and Shape enforcement
            if not isinstance(vol, np.ndarray):
                raise ValueError(f"Loaded data for {mod} is not a valid numpy array.")
            
            if common_shape is None:
                common_shape = vol.shape
                logger.info(f"Reference shape set from {mod.upper()}: {common_shape}")
            elif vol.shape != common_shape:
                raise ValueError(f"Shape mismatch: {mod.upper()} {vol.shape} does not match {common_shape}")
                
            # Normalize and add to list
            normalized_vol = _normalize(vol)
            volumes.append(normalized_vol)
            
        except Exception as e:
            if isinstance(e, ValueError): raise
            raise ValueError(f"Failed to load {mod.upper()} from {os.path.basename(path)}: {str(e)}")

    # 3. Stack and Finalize
    # Outcome shape: (4, H, W, D)
    stacked_data = np.stack(volumes, axis=0)
    logger.info(f"Final MRI stack successfully created. Shape: {stacked_data.shape}")
    
    return stacked_data
