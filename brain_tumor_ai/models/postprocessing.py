"""Post-processing module for tumor segmentation refinement.

This module converts raw probability maps from the AI model into 
clean, clinically relevant binary masks by removing noise and 
calculating tumor metrics.
"""

import logging
from typing import Any, Dict, Union
import numpy as np
import torch
from scipy.ndimage import label
from brain_tumor_ai.config import PROBABILITY_THRESHOLD, MIN_TUMOR_VOXELS

logger = logging.getLogger(__name__)

def postprocess_output(prob_map: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
    """Refines model predictions into binary masks with noise removal.

    Args:
        prob_map (Union[np.ndarray, torch.Tensor]): Model probability 
            map output (e.g., shape (1, 1, 128, 128, 64) or (128, 128, 64)).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'mask': Refined binary mask (np.uint8, 0 or 1).
            - 'tumor_volume': Total tumor voxel count after filtering (float).
            - 'tumor_detected': Boolean flag indicating if tumor exists.

    Raises:
        ValueError: If input is invalid or empty.
    """
    logger.info("Initializing post-processing logic...")

    if prob_map is None:
        raise ValueError("Probability map input cannot be None.")

    # 1. Standardize to NumPy and remove Batch/Channel dimensions
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.detach().cpu().numpy()
    
    # Squeeze to 3D (H, W, D)
    prob_map = np.squeeze(prob_map)
    
    if prob_map.ndim != 3:
        raise ValueError(f"Input must be 3D or squeezable to 3D. Got shape {prob_map.shape}")

    # 2. Binary Thresholding
    mask = (prob_map > PROBABILITY_THRESHOLD).astype(np.uint8)

    # 3. Noise Removal (Connected Component Analysis)
    # Identify clusters of voxels
    labeled_mask, num_features = label(mask)
    
    if num_features > 0:
        logger.info(f"Detected {num_features} potential tumor clusters. Filtering noise...")
        
        # Calculate size of each component
        refined_mask = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            comp_mask = (labeled_mask == i)
            comp_size = np.sum(comp_mask)
            
            # Keep only clusters larger than MIN_TUMOR_VOXELS
            if comp_size >= MIN_TUMOR_VOXELS:
                refined_mask[comp_mask] = 1
        
        mask = refined_mask
    else:
        logger.info("No tumor voxels detected above threshold.")

    # 4. Metric Calculations
    tumor_volume = float(np.sum(mask))
    tumor_detected = tumor_volume > 0

    logger.info(f"Post-processing complete. Volume: {tumor_volume} voxels | Detected: {tumor_detected}")

    return {
        "mask": mask,
        "tumor_volume": tumor_volume,
        "tumor_detected": tumor_detected
    }
