"""Model inference pipeline for brain tumor analysis.

This module handles the forward pass of the AI model and provides 
the structured output (mask, classification, confidence) for the UI.
"""

import logging
from typing import Any, Dict
import torch
import numpy as np
from brain_tumor_ai.models.model_loader import load_model
from brain_tumor_ai.models.postprocessing import postprocess_output
from brain_tumor_ai.config import TUMOR_VOLUME_THRESHOLD

logger = logging.getLogger(__name__)

def run_inference(mri_tensor: torch.Tensor) -> Dict[str, Any]:
    """Executes the AI model on the preprocessed MRI tensor.

    Args:
        mri_tensor (torch.Tensor): The preprocessed multi-modal MRI tensor 
            of shape (1, 4, 128, 128, 64).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'mask': Binary 3D segmentation mask (np.uint8, shape (128,128,64)).
            - 'tumor_type': Classified tumor category ("HGG" or "LGG").
            - 'confidence': Mean probability score (float).
            - 'volume_voxels': Total tumor voxel count (int).
            - 'tumor_detected': Boolean flag.
    """
    logger.info(f"Starting model inference on tensor of shape {mri_tensor.shape}...")
    
    # 1. Load model via singleton
    model = load_model()

    try:
        # 2. Forward Pass
        # Ensure model is in evaluation mode
        model.eval()
        with torch.no_grad():
            output = model(mri_tensor)
            
            # 3. Apply Activation (Sigmoid for binary segmentation)
            probs = torch.sigmoid(output)
            
            # 4. Refined Post-processing
            # Convert raw 5D probability Map (1, 1, 128, 128, 64) -> Clean 3D Metrics
            # Post-processing removes noise/small clusters < MIN_TUMOR_VOXELS
            logger.info("Triggering post-processing refinement...")
            post_results = postprocess_output(probs)
            
            mask = post_results['mask']
            volume_voxels = int(post_results['tumor_volume'])
            tumor_detected = post_results['tumor_detected']
            
            # 5. Rule-based Classification
            if not tumor_detected:
                tumor_type = "No Tumor Detected"
                logger.info("Classification: No Tumor Detected.")
            elif volume_voxels > TUMOR_VOLUME_THRESHOLD:
                tumor_type = "HGG"
                logger.info(f"Classification: High-Grade Glioma (HGG) [Volume: {volume_voxels}]")
            else:
                tumor_type = "LGG"
                logger.info(f"Classification: Low-Grade Glioma (LGG) [Volume: {volume_voxels}]")
                
            # Average probability in the output heatmap as confidence
            # (Note: Use original probs map for confidence to reflect model's overall certainty)
            prob_map = probs.squeeze().cpu().numpy()
            confidence = float(np.mean(prob_map))

            logger.info("Inference successfully completed.")
            
            return {
                "mask": mask,
                "tumor_type": tumor_type,
                "confidence": confidence,
                "volume_voxels": volume_voxels,
                "tumor_detected": tumor_detected
            }
            
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback to empty/error response
        return {
            "mask": np.zeros((128, 128, 64), dtype=np.uint8),
            "tumor_type": "Unknown",
            "confidence": 0.0,
            "volume_voxels": 0,
            "tumor_detected": False,
            "error": str(e)
        }
