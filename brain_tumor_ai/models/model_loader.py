"""Model loading utility using the Singleton pattern.

This module ensures that the AI model (MONAI 3D UNet) is loaded 
only once into memory on the CPU for inference.
"""

import logging
from typing import Optional
import torch
from monai.networks.nets import UNet
from brain_tumor_ai.config import DEVICE

logger = logging.getLogger(__name__)

# Global variable to cache the loaded model
_cached_model: Optional[torch.nn.Module] = None

def load_model() -> torch.nn.Module:
    """Loads and returns the pretrained 3D UNet model.

    If the model is already loaded, it returns the cached instance.
    Otherwise, it initializes the MONAI architecture.

    Returns:
        torch.nn.Module: The initialized PyTorch model instance on CPU.

    Raises:
        RuntimeError: If there's an error during model initialization.
    """
    global _cached_model

    if _cached_model is not None:
        return _cached_model

    try:
        # 1. Initialize MONAI 3D UNet Architecture
        # Specs: 3D, 4 channels in (FLAIR, T1, T1CE, T2), 1 channel out (Tumor mask)
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,  # Default robustness unit
        )
        
        # 2. Force to CPU DEVICE
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        logger.info(f"Initialized MONAI 3D UNet on {device}.")
        
        # 3. Placeholder for loading weights
        # if os.path.exists(MODEL_PATH):
        #     model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        #     logger.info("Pretrained weights loaded.")
        # else:
        #     logger.warning("No pretrained weights found at MODEL_PATH. Using random initialization.")

        _cached_model = model
        return _cached_model

    except Exception as e:
        logger.error(f"Failed to initialize AI model: {str(e)}")
        raise RuntimeError(f"Model initialization failed: {e}")
