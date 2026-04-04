"""2D Slice visualization for brain MRI scans.

This module provides functions to generate axial, sagittal, and 
coronal views from 3D MRI volumes with optional segmentation overlays.
"""

import os
import logging
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def generate_2d_views(mri_data: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Generates and saves a three-view 2D slice plot as an image file.

    Args:
        mri_data (np.ndarray): 4-channel multi-modal MRI of shape (4, H, W, D).
        mask (Optional[np.ndarray]): Binary segmentation mask overlay (H, W, D).

    Returns:
        str: Absolute path to the saved 2D visualization image.
    """
    logger.info("Generating 2D slice views...")

    try:
        # 1. Directory Setup
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "analysis_slices.png")

        # 2. Slice Extraction (Using channel 0 - typically FLAIR)
        # Using middle index for each dimension
        _, h, w, d = mri_data.shape
        
        # Axial: Transverse (fixed H)
        slice_axial = mri_data[0, h // 2, :, :]
        # Sagittal: Side-view (fixed W)
        slice_sagittal = mri_data[0, :, w // 2, :]
        # Coronal: Front-view (fixed D)
        slice_coronal = mri_data[0, :, :, d // 2]

        # 3. Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        views = [
            (slice_axial, "Axial View"),
            (slice_sagittal, "Sagittal View"),
            (slice_coronal, "Coronal View")
        ]

        for i, (img_slice, title) in enumerate(views):
            axes[i].imshow(img_slice, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')

            # Overlay mask if provided
            if mask is not None:
                # Mask must align with the slice orientation
                if i == 0:  # Axial
                    m_slice = mask[h // 2, :, :]
                elif i == 1:  # Sagittal
                    m_slice = mask[:, w // 2, :]
                else:  # Coronal
                    m_slice = mask[:, :, d // 2]
                
                # Check for any values in mask slice
                if np.max(m_slice) > 0:
                    axes[i].imshow(m_slice, cmap='jet', alpha=0.5)

        # 4. Save and Cleanup
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        print(f"Saved image at: {output_path}")
        logger.info(f"2D Views successfully saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to generate 2D views: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
