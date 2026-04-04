"""Professional 2D MRI Visualization module with tumor overlay support.

This module generates high-quality 2D slice views (Axial, Sagittal, Coronal) 
of MRI scans with optional tumor segmentation overlays.
"""

import os
import logging
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

def generate_2d_views(mri_data: np.ndarray, mask: Optional[np.ndarray] = None) -> str:
    """Generates a professional 3-view 2D MRI plot with optional tumor overlay.

    Args:
        mri_data (np.ndarray): Multi-modal MRI of shape (4, H, W, D).
        mask (Optional[np.ndarray]): Binary segmentation mask (H_m, W_m, D_m).

    Returns:
        str: Absolute path to the saved visualization image (mri_overlay.png).
    """
    logger.info("Generating professional 2D slice views...")

    try:
        # 1. Component Selection
        # Use FLAIR (index 0) for the grayscale background
        brain = mri_data[0]
        
        # Explicit debug logging as requested
        print(f"Brain shape: {brain.shape}")
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")

        # 2. Extract Dimensions
        h, w, d = brain.shape
        
        # 3. Extract Middle Slices (Symmetrical/Anatomical Centering)
        # Background slices
        slice_axial = brain[:, :, d // 2]
        slice_sagittal = brain[:, w // 2, :]
        slice_coronal = brain[h // 2, :, :]

        # 4. Plot Layout Setup
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='#020617')
        
        # Views mapping
        views = [
            (slice_axial, "Axial View"),
            (slice_sagittal, "Sagittal View"),
            (slice_coronal, "Coronal View")
        ]

        for i, (img_slice, title) in enumerate(views):
            # Show MRI background in grayscale
            # Using .T or specific flip for standard radiological orientation
            axes[i].imshow(img_slice.T if i != 0 else img_slice, cmap='gray', origin='lower')
            axes[i].set_title(title, color='white', pad=10)
            axes[i].axis('off')

            # 5. Handle Overlay (Mask Alignment)
            if mask is not None:
                h_m, w_m, d_m = mask.shape
                
                # Extract corresponding mask slice using its OWN shape info
                if i == 0:  # Axial (Axis 2 is fixed)
                    m_slice = mask[:, :, d_m // 2]
                elif i == 1:  # Sagittal (Axis 1 is fixed)
                    m_slice = mask[:, w_m // 2, :]
                else:  # Coronal (Axis 0 is fixed)
                    m_slice = mask[h_m // 2, :, :]

                # Only overlay if there's predictive data
                if np.max(m_slice) > 0:
                    # Robust Alignment: Zoom mask subset to match MRI resolution
                    zoom_factor = (img_slice.shape[0] / m_slice.shape[0], 
                                   img_slice.shape[1] / m_slice.shape[1])
                    
                    aligned_mask = zoom(m_slice, zoom_factor, order=0) # order 0 for labels
                    
                    # Apply Color Overlay (jet/red) with transparency
                    overlay = aligned_mask.T if i != 0 else aligned_mask
                    axes[i].imshow(overlay, cmap='jet', alpha=0.4, origin='lower')

        # 6. Save and Finish
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "mri_overlay.png")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        
        logger.info(f"Professional 2D Views saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
