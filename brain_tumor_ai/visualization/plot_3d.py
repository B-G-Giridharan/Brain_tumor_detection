"""Stable 3D Interactive visualization for brain MRI using Scatter3d point cloud.

This module uses go.Scatter3d (not go.Volume) for reliable rendering across
all environments. Brain tissue is shown as a semi-transparent point cloud,
and tumor is shown as a dense red cluster.
"""

import logging
from typing import Optional
import plotly.graph_objects as go
import numpy as np

logger = logging.getLogger(__name__)


def generate_3d_plot(mri_data: np.ndarray, mask: np.ndarray, theme: str = "grayscale") -> go.Figure:
    """Generates a stable, interactive 3D brain + tumor visualization.

    Uses Scatter3d point cloud for guaranteed rendering stability.

    Args:
        mri_data (np.ndarray): Multi-modal MRI scan of shape (4, H, W, D).
        mask (np.ndarray): Binary segmentation mask of shape (H, W, D).
        theme (str): Visual color theme ('grayscale' or 'thermal').

    Returns:
        go.Figure: A Plotly Figure with brain cloud and tumor overlay.
    """
    logger.info(f"Generating stable 3D point cloud (Theme: {theme})")

    try:
        # ── Step 1: Input Validation ──────────────────────────────────────────
        if mri_data is None:
            raise ValueError("MRI data is None")
        if mask is None:
            raise ValueError("Mask is required for visualization")

        # Use FLAIR modality (index 0) as anatomical background
        brain = mri_data[0]

        # ── Step 2: Normalize ─────────────────────────────────────────────────
        brain = brain.astype("float32")
        brain = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)

        # ── Step 3: Downsample (CRITICAL for performance) ─────────────────────
        # Stride of 3 reduces voxel count by 27x — smooth 60fps interaction
        brain = brain[::3, ::3, ::3]
        mask  = mask[::3, ::3, ::3]

        # ── Step 4: Shape Validation ──────────────────────────────────────────
        if brain.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch after downsampling: brain={brain.shape} mask={mask.shape}"
            )

        # ── Step 5: Coordinate Grid ───────────────────────────────────────────
        h, w, d = brain.shape
        grid_x, grid_y, grid_z = np.mgrid[0:h, 0:w, 0:d]

        # ── Step 6: Flatten Everything ────────────────────────────────────────
        x = grid_x.flatten()
        y = grid_y.flatten()
        z = grid_z.flatten()
        brain_values = brain.flatten()
        mask_values  = mask.flatten().astype(np.float32)

        print(f"[3D Plot] Grid shapes: x={x.shape} y={y.shape} z={z.shape} "
              f"brain={brain_values.shape} mask={mask_values.shape}")

        # ── Step 7: Filter Low-Intensity Brain Voxels ─────────────────────────
        # Removes empty black background — reveals only actual brain tissue
        threshold   = 0.1
        brain_mask  = brain_values > threshold

        x_brain      = x[brain_mask]
        y_brain      = y[brain_mask]
        z_brain      = z[brain_mask]
        brain_fil    = brain_values[brain_mask]

        # ── Step 8: Filter Tumor Voxels ───────────────────────────────────────
        tumor_mask = mask_values > 0
        x_tumor    = x[tumor_mask]
        y_tumor    = y[tumor_mask]
        z_tumor    = z[tumor_mask]

        print(f"[3D Plot] Brain points after filter: {len(x_brain):,}  |  "
              f"Tumor points: {len(x_tumor):,}")

        # ── Step 9: Brain Scatter3d Trace ─────────────────────────────────────
        brain_colorscale = "Gray" if theme == "grayscale" else "Portland"

        brain_trace = go.Scatter3d(
            x=x_brain,
            y=y_brain,
            z=z_brain,
            mode="markers",
            marker=dict(
                size=2,
                color=brain_fil,
                colorscale=brain_colorscale,
                opacity=0.08,
                showscale=False,
            ),
            name="Brain",
        )

        # ── Step 10: Tumor Scatter3d Trace ────────────────────────────────────
        tumor_trace = go.Scatter3d(
            x=x_tumor,
            y=y_tumor,
            z=z_tumor,
            mode="markers",
            marker=dict(
                size=4,
                color="red",
                opacity=0.85,
            ),
            name="Tumor",
        )

        # ── Step 11: Assemble Figure ──────────────────────────────────────────
        traces = [brain_trace]
        if len(x_tumor) > 0:          # Only add tumor layer if tumour detected
            traces.append(tumor_trace)

        fig = go.Figure(data=traces)

        # ── Step 12: Layout ───────────────────────────────────────────────────
        fig.update_layout(
            title={
                "text": f"3D Brain Tumor Visualization ({theme.capitalize()} Theme)",
                "x": 0.5,
                "xanchor": "center",
                "font": {"color": "white", "size": 20},
            },
            template="plotly_dark",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgb(2, 6, 23)",         # GFG dark background
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=600,
            legend=dict(
                font=dict(color="white", size=13),
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
            ),
        )

        logger.info("Stable 3D plot generated successfully.")
        return fig

    except Exception as e:
        logger.error(f"3D Visualization Error: {str(e)}")
        print(f"3D Visualization Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return go.Figure()
