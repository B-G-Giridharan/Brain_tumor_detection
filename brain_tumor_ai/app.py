"""Main entry point for Brain Tumor Analysis Web Application.

This module initializes the Gradio web interface, handles file uploads, 
coordinates model analysis, and presents the 2D/3D visualizations 
and clinical report.
"""

import os
import logging
from typing import Any, List, Tuple
import gradio as gr
import numpy as np
import torch

from brain_tumor_ai.config import THEME_COLOR, DARK_MODE_BG
from brain_tumor_ai.preprocessing.loader import load_mri_data
from brain_tumor_ai.preprocessing.transforms import preprocess_mri
from brain_tumor_ai.models.inference import run_inference
from brain_tumor_ai.visualization.plot_2d import generate_2d_views
from brain_tumor_ai.visualization.plot_3d import generate_3d_plot
from brain_tumor_ai.reports.generator import generate_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("brain_tumor_ai")

def run_analysis(mri_files: List[Any], theme: str) -> Tuple[Any, Any, str]:
    """Coordinates the full brain tumor analysis pipeline.

    Args:
        mri_files (List[Any]): List of uploaded MRI modality files.
        theme (str): Visualization theme selected by the user.

    Returns:
        Tuple[Any, Any, str]: 2D view path, 3D Plotly figure, and clinical report.
    """
    logger.info("Analysis triggered via UI...")

    if not mri_files or len(mri_files) == 0:
        return None, None, "## ⚠️ Error: No MRI files provided.\nPlease upload 4 modalities (.npy)."

    try:
        # --- 1. Load data ---
        data = load_mri_data(mri_files)
        
        # --- 2. Preprocess ---
        tensor = preprocess_mri(data)
        
        # --- 3. Model Inference ---
        # Using real AI inference logic
        results = run_inference(tensor)
        mask = results['mask']
        
        # --- 4. Generate Visualizations & Report ---
        # Path string for 2D slices
        image_path = generate_2d_views(data, mask)
        
        # Plotly figure for 3D interaction
        plot_3d_figure = generate_3d_plot(data[0], theme)
        
        # Generate final formatted report
        report_text = generate_report(results)
        
        logger.info("Analysis successfully completed.")
        return image_path, plot_3d_figure, report_text

    except ValueError as ve:
        # User-facing validation errors
        logger.warning(f"Validation Error: {str(ve)}")
        return None, None, f"## ❌ Validation Error\n{str(ve)}"
    except Exception as e:
        # Unexpected system errors
        logger.error(f"Analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, f"## 🔧 System Error\nAn unexpected error occurred: {str(e)}"

# Build Gradio UI
with gr.Blocks(title="AI Brain Tumor Analysis") as app:
    gr.Markdown(f"# 🧠 AI-Powered Brain Tumor Analysis System")
    gr.Markdown("### Production-grade MRI segmentation and classification scaffold.")

    with gr.Row():
        with gr.Column(scale=1):
            mri_upload = gr.File(
                label="📁 Upload MRI Modalities (.npy) [Required: FLAIR, T1, T1CE, T2]", 
                file_count="multiple"
            )
            theme_choice = gr.Dropdown(
                label="🎨 Visualization Theme", 
                choices=["grayscale", "thermal", "rainbow", "plasma"], 
                value="grayscale"
            )
            analyze_btn = gr.Button("🔍 Run Analysis", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("🖼️ 2D Slices"):
                    slice_output = gr.Image(label="Multi-View Slices", type="filepath")
                with gr.TabItem("📊 3D Interactive"):
                    plot_output = gr.Plot(label="3D Volume Rendering")
                with gr.TabItem("📄 Clinical Report"):
                    report_output = gr.Markdown(label="Report Summary")

    # Connect button to pipeline
    analyze_btn.click(
        fn=run_analysis,
        inputs=[mri_upload, theme_choice],
        outputs=[slice_output, plot_output, report_output]
    )

if __name__ == "__main__":
    logger.info("Starting Gradio application server...")
    app.launch(theme=gr.themes.Default(primary_hue="green"))
