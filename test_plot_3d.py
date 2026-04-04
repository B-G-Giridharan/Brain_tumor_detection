import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.visualization.plot_3d import generate_3d_plot

def test_3d_figure_generation():
    """Verify that the 3D plot is generated with correct traces and layout."""
    print("Testing 3D Volume Rendering Generation...")
    
    # 1. Mock Data
    # MRI: (4, 64, 64, 32)
    mri_data = np.random.rand(4, 64, 64, 32).astype(np.float32)
    # Mask: (64, 64, 32)
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[10:30, 10:30, 5:15] = 1
    
    # 2. Generate Plot
    fig = generate_3d_plot(mri_data, mask, theme="grayscale")
    
    # 3. Assertions
    assert isinstance(fig, go.Figure), "Output must be a Plotly Figure"
    
    # Check trace count (should be 2: Brain + Tumor)
    assert len(fig.data) == 2, f"Expected 2 traces, found {len(fig.data)}"
    
    # Check trace types
    assert isinstance(fig.data[0], go.Volume), "Trace 0 should be go.Volume (Brain)"
    assert isinstance(fig.data[1], go.Volume), "Trace 1 should be go.Volume (Tumor)"
    
    # Check colorscale
    assert fig.data[0].colorscale[0][1] == 'rgb(0, 0, 0)', "Grayscale theme should use Gray scale"
    
    # Test Thermal Theme
    fig_thermal = generate_3d_plot(mri_data, mask, theme="thermal")
    # Hot scale usually starts with black or dark red
    assert fig_thermal.data[0].colorscale is not None
    
    print("SUCCESS: 3D Visualization figure generated correctly with 2 Volume traces.")

if __name__ == "__main__":
    try:
        test_3d_figure_generation()
        print("\n--- 3D VISUALIZATION TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
