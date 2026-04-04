import os
import numpy as np
import sys
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.visualization.plot_3d import generate_3d_plot

def test_hifi_figure_structure():
    """Verify that the Hi-Fi 3D plot contains the required anatomical rendering features."""
    print("Testing Hi-Fi 3D Volume Rendering Structure...")
    
    # 1. Mock Data: Small size for fast generation
    mri_data = np.random.rand(4, 16, 16, 16).astype(np.float32)
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    
    # 2. Generate Hi-Fi Plot
    fig = generate_3d_plot(mri_data, mask, theme="thermal")
    
    # 3. Assertions using to_plotly_json (most robust way)
    fig_json = fig.to_plotly_json()
    fig_data = fig_json['data']
    
    assert len(fig_data) == 2, "Figure should contain 2 traces"
    
    brain_trace = fig_data[0]
    
    # Check for Opacity Scale (the "discovery" magic)
    assert 'opacityscale' in brain_trace, "Hi-Fi rendering MUST include an opacityscale mapping"
    
    # Check for Color Bar
    assert brain_trace.get('showscale') is True, "Colorbar must be visible"
    
    # Check for Surface Density
    # Plotly 5.x+ often maps surface_count to 'surface' -> 'count' or just 'surface_count'
    # Actually, in to_plotly_json, it should be 'surface_count'
    found_count = brain_trace.get('surface_count')
    assert found_count == 30, f"Expected 30 surfaces, got {found_count}"
    
    # Check Theme mapping (Portland matches user screenshot)
    assert brain_trace.get('colorscale') == "Portland", f"Expected Portland, got {brain_trace.get('colorscale')}"
    
    print("SUCCESS: 3D Visualization figure contains all Hi-Fi anatomical discovery artifacts.")

if __name__ == "__main__":
    try:
        test_hifi_figure_structure()
        print("\n--- HI-FI 3D VISUALIZATION TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
