import os
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.visualization.plot_2d import generate_2d_views

def test_professional_viz():
    """Verify that professional visualization generates the correct output file."""
    print("Testing Professional 2D Visualization...")
    
    # 1. Simulate MRI data (4, 240, 240, 155)
    mri_data = np.random.rand(4, 240, 240, 155).astype(np.float32)
    
    # 2. Simulate AI Mask (128, 128, 64)
    mask = np.zeros((128, 128, 64), dtype=np.uint8)
    mask[20:100, 20:100, 10:50] = 1 # A nice big tumor block
    
    # 3. Trigger visualization
    output_path = generate_2d_views(mri_data, mask)
    
    # Check results
    assert os.path.exists(output_path), f"Output image not found at {output_path}"
    assert "mri_overlay.png" in output_path, f"Output filename mismatch: {output_path}"
    
    print(f"SUCCESS: Visualization saved at {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")

def test_viz_no_mask():
    """Verify that visualization works even when mask is None."""
    print("\nTesting Visualization without mask...")
    mri_data = np.random.rand(4, 128, 128, 64).astype(np.float32)
    output_path = generate_2d_views(mri_data, mask=None)
    assert os.path.exists(output_path), "Failed to generate image without mask"
    print(f"SUCCESS: Image generated without mask: {output_path}")

if __name__ == "__main__":
    try:
        test_professional_viz()
        test_viz_no_mask()
        print("\n--- ALL PROFESSIONAL VIZ TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
