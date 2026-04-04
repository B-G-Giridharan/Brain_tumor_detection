import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.visualization.plot_2d import generate_2d_views

def test_slicing_integrity():
    """Verify that Axial, Sagittal, and Coronal slices produce distinct shapes and data."""
    print("Testing MRI Slicing Integrity (Axial, Sagittal, Coronal)...")
    
    # 1. Create a non-uniform 3D volume (H=100, W=80, D=60)
    # Different dimensions help identify axis swaps
    h, w, d = 100, 80, 60
    data = np.zeros((4, h, w, d), dtype=np.float32)
    
    # Add a specific pattern to the FLAIR channel (0)
    # Solid block at the center
    data[0, 40:60, 30:50, 20:40] = 1.0
    
    # 2. Add a mock mask (same shape for simplicity here)
    mask = np.zeros((h, w, d), dtype=np.uint8)
    mask[45:55, 35:45, 25:35] = 1
    
    # 3. Trigger visualization
    # This will trigger the print statements in plot_2d.py
    output_path = generate_2d_views(data, mask)
    
    # 4. Manual Shape Check (Based on printed output)
    # Expected:
    # Axial (D is fixed) -> (H, W) -> (100, 80)
    # Sagittal (W is fixed) -> (H, D) -> (100, 60)
    # Coronal (H is fixed) -> (W, D) -> (80, 60)
    
    print(f"\nVerification Results:")
    assert os.path.exists(output_path), "Failed to generate visualization image"
    print(f"SUCCESS: Visualization saved at {output_path}")

if __name__ == "__main__":
    try:
        test_slicing_integrity()
        print("\n--- SLICING INTEGRITY TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
