import os
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.visualization.plot_2d import generate_2d_views

def test_fix_viz_mismatched_shapes():
    """Verify that plot_2d handles mismatched shapes without IndexError."""
    print("Testing 2D Visualization with mismatched shapes...")
    
    # Original BraTS-like MRI: (Channels, H, W, D)
    mri_data = np.random.rand(4, 240, 240, 155).astype(np.float32)
    
    # AI Processed Mask: (H_m, W_m, D_m)
    mask = np.zeros((128, 128, 64), dtype=np.uint8)
    # Add a mock "tumor" in the mask
    mask[40:80, 40:80, 20:40] = 1
    
    try:
        # Call visualization
        output_path = generate_2d_views(mri_data, mask)
        
        # 1. Check path validity
        assert os.path.exists(output_path), f"Output image not found at {output_path}"
        assert output_path.endswith(".png"), f"Expected .png file, got {output_path}"
        
        # 2. Check file size
        file_size = os.path.getsize(output_path)
        assert file_size > 0, "Generated image is empty"
        
        print(f"SUCCESS: Visualization generated at {output_path}")
        print(f"File size: {file_size} bytes")
        
    except IndexError as e:
        print(f"FAILURE: IndexError occurred! index out of bounds.")
        raise e
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        test_fix_viz_mismatched_shapes()
        print("\n--- VISUALIZATION FIX TEST PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
