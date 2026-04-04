import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.visualization.plot_3d import generate_3d_plot

def test_3d_coordinate_synchronization():
    """Verify that all 5 arrays (x, y, z, brain, mask) have perfectly matching shapes."""
    print("Testing 3D Coordinate Synchronization (Fixing Origin Collapse)...")
    
    # 1. Mock Data: (4, 64, 64, 32)
    mri_data = np.random.rand(4, 64, 64, 32).astype(np.float32)
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[20:40, 20:40, 10:20] = 1 # Mock tumor
    
    # 2. Generate Figure
    # This will trigger the shapes print in plot_3d.py
    fig = generate_3d_plot(mri_data, mask)
    
    # 3. Assert trace properties
    brain_trace = fig.data[0]
    tumor_trace = fig.data[1]
    
    # All coordinate lists must be the same length
    len_x = len(brain_trace.x)
    len_y = len(brain_trace.y)
    len_z = len(brain_trace.z)
    len_v = len(brain_trace.value)
    
    len_tm_v = len(tumor_trace.value)
    
    print(f"Verified Trace Lengths: x={len_x}, y={len_y}, z={len_z}, brain_v={len_v}, tumor_v={len_tm_v}")
    
    assert len_x == len_y == len_z == len_v == len_tm_v, "FATAL: Array length mismatch in 3D traces!"
    
    # Downsampling Check: (64/2)*(64/2)*(32/2) = 32*32*16 = 16384
    expected_len = 16384
    assert len_x == expected_len, f"Expected 16384 points (64/2 * 64/2 * 32/2), got {len_x}"
    
    print("SUCCESS: All 3D coordinate and value arrays are perfectly synchronized.")

if __name__ == "__main__":
    try:
        test_3d_coordinate_synchronization()
        print("\n--- 3D COORDINATE FIX TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
