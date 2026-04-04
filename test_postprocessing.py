import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.models.postprocessing import postprocess_output
from brain_tumor_ai.config import MIN_TUMOR_VOXELS

def test_noise_removal():
    """Verify that small voxel clusters are removed while large ones stay."""
    print("Testing Noise Removal (Connected Component Analysis)...")
    
    # Create a 3D volume of zeros
    vol = np.zeros((128, 128, 64), dtype=np.float32)
    
    # 1. Add "Noise": Small cluster of 10 voxels (below threshold 100)
    vol[10:12, 10:12, 10:12] = 0.9
    
    # 2. Add "Tumor": Large cluster of 200 voxels (above threshold 100)
    vol[50:60, 50:60, 50:52] = 0.9  # 10*10*2 = 200 voxels
    
    # Run post-processing
    results = postprocess_output(vol)
    
    mask = results["mask"]
    volume = results["tumor_volume"]
    
    # The noise cluster (10x10x10 is 8 voxels if 2x2x2) 
    # Wait, 10:12 is 2x2x2 = 8 voxels.
    # 50:60, 50:60, 50:52 is 10x10x2 = 200 voxels.
    
    assert volume == 200.0, f"Expected volume 200, but got {volume}"
    assert results["tumor_detected"] is True, "Tumor should be detected"
    
    # Verify noise area is empty
    assert np.sum(mask[10:12, 10:12, 10:12]) == 0, "Noise cluster was not removed!"
    # Verify tumor area is present
    assert np.sum(mask[50:60, 50:60, 50:52]) == 200, "Large tumor cluster was incorrectly removed!"
    
    print(f"SUCCESS: Noise (8 voxels) removed. Tumor (200 voxels) preserved.")

def test_no_tumor_case():
    """Verify that if all clusters are small, no tumor is detected."""
    print("\nTesting No Tumor Detection (only noise)...")
    
    vol = np.zeros((128, 128, 64), dtype=np.float32)
    vol[5:10, 5:10, 5:8] = 0.9  # 5*5*3 = 75 voxels (below threshold 100)
    
    results = postprocess_output(vol)
    
    assert results["tumor_volume"] == 0.0, "Volume should be 0 after noise filtering"
    assert results["tumor_detected"] is False, "Tumor should NOT be detected"
    print("SUCCESS: 75-voxel cluster correctly identified as noise.")

if __name__ == "__main__":
    try:
        test_noise_removal()
        test_no_tumor_case()
        print("\n--- ALL POST-PROCESSING TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
