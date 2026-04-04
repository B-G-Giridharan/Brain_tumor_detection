import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.preprocessing.transforms import preprocess_mri

def test_preprocess_success():
    """Test successful preprocessing of a 4D NumPy array."""
    print("Testing successful preprocessing...")
    # Mock BrATS shape: (4, 240, 240, 155)
    mock_data = np.random.rand(4, 240, 240, 155).astype(np.float32)
    
    tensor = preprocess_mri(mock_data)
    
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
    assert tensor.shape == (1, 4, 128, 128, 64), f"Expected shape (1, 4, 128, 128, 64), got {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0, "Normalization failed"
    print("SUCCESS: Input shape (4, 240, 240, 155) transformed to (1, 4, 128, 128, 64).")

def test_preprocess_invalid_shape():
    """Test failure when input shape is incorrect."""
    print("Testing invalid shape detection...")
    # Wrong channel count
    bad_data = np.random.rand(3, 128, 128, 64)
    try:
        preprocess_mri(bad_data)
        assert False, "Should have raised ValueError for wrong channel count"
    except ValueError as e:
        assert "Input shape mismatch" in str(e)
        print("SUCCESS: Correctly detected wrong channel count.")

def test_preprocess_non_numpy():
    """Test failure when input is not a NumPy array."""
    print("Testing non-numpy input detection...")
    try:
        preprocess_mri([1, 2, 3])
        assert False, "Should have raised ValueError for non-numpy input"
    except ValueError as e:
        assert "Input must be a NumPy array" in str(e)
        print("SUCCESS: Correctly detected non-numpy input.")

if __name__ == "__main__":
    try:
        test_preprocess_success()
        test_preprocess_invalid_shape()
        test_preprocess_non_numpy()
        print("\n--- ALL PREPROCESSING TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
