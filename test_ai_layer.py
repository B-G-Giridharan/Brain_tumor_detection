import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from brain_tumor_ai.models.model_loader import load_model, _cached_model
from brain_tumor_ai.models.inference import run_inference
from brain_tumor_ai.config import TUMOR_VOLUME_THRESHOLD

def test_model_singleton():
    """Verify the model is loaded once and is the correct architecture."""
    print("Testing Singleton Model Loading...")
    
    # Reset global for clean test
    import brain_tumor_ai.models.model_loader as ml
    ml._cached_model = None
    
    m1 = load_model()
    m2 = load_model()
    
    assert m1 is m2, "Singleton failed: model instances are different!"
    assert "UNet" in str(type(m1)), f"Expected MONAI UNet, got {type(m1)}"
    assert next(m1.parameters()).device.type == "cpu", "Model should be on CPU"
    print("SUCCESS: Model loaded as Singleton on CPU.")

def test_inference_pipeline():
    """Verify the full inference pipeline with dummy data."""
    print("\nTesting Inference Pipeline...")
    
    # Mock input: (Batch, Channel, H, W, D)
    dummy_input = torch.randn(1, 4, 128, 128, 64)
    
    results = run_inference(dummy_input)
    
    # Check dictionary structure
    expected_keys = ["mask", "tumor_type", "confidence", "volume_voxels"]
    for key in expected_keys:
        assert key in results, f"Missing key '{key}' in results"
        
    # Check mask
    mask = results["mask"]
    assert isinstance(mask, np.ndarray), "Mask should be a numpy array"
    assert mask.shape == (128, 128, 64), f"Expected mask shape (128,128,64), got {mask.shape}"
    assert mask.dtype == np.uint8, f"Expected mask dtype uint8, got {mask.dtype}"
    
    # Check logical consistency
    vol = results["volume_voxels"]
    expected_type = "HGG" if vol > TUMOR_VOLUME_THRESHOLD else "LGG"
    assert results["tumor_type"] == expected_type, f"Classification mismatch: {results['tumor_type']} vs {expected_type}"
    
    print(f"SUCCESS: Inference completed. Type: {results['tumor_type']}, Volume: {vol} voxels.")

if __name__ == "__main__":
    try:
        test_model_singleton()
        test_inference_pipeline()
        print("\n--- ALL AI LAYER TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
