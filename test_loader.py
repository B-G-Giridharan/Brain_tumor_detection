import os
import shutil
import tempfile
import numpy as np
from brain_tumor_ai.preprocessing.loader import load_mri_data

# Helper to create mock .npy files
def create_mock_npy(dir_path, name, shape=(128, 128, 64), value=1.0):
    full_path = os.path.join(dir_path, name)
    arr = np.full(shape, value, dtype=np.float32)
    np.save(full_path, arr)
    # np.save appends .npy if not present, but our name already has it
    return full_path

def test_loader_success(test_dir):
    """Test successful loading of all 4 modalities."""
    files = [
        create_mock_npy(test_dir, "brain_flair.npy", value=100),
        create_mock_npy(test_dir, "brain_t1.npy", value=200),
        create_mock_npy(test_dir, "brain_t1ce.npy", value=300),
        create_mock_npy(test_dir, "brain_t2.npy", value=400),
    ]
    
    print(f"Files created: {files}")
    stacked = load_mri_data(files)
    
    assert stacked.shape == (4, 128, 128, 64)
    assert np.all(stacked >= 0) and np.all(stacked <= 1)
    print("SUCCESS: Loader handled all 4 modalities correctly.")

def test_loader_missing_modality(test_dir):
    """Test failure when a modality is missing."""
    d = os.path.join(test_dir, "missing")
    os.makedirs(d, exist_ok=True)
    files = [
        create_mock_npy(d, "brain_flair.npy"),
        create_mock_npy(d, "brain_t1.npy"),
        create_mock_npy(d, "brain_t2.npy"),
    ]
    
    try:
        load_mri_data(files)
        assert False, "Should have raised ValueError for missing t1ce"
    except ValueError as e:
        assert "Missing required modalities: t1ce" in str(e)
        print("SUCCESS: Correctly detected missing t1ce.")

def test_loader_duplicate_modality(test_dir):
    """Test failure when a modality is duplicated."""
    d = os.path.join(test_dir, "duplicate")
    os.makedirs(d, exist_ok=True)
    files = [
        create_mock_npy(d, "brain_flair.npy"),
        create_mock_npy(d, "brain_t1_v1.npy"),
        create_mock_npy(d, "brain_t1_v2.npy"),
        create_mock_npy(d, "brain_t2.npy"),
    ]
    
    try:
        load_mri_data(files)
        assert False, "Should have raised ValueError for duplicate t1"
    except ValueError as e:
        assert "Duplicate modality detected for 't1'" in str(e)
        print("SUCCESS: Correctly detected duplicate t1.")

if __name__ == "__main__":
    t_dir = tempfile.mkdtemp()
    try:
        test_loader_success(t_dir)
        test_loader_missing_modality(t_dir)
        test_loader_duplicate_modality(t_dir)
        print("\n--- ALL TESTS PASSED ---")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(t_dir)
