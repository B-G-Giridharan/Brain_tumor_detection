"""Test stable 3D point cloud visualization rebuild."""
import numpy as np
import sys
import os
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from brain_tumor_ai.visualization.plot_3d import generate_3d_plot


def test_stable_trace_types():
    """Ensure traces are Scatter3d and not Volume."""
    print("Test 1: Checking trace types...")
    mri_data = np.random.rand(4, 64, 64, 32).astype("float32")
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[20:40, 20:40, 10:20] = 1  # mock tumor block

    fig = generate_3d_plot(mri_data, mask, theme="grayscale")

    assert isinstance(fig, go.Figure), "Output must be a Plotly Figure"
    assert len(fig.data) >= 1, "Figure must have at least 1 trace"

    for trace in fig.data:
        assert isinstance(trace, go.Scatter3d), (
            f"Expected Scatter3d, got {type(trace).__name__}"
        )
    print(f"  PASS — {len(fig.data)} Scatter3d trace(s) found.")


def test_brain_points_not_empty():
    """Brain trace must contain a meaningful number of points."""
    print("Test 2: Checking brain point count...")
    mri_data = np.random.rand(4, 64, 64, 32).astype("float32")
    # push values above threshold so filter passes
    mri_data[0] = mri_data[0] * 0.9 + 0.15
    mask = np.zeros((64, 64, 32), dtype=np.uint8)

    fig = generate_3d_plot(mri_data, mask, theme="thermal")
    brain_trace = fig.data[0]
    assert len(brain_trace.x) > 100, "Expected > 100 brain points after threshold filter"
    print(f"  PASS — {len(brain_trace.x):,} brain points rendered.")


def test_tumor_trace_present():
    """When mask has voxels, a tumor Scatter3d trace must be added."""
    print("Test 3: Checking tumor trace presence...")
    mri_data = np.random.rand(4, 64, 64, 32).astype("float32")
    mri_data[0] = mri_data[0] * 0.9 + 0.15
    mask = np.zeros((64, 64, 32), dtype=np.uint8)
    mask[15:30, 15:30, 8:16] = 1

    fig = generate_3d_plot(mri_data, mask)
    assert len(fig.data) == 2, f"Expected 2 traces (brain + tumor), got {len(fig.data)}"
    tumor_trace = fig.data[1]
    assert len(tumor_trace.x) > 0, "Tumor trace has no points"
    print(f"  PASS — Tumor trace has {len(tumor_trace.x):,} points.")


def test_no_tumor_single_trace():
    """When mask is empty, only 1 trace should be rendered."""
    print("Test 4: Checking single-trace when no tumor...")
    mri_data = np.random.rand(4, 64, 64, 32).astype("float32")
    mri_data[0] = mri_data[0] * 0.9 + 0.15
    mask = np.zeros((64, 64, 32), dtype=np.uint8)  # no tumor

    fig = generate_3d_plot(mri_data, mask)
    assert len(fig.data) == 1, f"Expected 1 trace when no tumor, got {len(fig.data)}"
    print("  PASS — Only brain trace rendered when no tumor detected.")


if __name__ == "__main__":
    try:
        test_stable_trace_types()
        test_brain_points_not_empty()
        test_tumor_trace_present()
        test_no_tumor_single_trace()
        print("\n✅ ALL STABLE 3D VISUALIZATION TESTS PASSED")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
