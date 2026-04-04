import numpy as np
import os
import sys

# Add the project directory to sys.path
sys.path.append('d:/Python/Detection')

from brain_tumor_ai.visualization.plot_2d import generate_2d_views

# Mock data (4, 128, 128, 64)
data = np.random.rand(4, 128, 128, 64)
# Mock mask (128, 128, 64)
mask = np.zeros((128, 128, 64))
mask[50:80, 50:80, 20:40] = 1

print("Testing 2D slice generation...")
path = generate_2d_views(data, mask)

if os.path.exists(path):
    print(f"SUCCESS: Image saved at {path}")
    print(f"File size: {os.path.getsize(path)} bytes")
else:
    print(f"FAILED: Image not found at {path}")
