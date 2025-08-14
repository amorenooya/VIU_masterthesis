# -*- coding: utf-8 -*-
"""
Visualization utilities for GenerateCT outputs.
"""

import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path


def load_nifti(path: Path):
    """Load a NIfTI file and return the data array."""
    return nib.load(str(path)).get_fdata()


def compare_pairs(real_path: Path, generated_path: Path, slice_idx: int = None):
    """Plot a real vs generated CT slice side by side."""
    real_data = load_nifti(real_path)
    gen_data = load_nifti(generated_path)

    if slice_idx is None:
        slice_idx = real_data.shape[2] // 2  # middle slice

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(real_data[:, :, slice_idx], cmap="gray")
    axes[0].set_title("Real CT")
    axes[0].axis("off")

    axes[1].imshow(gen_data[:, :, slice_idx], cmap="gray")
    axes[1].set_title("Generated CT")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
