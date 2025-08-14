# -*- coding: utf-8 -*-
"""
Visualization utilities for GenerateCT outputs.
"""

import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap


def load_nifti(path: Path):
    """Load a NIfTI file and return the data array."""
    return nib.load(str(path)).get_fdata()


def wrap_prompt_text(prompt: str, width: int = 60):
    """Wrap prompt text into multiple lines for display."""
    return "\n".join(textwrap.wrap(prompt, width))


def plot_slice(data, axis, slice_idx, ax, title=None):
    """Plot a specific slice along the given axis."""
    if axis == "axial":
        img = data[:, :, slice_idx]
    elif axis == "sagittal":
        img = data[slice_idx, :, :]
    elif axis == "coronal":
        img = data[:, slice_idx, :]
    else:
        raise ValueError(f"Unknown axis: {axis}")

    ax.imshow(img.T if axis != "axial" else img, cmap="gray", origin="lower")
    ax.axis("off")
    if title:
        ax.set_title(title)


def show_middle_slices_all_axes(path: Path, prompt: str = None):
    """
    Show the middle slice for axial, sagittal, and coronal views in one figure.
    """
    data = load_nifti(path)
    
    mid_slices = {
        "sagittal": data.shape[0] // 2,
        "coronal": data.shape[1] // 2,
        "axial": data.shape[2] // 2
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for ax_idx, (view, slice_idx) in enumerate(mid_slices.items()):
        plot_slice(data, view, slice_idx, axes[ax_idx], title=f"{view.capitalize()} (slice {slice_idx})")
    
    if prompt:
        fig.suptitle(wrap_prompt_text(prompt), fontsize=12)
    
    plt.tight_layout()
    plt.show()


def show_four_axial_slices(path: Path, prompt: str = None):
    """Show four evenly spaced axial slices."""
    data = load_nifti(path)
    z_slices = data.shape[2]
    indices = [
        z_slices // 5,
        2 * z_slices // 5,
        3 * z_slices // 5,
        4 * z_slices // 5
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, idx in enumerate(indices):
        plot_slice(data, "axial", idx, axes[i], title=f"Axial {idx}")
    
    if prompt:
        fig.suptitle(wrap_prompt_text(prompt), fontsize=12)
    
    plt.tight_layout()
    plt.show()


def show_continuity(path: Path, axis: str = "axial", start_idx: int = None, num_slices: int = 4, prompt: str = None):
    """
    Show consecutive slices along a chosen axis to check spatial continuity.
    """
    data = load_nifti(path)
    
    if start_idx is None:
        start_idx = (data.shape[2] if axis == "axial" else data.shape[0] if axis == "sagittal" else data.shape[1]) // 2
    
    fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
    
    for i in range(num_slices):
        plot_slice(data, axis, start_idx + i, axes[i], title=f"{axis.capitalize()} {start_idx + i}")
    
    if prompt:
        fig.suptitle(wrap_prompt_text(prompt), fontsize=12)
    
    plt.tight_layout()
    plt.show()
