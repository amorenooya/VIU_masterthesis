#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import torch
from omegaconf import OmegaConf
from super_resolution import Unet, ElucidatedSuperres, SuperResolutionTrainer, NullUnet
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from super_resolution.superres_pytorch import Superres

# ========================
# CONFIGURATION
# ========================
drive_root = "/path/to/your/drive"  # <-- UPDATE THIS

config_path = os.path.join(drive_root, "super_resolution", "superres.yaml")
lowres_folder = os.path.join(drive_root, "inference_89")
output_folder = os.path.join(drive_root, "superres_output_89")
weight_path = os.path.join(drive_root, "pretrained_models", "superres_pretrained.pt")

# ========================
# DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# LOAD CONFIG AND MODELS
# ========================
config = OmegaConf.load(config_path)

unet1 = NullUnet()
unet2 = Unet(**config.unets.unet1, lowres_cond=True)

superres_klass = ElucidatedSuperres if config.superres.get('elucidated', False) else Superres
superres = superres_klass(
    unets=(unet1, unet2),
    **OmegaConf.to_container(config.superres.params),
)

infer = SuperResolutionTrainer(
    superres=superres,
    **config.trainer.params,
).to(device)

infer.load(weight_path)

os.makedirs(output_folder, exist_ok=True)

# ========================
# UTILITY FUNCTIONS
# ========================
def load_lowres_nifti(filepath):
    nii = nib.load(filepath)
    data = nii.get_fdata()
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    elif data.ndim != 3:
        raise ValueError(f"Unexpected image dimensions: {data.shape}")
    tensor = torch.from_numpy(data).float()
    return tensor.unsqueeze(0).unsqueeze(0)

def read_prompt(filepath):
    with open(filepath, 'r') as f:
        return f.read().strip()

# ========================
# MAIN SUPER-RESOLUTION LOOP
# ========================
def superres_case(case_folder):
    case_path = os.path.join(lowres_folder, case_folder)
    if not os.path.isdir(case_path):
        return

    nifti_file = None
    prompt_file = None
    for f in os.listdir(case_path):
        if f.endswith('.nii.gz'):
            nifti_file = os.path.join(case_path, f)
        elif 'prompt' in f.lower() or f.endswith('.txt'):
            prompt_file = os.path.join(case_path, f)

    if not nifti_file:
        print(f"No NIfTI file found in {case_folder}")
        return

    output_case_folder = os.path.join(output_folder, case_folder)
    output_path = os.path.join(output_case_folder, f"superres_{os.path.basename(nifti_file)}")

    if os.path.exists(output_path):
        print(f"Already processed: {case_folder}, skipping.")
        return

    lowres_img = load_lowres_nifti(nifti_file).to(device)
    prompt = [read_prompt(prompt_file)] if prompt_file else [""]

    _, _, D, H, W = lowres_img.shape
    highres_slices = []

    for d in range(W):
        slice_img = lowres_img[:, :, :, :, d]
        with torch.no_grad():
            highres_slice = infer.sample(
                cond_scale=config.checkpoint.cond_scale,
                texts=prompt,
                start_image_or_video=slice_img,
                start_at_unet_number=2,
            ).detach().cpu()
        highres_slices.append(highres_slice)

        if d % 10 == 0:
            plt.figure(figsize=(10, 5))
            lowres_slice = slice_img.squeeze().cpu().numpy()
            highres_np = highres_slice.squeeze().numpy()
            plt.subplot(1, 2, 1)
            plt.imshow(lowres_slice, cmap='gray')
            plt.title(f"Low-res slice {d}, shape {lowres_slice.shape}")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(highres_np, cmap='gray')
            plt.title(f"Super-res slice {d}, shape {highres_np.shape}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    highres_volume = torch.cat(highres_slices, dim=0).permute(1, 2, 3, 0).unsqueeze(0)
    os.makedirs(output_case_folder, exist_ok=True)
    nib.save(nib.Nifti1Image(highres_volume.squeeze().numpy(), np.eye(4)), output_path)

    if prompt_file:
        shutil.copy(prompt_file, os.path.join(output_case_folder, os.path.basename(prompt_file)))

    print(f"Processed: {case_folder}")

# ========================
# EXECUTION
# ========================
if __name__ == "__main__":
    for case_folder in os.listdir(lowres_folder):
        superres_case(case_folder)

    print("Super-resolution completed!")
