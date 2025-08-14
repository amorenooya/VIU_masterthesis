import os
from glob import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
import imageio
import pandas as pd
import torch
from torchvision import transforms
from torchmetrics.multimodal import CLIPScore

# -----------------------------
# User-defined paths
# -----------------------------
drive_root = "/content/drive/MyDrive/TFM"
sample_dirs = glob(os.path.join(drive_root, "prompt_influence/sex/generated_data_109_2/*"))
output_base_dir = os.path.join(drive_root, "prompt_influence/sex/generated_data_109_2")

# -----------------------------
# Helper function
# -----------------------------
def normalize_slice(slice_2d):
    slice_min, slice_max = slice_2d.min(), slice_2d.max()
    if slice_max - slice_min == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    slice_norm = (slice_2d - slice_min) / (slice_max - slice_min) * 255
    return slice_norm.astype(np.uint8)

# -----------------------------
# Process each sample folder
# -----------------------------
for sample_path in tqdm(sample_dirs):
    sample_id = os.path.basename(sample_path)

    # Load prompt to determine sex
    txt_files = glob(os.path.join(sample_path, "*.txt"))
    if not txt_files:
        print(f"[WARNING] No .txt file found in {sample_id}, skipping.")
        continue

    with open(txt_files[0], "r") as f:
        prompt = f.read().strip().lower()

    if "female" in prompt:
        sex_label = "female"
    elif "male" in prompt:
        sex_label = "male"
    else:
        print(f"[WARNING] Could not determine sex from prompt in {sample_id}: {prompt}")
        continue

    # Find NIfTI file inside folder
    nii_files = glob(os.path.join(sample_path, '*.nii')) + glob(os.path.join(sample_path, '*.nii.gz'))
    if not nii_files:
        print(f"No .nii or .nii.gz file found in {sample_path}, skipping.")
        continue

    nii_path = nii_files[0]
    try:
        volume = nib.load(nii_path).get_fdata()
    except Exception as e:
        print(f"Could not load {nii_path}: {e}")
        continue

    H, W, D = volume.shape
    if D < 16:
        print(f"Skipping {sample_id}: only {D} slices.")
        continue

    # Create output folder under male/ or female/
    out_dir = os.path.join(output_base_dir, sex_label, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    # Save each slice
    for i, idx in enumerate(range(D)):
        slice_2d = volume[:, :, idx]
        slice_2d = normalize_slice(slice_2d)
        out_path = os.path.join(out_dir, f"{i:03d}.png")
        imageio.imwrite(out_path, slice_2d)

# -----------------------------
# Run FID and FVD metrics
# -----------------------------
metrics_commands = [
    f"python src/scripts/calc_metrics_for_dataset.py --metrics fid50k_full --real_data_path {drive_root}/real_data_128_2 --fake_data_path {output_base_dir}/male --resolution 128",
    f"python src/scripts/calc_metrics_for_dataset.py --metrics fid50k_full --real_data_path {drive_root}/real_data_128_2 --fake_data_path {output_base_dir}/female --resolution 128",
    f"python src/scripts/calc_metrics_for_dataset.py --metrics fvd2048_16f --real_data_path {drive_root}/real_data_128 --fake_data_path {drive_root}/prompt_influence/sex/generated_data_109/male --resolution 128",
    f"python src/scripts/calc_metrics_for_dataset.py --metrics fvd2048_16f --real_data_path {drive_root}/real_data_128 --fake_data_path {drive_root}/prompt_influence/sex/generated_data_109/female --resolution 128"
]

for cmd in metrics_commands:
    os.system(cmd)
  
# -----------------------------
# Compute CLIP scores
# -----------------------------
clipscore = CLIPScore().to("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])

def load_nii_as_image(nii_path):
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    slice_index = data.shape[2] // 2
    slice_img = data[:, :, slice_index]
    slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255).astype(np.uint8)
    return Image.fromarray(cv2.resize(slice_img, (224, 224))).convert("RGB")

results = []
root_dir = os.path.join(drive_root, "inference_109")

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if os.path.isdir(subfolder_path):
        sample_id = subfolder.split('.')[-1].split('_')[0]
        nii_path = os.path.join(subfolder_path, f"{sample_id}_0.nii.gz")
        txt_path = os.path.join(subfolder_path, f"{sample_id}_0.txt")
        if os.path.exists(nii_path) and os.path.exists(txt_path):
            image = load_nii_as_image(nii_path)
            with open(txt_path, 'r') as f:
                prompt = f.read().strip()
            image_tensor = transform(image).unsqueeze(0).to(clipscore.device)
            score = clipscore(image_tensor, [prompt]).item()
            results.append({'sample_id': sample_id, 'clip_score': score})

pd.DataFrame(results).to_csv(os.path.join(drive_root, "inference_109/clip_scores.csv"), index=False)
