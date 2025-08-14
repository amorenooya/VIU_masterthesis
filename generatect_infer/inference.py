import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- Import helper functions from your utils modules ---
from utils.model_utils import build_ctvit, build_transformer, load_latest_checkpoint
from utils.video_utils import adjust_num_frames, tensor_to_nifti


def run_inference(
    drive_root: Path,
    prompts_csv: Path,
    output_suffix: str = "",
    use_finetuned: bool = True
):
    """
    Run text-conditioned video generation using either fine-tuned or pretrained models.

    Parameters
    ----------
    drive_root : Path
        Root path of the project (contains pretrained_models and finetuning_results).
    prompts_csv : Path
        CSV file with columns 'id' and 'caption' for prompts.
    output_suffix : str, optional
        Extra label for output folder naming.
    use_finetuned : bool, optional
        If True, load latest fine-tuned checkpoints. If False, use pretrained ones.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load CTViT ---
    ctvit = build_ctvit(device)
    if use_finetuned:
        print("[INFO] Using fine-tuned CTViT")
        ctvit_ckpt_path = load_latest_checkpoint(drive_root / "finetuning_results")
    else:
        print("[INFO] Using pretrained CTViT")
        ctvit_ckpt_path = drive_root / "pretrained_models" / "ctvit_pretrained.pt"

    ctvit_ckpt = torch.load(ctvit_ckpt_path, map_location=device)
    ctvit.load_state_dict(ctvit_ckpt, strict=False)

    # --- Load Transformer ---
    if use_finetuned:
        transformer_path = load_latest_checkpoint(drive_root / "finetuning_results_transformer")
    else:
        transformer_path = drive_root / "pretrained_models" / "transformer_pretrained.pt"

    transformer_model = build_transformer(ctvit, device, transformer_path)

    # --- Load prompts ---
    labels_data = pd.read_csv(prompts_csv)
    texts_dict = dict(zip(labels_data["id"], labels_data["caption"]))
    print(f"[INFO] Loaded {len(texts_dict)} prompts.")

    # --- Adjust frames ---
    raw_num_frames = 110
    adjusted_num_frames = adjust_num_frames(raw_num_frames, ctvit.temporal_patch_size)
    if adjusted_num_frames != raw_num_frames:
        print(f"[WARNING] Adjusted frames from {raw_num_frames} to {adjusted_num_frames}")

    # --- Output folder ---
    mode_label = "finetuned" if use_finetuned else "pretrained"
    main_output_folder = drive_root / f"inference_{mode_label}_{adjusted_num_frames}{output_suffix}"
    main_output_folder.mkdir(parents=True, exist_ok=True)

    # --- Generation loop ---
    for i, (input_name, text) in tqdm(enumerate(texts_dict.items()), total=len(texts_dict)):
        sampled_videos_path = main_output_folder / f"samples.{input_name}_{i}"
        output_nii = sampled_videos_path / f"{input_name}_0.nii.gz"

        if output_nii.exists():
            print(f"[INFO] Skipping existing {output_nii}")
            continue

        sampled_videos_path.mkdir(parents=True, exist_ok=True)

        out = transformer_model.sample(
            texts=text,
            num_frames=adjusted_num_frames,
            cond_scale=5.0
        )

        for tensor in out.unbind(dim=0):
            tensor_to_nifti(tensor.cpu(), str(output_nii))
            with open(sampled_videos_path / f"{input_name}_0.txt", "w", encoding="utf-8") as f:
                f.write(text)

    print(f"[INFO] Inference complete. Results saved in: {main_output_folder}")


if __name__ == "__main__":
    DRIVE_ROOT = Path("/path/to/TFM")
    PROMPTS_CSV = DRIVE_ROOT / "finetune_generate" / "text_prompts.csv"
    
    # Change use_finetuned to False for pretrained models
    run_inference(DRIVE_ROOT, PROMPTS_CSV, output_suffix="", use_finetuned=True)
