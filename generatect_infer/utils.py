from pathlib import Path
import torch

def load_latest_checkpoint(folder: Path, suffix: str = ".pt"):
    """Return path to most recent checkpoint in a folder."""
    files = [f for f in folder.glob(f"*{suffix}") if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {folder}")
    return max(files, key=lambda f: f.stat().st_mtime)


def adjust_num_frames(num_frames: int, temporal_patch_size: int) -> int:
    """Adjust number of frames to be divisible by temporal patch size."""
    adjusted = num_frames - ((num_frames - 1) % temporal_patch_size)
    return adjusted
