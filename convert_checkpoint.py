#!/usr/bin/env python3
"""
Convert veRL FSDP checkpoint to HuggingFace format for inference.

Usage:
    python convert_checkpoint.py /path/to/global_step_XXX
    python convert_checkpoint.py /path/to/outputs/run_name  # converts all checkpoints
"""
from pathlib import Path

import torch


def convert_single_checkpoint(ckpt_dir: Path, base_model_path: str | None = None):
    """Convert a single veRL checkpoint to HuggingFace format.

    Handles both GRPO (has actor/ subdir) and SFT (weights directly in step dir).
    """
    # Determine layout: GRPO has actor/, SFT does not
    actor_dir = ckpt_dir / "actor"
    if actor_dir.exists():
        weights_dir = actor_dir
        hf_dir = actor_dir / "huggingface"
    else:
        weights_dir = ckpt_dir
        hf_dir = ckpt_dir / "huggingface"

    if not hf_dir.exists():
        print(f"[SKIP] No huggingface dir: {ckpt_dir}")
        return False

    # Check if already converted
    safetensors_file = hf_dir / "model.safetensors"
    pytorch_file = hf_dir / "pytorch_model.bin"
    if safetensors_file.exists() or pytorch_file.exists():
        print(f"[SKIP] Already converted: {ckpt_dir.name}")
        return True

    # Find the model checkpoint file
    model_file = weights_dir / "model_world_size_1_rank_0.pt"
    if not model_file.exists():
        model_files = list(weights_dir.glob("model_world_size_*.pt"))
        if not model_files:
            print(f"[SKIP] No model weights found: {ckpt_dir}")
            return False
        model_file = model_files[0]
        print(f"[WARN] Using {model_file.name} (multi-GPU checkpoint may need special handling)")

    print(f"[INFO] Converting: {ckpt_dir.parent.name}/{ckpt_dir.name}")

    # Load config from huggingface dir
    config_file = hf_dir / "config.json"
    if not config_file.exists():
        print(f"[ERROR] No config.json in {hf_dir}")
        return False

    # Load the state dict
    print(f"  Loading weights from {model_file.name}...")
    state_dict = torch.load(model_file, map_location="cpu", weights_only=False)

    # veRL saves with a specific format - extract the actual weights
    # The state dict might be nested or have prefixes
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Remove any "module." prefix from FSDP/DDP
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("_fsdp_wrapped_module."):
            new_key = new_key[21:]
        cleaned_state_dict[new_key] = value

    # Save as safetensors (preferred) or pytorch
    print(f"  Saving to {hf_dir}...")
    try:
        from safetensors.torch import save_file
        save_file(cleaned_state_dict, safetensors_file)
        print(f"  Saved: model.safetensors")
    except ImportError:
        torch.save(cleaned_state_dict, pytorch_file)
        print(f"  Saved: pytorch_model.bin")

    print(f"[DONE] {ckpt_dir.name}")
    return True


def find_checkpoints(path: Path) -> list[Path]:
    """Find all checkpoint directories under a path.

    Handles three cases:
      1. path is a checkpoint dir itself (has actor/ or model_world_size_*.pt)
      2. path is a run dir (contains global_step_* dirs)
      3. path is a parent dir (contains multiple run dirs, each with global_step_*)
    """
    checkpoints = []

    # Case 1: this is a checkpoint dir itself
    if (path / "actor").exists() or list(path.glob("model_world_size_*.pt")):
        return [path]

    # Look for global_step_* directories directly
    step_dirs = sorted(
        [d for d in path.iterdir() if d.is_dir() and d.name.startswith("global_step_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )

    if step_dirs:
        # Case 2: this is a run dir
        return step_dirs

    # Case 3: parent dir containing multiple run dirs
    for run_dir in sorted(path.iterdir()):
        if not run_dir.is_dir():
            continue
        run_steps = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("global_step_")],
            key=lambda d: int(d.name.split("_")[-1]),
        )
        checkpoints.extend(run_steps)

    return checkpoints


OUTPUTS = "/mnt/data8tb/Documents/project/rlvr_winter/verl-my-rlvr/outputs"

# ============================================================
# Edit these to change what gets converted
# ============================================================
PATHS = [
    f"{OUTPUTS}/grpo_09-02_2155-first",
    f"{OUTPUTS}/grpo-single_09-02_1440-first",
    f"{OUTPUTS}/grpo-single-dsr_09-02_1759-first",
    f"{OUTPUTS}/sft_09-02_1146-first",
]


def main():
    all_checkpoints = []
    for p in PATHS:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] Path does not exist: {path}")
            continue
        all_checkpoints.extend(find_checkpoints(path))

    if not all_checkpoints:
        print("[ERROR] No checkpoints found")
        return 1

    print(f"[INFO] Found {len(all_checkpoints)} checkpoint(s) across {len(PATHS)} runs\n")

    success = 0
    for ckpt in all_checkpoints:
        if convert_single_checkpoint(ckpt):
            success += 1

    print(f"\n[SUMMARY] Converted {success}/{len(all_checkpoints)} checkpoints")
    return 0 if success == len(all_checkpoints) else 1


if __name__ == "__main__":
    exit(main())
