#!/usr/bin/env python3
"""Download pyannote speaker-diarization-community-1 models from HuggingFace.

Usage:
    export HF_TOKEN=your_huggingface_token
    python download_models.py [--model-dir models/]
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def get_hf_token() -> str:
    """Get HuggingFace token from environment, fail with clear error if not set."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN environment variable is not set.\n"
            "\n"
            "To get a token:\n"
            "  1. Go to https://huggingface.co/settings/tokens\n"
            "  2. Create a token with 'read' access\n"
            "  3. Accept the model license at:\n"
            "     https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  4. Export the token: export HF_TOKEN=hf_...\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def print_dir_summary(directory: Path, indent: int = 0) -> int:
    """Print files in directory with sizes, return total size in bytes."""
    total_size = 0
    prefix = "  " * indent
    for item in sorted(directory.iterdir()):
        if item.is_dir():
            print(f"{prefix}{item.name}/")
            total_size += print_dir_summary(item, indent + 1)
        else:
            size = item.stat().st_size
            total_size += size
            if size >= 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size >= 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{prefix}{item.name}  ({size_str})")
    return total_size


def verify_model_files(model_dir: Path) -> bool:
    """Verify that critical model weight files exist after download."""
    # The community pipeline references segmentation and embedding sub-models.
    # Check for key directories and any weight files within them.
    issues = []

    # Check for segmentation model files
    seg_dir = model_dir / "segmentation"
    if not seg_dir.exists():
        issues.append(f"Missing directory: {seg_dir}")
    else:
        weight_files = list(seg_dir.glob("*.bin")) + list(seg_dir.glob("*.pt")) + list(seg_dir.glob("*.safetensors"))
        if not weight_files:
            issues.append(f"No weight files (*.bin, *.pt, *.safetensors) in {seg_dir}")

    # Check for embedding model files
    emb_dir = model_dir / "embedding"
    if not emb_dir.exists():
        issues.append(f"Missing directory: {emb_dir}")
    else:
        weight_files = list(emb_dir.glob("*.bin")) + list(emb_dir.glob("*.pt")) + list(emb_dir.glob("*.safetensors"))
        if not weight_files:
            issues.append(f"No weight files (*.bin, *.pt, *.safetensors) in {emb_dir}")

    if issues:
        print("\nWARNING: Some expected model files were not found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nThe pipeline may still work if models are referenced differently.")
        print("Check the downloaded directory structure above.\n")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download pyannote speaker-diarization-community-1 from HuggingFace"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Base directory for model storage (default: models/)",
    )
    args = parser.parse_args()

    token = get_hf_token()
    model_base = Path(args.model_dir)
    repo_id = "pyannote/speaker-diarization-community-1"
    local_dir = model_base / "speaker-diarization-community-1"

    print(f"Downloading {repo_id} ...")
    print(f"  Destination: {local_dir.resolve()}")
    print(f"  Token: {token[:8]}...{token[-4:]}\n")

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        token=token,
    )

    print(f"\nDownload complete. Contents of {local_dir}:\n")
    total = print_dir_summary(local_dir)
    print(f"\nTotal size: {total / 1_000_000:.1f} MB")

    verify_model_files(local_dir)
    print("Done.")


if __name__ == "__main__":
    main()
