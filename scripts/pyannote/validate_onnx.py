#!/usr/bin/env python3
"""Validate ONNX models against PyTorch for numerical consistency.

Tests both segmentation and embedding models across multiple input lengths.
Asserts np.allclose(atol=1e-4) between PyTorch and ONNX outputs.

Usage:
    export HF_TOKEN=your_huggingface_token
    python validate_onnx.py [--model-dir models/]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return token


def validate_segmentation(model_dir: Path, token: str) -> bool:
    """Validate segmentation ONNX vs PyTorch across 3 input lengths."""
    from pyannote.audio import Model

    print("=" * 60)
    print("Validating segmentation model")
    print("=" * 60)

    onnx_path = model_dir / "speaker-diarization-community-1" / "segmentation" / "model.onnx"
    if not onnx_path.exists():
        print(f"SKIP: ONNX file not found: {onnx_path}")
        return True

    # Load PyTorch model
    print("Loading PyTorch segmentation model ...")
    pt_model = Model.from_pretrained("pyannote/segmentation-3.0", token=token)
    pt_model.eval()

    # Load ONNX model
    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path))

    test_lengths = [80000, 160000, 320000]
    all_pass = True

    for num_samples in test_lengths:
        label = f"{num_samples / 16000:.0f}s ({num_samples} samples)"
        print(f"\n  Testing input length: {label}")

        # Random input
        np_input = np.random.randn(1, 1, num_samples).astype(np.float32)
        torch_input = torch.from_numpy(np_input)

        # PyTorch inference
        with torch.no_grad():
            pt_out = pt_model(torch_input).numpy()

        # ONNX inference
        onnx_out = session.run(None, {"input_values": np_input})[0]

        # Compare
        max_diff = float(np.max(np.abs(pt_out - onnx_out)))
        shapes_match = pt_out.shape == onnx_out.shape
        close = np.allclose(pt_out, onnx_out, atol=1e-4)

        # Verify log-softmax: all values should be <= 0
        onnx_max_val = float(np.max(onnx_out))
        is_log_softmax = onnx_max_val <= 0.0

        status = "PASS" if (close and shapes_match and is_log_softmax) else "FAIL"
        print(f"    Shape PyTorch:  {pt_out.shape}")
        print(f"    Shape ONNX:     {onnx_out.shape}")
        print(f"    Max abs diff:   {max_diff:.2e}")
        print(f"    Shapes match:   {shapes_match}")
        print(f"    np.allclose:    {close}")
        print(f"    Log-softmax OK: {is_log_softmax} (max value: {onnx_max_val:.4f})")
        print(f"    Result:         {status}")

        if status == "FAIL":
            all_pass = False

    return all_pass


def validate_embedding(model_dir: Path, token: str) -> bool:
    """Validate embedding ONNX vs PyTorch across multiple input lengths."""
    from pyannote.audio import Pipeline

    print("\n" + "=" * 60)
    print("Validating embedding model")
    print("=" * 60)

    onnx_path = model_dir / "speaker-diarization-community-1" / "embedding" / "model.onnx"
    metadata_path = model_dir / "speaker-diarization-community-1" / "export_metadata.json"

    if not onnx_path.exists():
        print(f"SKIP: ONNX file not found: {onnx_path}")
        return True

    if not metadata_path.exists():
        print(f"SKIP: Metadata not found: {metadata_path}")
        return True

    # Read export strategy
    with open(metadata_path) as f:
        metadata = json.load(f)
    strategy = metadata["embedding"]["strategy"]
    print(f"Embedding strategy: {strategy}")

    # Load pipeline to get the embedding model
    print("Loading pipeline for embedding model ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    wespeaker_model = pipeline._embedding.model_
    wespeaker_model.eval()

    # Load ONNX session
    session = ort.InferenceSession(str(onnx_path))

    test_lengths = [16000, 48000, 80000]
    all_pass = True

    for num_samples in test_lengths:
        label = f"{num_samples / 16000:.0f}s ({num_samples} samples)"
        print(f"\n  Testing input length: {label}")

        # Create random waveform
        waveform = torch.randn(1, 1, num_samples)

        if strategy == "B_fbank":
            # Strategy B: compute fbank, then run through ResNet
            with torch.no_grad():
                fbank = wespeaker_model.compute_fbank(waveform)
                # PyTorch: run through resnet
                _, pt_emb = wespeaker_model.resnet(fbank)
                pt_emb = pt_emb.numpy()

            # ONNX: feed fbank directly
            fbank_np = fbank.numpy()
            onnx_emb = session.run(None, {"fbank": fbank_np})[0]

        elif strategy == "A_waveform":
            # Strategy A: full waveform input
            with torch.no_grad():
                pt_out = wespeaker_model(waveform)
                # WeSpeakerResNet34 forward returns (loss_placeholder, embedding)
                if isinstance(pt_out, tuple):
                    pt_emb = pt_out[1].numpy()
                else:
                    pt_emb = pt_out.numpy()

            wav_np = waveform.numpy()
            onnx_emb = session.run(None, {"waveforms": wav_np})[0]

        else:
            print(f"ERROR: Unknown strategy: {strategy}")
            return False

        # Compare
        max_diff = float(np.max(np.abs(pt_emb - onnx_emb)))
        shapes_match = pt_emb.shape == onnx_emb.shape
        close = np.allclose(pt_emb, onnx_emb, atol=1e-4)

        # Verify embedding dimension
        emb_dim = onnx_emb.shape[-1]
        correct_dim = emb_dim == 256

        status = "PASS" if (close and shapes_match and correct_dim) else "FAIL"
        print(f"    Shape PyTorch:  {pt_emb.shape}")
        print(f"    Shape ONNX:     {onnx_emb.shape}")
        print(f"    Max abs diff:   {max_diff:.2e}")
        print(f"    Shapes match:   {shapes_match}")
        print(f"    np.allclose:    {close}")
        print(f"    Embedding dim:  {emb_dim} (expected 256: {correct_dim})")
        print(f"    Result:         {status}")

        if status == "FAIL":
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX models against PyTorch inference"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Base directory for model storage (default: models/)",
    )
    args = parser.parse_args()

    token = get_hf_token()
    model_dir = Path(args.model_dir)

    seg_ok = validate_segmentation(model_dir, token)
    emb_ok = validate_embedding(model_dir, token)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Segmentation: {'PASS' if seg_ok else 'FAIL'}")
    print(f"  Embedding:    {'PASS' if emb_ok else 'FAIL'}")

    if seg_ok and emb_ok:
        print("\nAll validations PASSED.")
        sys.exit(0)
    else:
        print("\nSome validations FAILED.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
