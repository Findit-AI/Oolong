#!/usr/bin/env python3
"""Export pyannote segmentation and embedding models to ONNX format.

Usage:
    export HF_TOKEN=your_huggingface_token
    python export_onnx.py [--model-dir models/] [--segmentation-only] [--embedding-only]

Requires: pyannote.audio, torch, onnx, onnxruntime
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

OPSET_VERSION = 17
# Use legacy TorchScript exporter; dynamo exporter chokes on SincNet + LSTM.
_LEGACY_EXPORT_KWARGS = {"dynamo": False}
REPO_ID_PIPELINE = "pyannote/speaker-diarization-community-1"
REPO_ID_SEGMENTATION = "pyannote/segmentation-3.0"


def get_hf_token() -> str:
    """Get HuggingFace token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN environment variable is not set.\n"
            "See download_models.py for instructions.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def export_segmentation(model_dir: Path, token: str) -> dict:
    """Export the segmentation model to ONNX.

    Returns metadata about the export.
    """
    from pyannote.audio import Model

    output_dir = model_dir / "speaker-diarization-community-1" / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"

    print("=" * 60)
    print("Exporting segmentation model")
    print("=" * 60)

    # Load the segmentation model
    print(f"Loading {REPO_ID_SEGMENTATION} ...")
    model = Model.from_pretrained(
        REPO_ID_SEGMENTATION,
        token=token,
    )
    model.eval()

    # Use batch=2 to ensure dynamic batch tracing works correctly
    dummy_input = torch.zeros(2, 1, 160000)

    print(f"Exporting to ONNX (opset {OPSET_VERSION}) ...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {
                0: "batch_size",
                1: "num_channels",
                2: "num_samples",
            },
            "logits": {
                0: "batch_size",
                1: "num_frames",
            },
        },
        **_LEGACY_EXPORT_KWARGS,
    )

    # Validate ONNX model
    print("Validating ONNX model ...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX validation passed.")

    # Run quick inference to get actual output shape
    print("Running test inference ...")
    session = ort.InferenceSession(str(output_path))
    test_input = np.zeros((1, 1, 160000), dtype=np.float32)
    outputs = session.run(None, {"input_values": test_input})
    logits = outputs[0]
    num_frames = logits.shape[1]
    num_classes = logits.shape[2]
    print(f"  Input shape:  (1, 1, 160000)")
    print(f"  Output shape: {logits.shape}")
    print(f"  Num frames:   {num_frames}")
    print(f"  Num classes:  {num_classes}")

    file_size = output_path.stat().st_size
    print(f"  File size:    {file_size / 1_000_000:.1f} MB")
    print(f"  Saved to:     {output_path}")
    print()

    return {
        "output_path": str(output_path),
        "num_frames_for_160000_samples": num_frames,
        "num_classes": num_classes,
        "file_size_bytes": file_size,
    }


def export_embedding(model_dir: Path, token: str) -> dict:
    """Export the embedding model to ONNX.

    Tries Strategy A (full waveform wrapper) first, falls back to Strategy B (fbank input).
    Returns metadata about the export including which strategy was used.

    Strategy A exports the full WeSpeakerResNet34 model (waveform -> embedding) but
    typically fails because torchaudio's fbank uses torch.vmap which can't be traced.
    Strategy B exports just the ResNet backbone (fbank -> embedding), meaning Phase 2
    Rust code must compute 80-dim fbank features externally.
    """
    from pyannote.audio import Pipeline

    output_dir = model_dir / "speaker-diarization-community-1" / "embedding"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"

    print("=" * 60)
    print("Exporting embedding model")
    print("=" * 60)

    # Load the full pipeline to access the embedding model
    print(f"Loading pipeline {REPO_ID_PIPELINE} ...")
    pipeline = Pipeline.from_pretrained(
        REPO_ID_PIPELINE,
        token=token,
    )

    embedding_wrapper = pipeline._embedding
    wespeaker_model = embedding_wrapper.model_
    wespeaker_model.eval()
    print(f"Embedding wrapper type: {type(embedding_wrapper).__name__}")
    print(f"Inner model type: {type(wespeaker_model).__name__}")

    # ---- Strategy A: Export full model (waveform -> embedding) ----
    strategy = None
    strategy_error = None

    try:
        print("\n--- Strategy A: Export full waveform model ---")
        dummy_input = torch.zeros(1, 1, 48000)  # 3 seconds at 16kHz

        torch.onnx.export(
            wespeaker_model,
            dummy_input,
            str(output_path),
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["waveforms"],
            output_names=["embedding"],
            dynamic_axes={
                "waveforms": {0: "batch_size", 2: "num_samples"},
                "embedding": {0: "batch_size"},
            },
            **_LEGACY_EXPORT_KWARGS,
        )

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(str(output_path))
        test_input = np.random.randn(1, 1, 48000).astype(np.float32)
        emb = session.run(None, {"waveforms": test_input})[0]

        if np.allclose(emb, 0.0):
            raise ValueError("Strategy A produced all-zero embeddings.")

        strategy = "A_waveform"
        print(f"Strategy A succeeded!")
        print(f"  Input:  waveforms (batch, 1, num_samples)")
        print(f"  Output: embedding {emb.shape}")

    except Exception as e:
        strategy_error = str(e)
        print(f"\nStrategy A failed: {e}")
        print("Falling back to Strategy B ...\n")

    # ---- Strategy B: Export ResNet backbone with fbank input ----
    if strategy is None:
        print("--- Strategy B: Export ResNet backbone (fbank input) ---")

        resnet = wespeaker_model.resnet

        # Wrapper to discard the loss placeholder (index 0) from ResNet output
        class _EmbeddingOnly(torch.nn.Module):
            def __init__(self, resnet):
                super().__init__()
                self.resnet = resnet

            def forward(self, fbank):
                _, embedding = self.resnet(fbank)
                return embedding

        wrapper = _EmbeddingOnly(resnet)
        wrapper.eval()

        # fbank input: (batch, num_frames, 80-dim fbank features)
        dummy_fbank = torch.zeros(1, 200, 80)

        torch.onnx.export(
            wrapper,
            (dummy_fbank,),
            str(output_path),
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["fbank"],
            output_names=["embedding"],
            dynamic_axes={
                "fbank": {0: "batch_size", 1: "num_frames"},
                "embedding": {0: "batch_size"},
            },
            **_LEGACY_EXPORT_KWARGS,
        )

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Validate with real fbank from the model's own compute_fbank
        test_wav = torch.randn(1, 1, 48000)
        with torch.no_grad():
            test_fbank = wespeaker_model.compute_fbank(test_wav)
            pt_emb = wrapper(test_fbank).numpy()

        session = ort.InferenceSession(str(output_path))
        onnx_emb = session.run(None, {"fbank": test_fbank.numpy()})[0]
        emb = onnx_emb

        max_diff = float(np.max(np.abs(pt_emb - onnx_emb)))
        print(f"  Max diff vs PyTorch: {max_diff:.2e}")
        assert np.allclose(pt_emb, onnx_emb, atol=1e-4), (
            f"Embedding ONNX diverges from PyTorch: max diff {max_diff}"
        )

        strategy = "B_fbank"
        print(f"Strategy B succeeded!")
        print(f"  Input:  fbank (batch, num_frames, 80)")
        print(f"  Output: embedding {emb.shape}")
        print()
        print(
            "NOTE: Phase 2 Rust code MUST compute 80-dim fbank features "
            "(scale waveform by 32768, then torchaudio-compatible fbank) "
            "before feeding to this model."
        )

    file_size = output_path.stat().st_size
    embedding_dim = int(emb.shape[-1]) if emb is not None else None
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  File size:     {file_size / 1_000_000:.1f} MB")
    print(f"  Saved to:      {output_path}")
    print()

    result = {
        "output_path": str(output_path),
        "strategy": strategy,
        "embedding_dim": embedding_dim,
        "file_size_bytes": file_size,
    }
    if strategy_error:
        result["strategy_a_error"] = strategy_error

    return result


def write_metadata(model_dir: Path, seg_meta: dict, emb_meta: dict) -> None:
    """Write export metadata JSON."""
    metadata_path = model_dir / "speaker-diarization-community-1" / "export_metadata.json"

    metadata = {
        "export_date": datetime.now(timezone.utc).isoformat(),
        "opset_version": OPSET_VERSION,
        "segmentation": {
            "output_frames_for_160000_samples": seg_meta.get("num_frames_for_160000_samples"),
            "num_classes": seg_meta.get("num_classes"),
            "file_size_bytes": seg_meta.get("file_size_bytes"),
        },
        "embedding": {
            "strategy": emb_meta.get("strategy"),
            "embedding_dim": emb_meta.get("embedding_dim"),
            "file_size_bytes": emb_meta.get("file_size_bytes"),
        },
    }

    if "strategy_a_error" in emb_meta:
        metadata["embedding"]["strategy_a_error"] = emb_meta["strategy_a_error"]

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata written to: {metadata_path}")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Export pyannote segmentation and embedding models to ONNX"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Base directory for model storage (default: models/)",
    )
    parser.add_argument(
        "--segmentation-only",
        action="store_true",
        help="Only export the segmentation model",
    )
    parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Only export the embedding model",
    )
    args = parser.parse_args()

    token = get_hf_token()
    model_dir = Path(args.model_dir)

    seg_meta = {}
    emb_meta = {}

    if not args.embedding_only:
        seg_meta = export_segmentation(model_dir, token)

    if not args.segmentation_only:
        emb_meta = export_embedding(model_dir, token)

    # Write metadata if both models were exported (or at least one)
    if seg_meta or emb_meta:
        write_metadata(model_dir, seg_meta, emb_meta)

    print("\n" + "=" * 60)
    print("Export complete!")
    if seg_meta:
        print(f"  Segmentation: {seg_meta.get('output_path')}")
    if emb_meta:
        print(f"  Embedding:    {emb_meta.get('output_path')} (Strategy {emb_meta.get('strategy', '?')})")
    print("=" * 60)


if __name__ == "__main__":
    main()
