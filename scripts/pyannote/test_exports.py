#!/usr/bin/env python3
"""Pytest suite for validating ONNX exports and reference outputs.

Tests map to project requirements:
  - test_segmentation_export       -> PREP-02
  - test_embedding_export          -> PREP-03
  - test_numerical_consistency_segmentation -> PREP-04
  - test_numerical_consistency_embedding    -> PREP-04
  - test_reference_outputs_exist   -> PREP-05

Usage:
    cd scripts/pyannote
    HF_TOKEN=your_token python -m pytest test_exports.py -v
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

# Resolve paths relative to project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_DIR = MODELS_DIR / "speaker-diarization-community-1"
SEG_ONNX = PIPELINE_DIR / "segmentation" / "model.onnx"
EMB_ONNX = PIPELINE_DIR / "embedding" / "model.onnx"
METADATA_PATH = PIPELINE_DIR / "export_metadata.json"
REFERENCE_DIR = PROJECT_ROOT / "reference_outputs"
PREPROCESSED_DIR = REFERENCE_DIR / "preprocessed"

HAS_SEG_ONNX = SEG_ONNX.exists()
HAS_EMB_ONNX = EMB_ONNX.exists()
HAS_METADATA = METADATA_PATH.exists()
HAS_HF_TOKEN = bool(os.environ.get("HF_TOKEN"))
HAS_REFERENCE = REFERENCE_DIR.exists() and any(REFERENCE_DIR.glob("*_segments.json"))


@pytest.mark.skipif(not HAS_SEG_ONNX, reason="Segmentation ONNX not found")
def test_segmentation_export():
    """PREP-02: Segmentation ONNX exists and is valid."""
    import onnx
    import onnxruntime as ort

    # Check file exists and is non-trivial
    assert SEG_ONNX.stat().st_size > 1_000_000, "Segmentation ONNX file suspiciously small"

    # Validate with ONNX checker
    model = onnx.load(str(SEG_ONNX))
    onnx.checker.check_model(model)

    # Verify input/output names
    session = ort.InferenceSession(str(SEG_ONNX))
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    assert "input_values" in input_names, f"Expected input 'input_values', got {input_names}"
    assert "logits" in output_names, f"Expected output 'logits', got {output_names}"

    # Verify dynamic axes work: run with different input sizes
    for num_samples in [80000, 160000]:
        test_input = np.zeros((1, 1, num_samples), dtype=np.float32)
        outputs = session.run(None, {"input_values": test_input})
        assert outputs[0].ndim == 3, f"Expected 3D output, got {outputs[0].ndim}D"
        assert outputs[0].shape[2] == 7, f"Expected 7 classes, got {outputs[0].shape[2]}"


@pytest.mark.skipif(not HAS_EMB_ONNX, reason="Embedding ONNX not found")
@pytest.mark.skipif(not HAS_METADATA, reason="Export metadata not found")
def test_embedding_export():
    """PREP-03: Embedding ONNX exists and is valid."""
    import onnx
    import onnxruntime as ort

    assert EMB_ONNX.stat().st_size > 1_000_000, "Embedding ONNX file suspiciously small"

    model = onnx.load(str(EMB_ONNX))
    onnx.checker.check_model(model)

    # Read metadata for strategy
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    strategy = metadata["embedding"]["strategy"]

    session = ort.InferenceSession(str(EMB_ONNX))
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    if strategy == "B_fbank":
        assert "fbank" in input_names, f"Strategy B expects 'fbank' input, got {input_names}"
        # Test with fbank input
        test_input = np.random.randn(1, 200, 80).astype(np.float32)
        outputs = session.run(None, {"fbank": test_input})
    elif strategy == "A_waveform":
        assert "waveforms" in input_names, f"Strategy A expects 'waveforms' input, got {input_names}"
        test_input = np.random.randn(1, 1, 48000).astype(np.float32)
        outputs = session.run(None, {"waveforms": test_input})
    else:
        pytest.fail(f"Unknown embedding strategy: {strategy}")

    assert "embedding" in output_names, f"Expected output 'embedding', got {output_names}"
    assert outputs[0].shape[-1] == 256, f"Expected 256-dim embedding, got {outputs[0].shape[-1]}"


@pytest.mark.skipif(not HAS_SEG_ONNX, reason="Segmentation ONNX not found")
@pytest.mark.skipif(not HAS_HF_TOKEN, reason="HF_TOKEN not set")
def test_numerical_consistency_segmentation():
    """PREP-04 (segmentation): ONNX matches PyTorch within atol=1e-4."""
    import onnxruntime as ort
    import torch
    from pyannote.audio import Model

    token = os.environ["HF_TOKEN"]
    pt_model = Model.from_pretrained("pyannote/segmentation-3.0", token=token)
    pt_model.eval()

    session = ort.InferenceSession(str(SEG_ONNX))

    for num_samples in [80000, 160000, 320000]:
        np_input = np.random.randn(1, 1, num_samples).astype(np.float32)
        torch_input = torch.from_numpy(np_input)

        with torch.no_grad():
            pt_out = pt_model(torch_input).numpy()

        onnx_out = session.run(None, {"input_values": np_input})[0]

        max_diff = float(np.max(np.abs(pt_out - onnx_out)))
        assert pt_out.shape == onnx_out.shape, (
            f"Shape mismatch at {num_samples}: {pt_out.shape} vs {onnx_out.shape}"
        )
        assert np.allclose(pt_out, onnx_out, atol=1e-4), (
            f"Numerical divergence at {num_samples} samples: max diff {max_diff:.2e}"
        )
        # Verify log-softmax: all values <= 0
        assert float(np.max(onnx_out)) <= 0.0, (
            f"Output values > 0 detected (not log-softmax): max={float(np.max(onnx_out))}"
        )


@pytest.mark.skipif(not HAS_EMB_ONNX, reason="Embedding ONNX not found")
@pytest.mark.skipif(not HAS_METADATA, reason="Export metadata not found")
@pytest.mark.skipif(not HAS_HF_TOKEN, reason="HF_TOKEN not set")
def test_numerical_consistency_embedding():
    """PREP-04 (embedding): ONNX matches PyTorch within atol=1e-4."""
    import onnxruntime as ort
    import torch
    from pyannote.audio import Pipeline

    token = os.environ["HF_TOKEN"]
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    wespeaker_model = pipeline._embedding.model_
    wespeaker_model.eval()

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    strategy = metadata["embedding"]["strategy"]

    session = ort.InferenceSession(str(EMB_ONNX))

    for num_samples in [16000, 48000, 80000]:
        waveform = torch.randn(1, 1, num_samples)

        if strategy == "B_fbank":
            with torch.no_grad():
                fbank = wespeaker_model.compute_fbank(waveform)
                _, pt_emb = wespeaker_model.resnet(fbank)
                pt_emb = pt_emb.numpy()

            onnx_emb = session.run(None, {"fbank": fbank.numpy()})[0]
        elif strategy == "A_waveform":
            with torch.no_grad():
                pt_out = wespeaker_model(waveform)
                pt_emb = pt_out[1].numpy() if isinstance(pt_out, tuple) else pt_out.numpy()

            onnx_emb = session.run(None, {"waveforms": waveform.numpy()})[0]
        else:
            pytest.fail(f"Unknown strategy: {strategy}")

        max_diff = float(np.max(np.abs(pt_emb - onnx_emb)))
        assert pt_emb.shape == onnx_emb.shape, (
            f"Shape mismatch at {num_samples}: {pt_emb.shape} vs {onnx_emb.shape}"
        )
        assert np.allclose(pt_emb, onnx_emb, atol=1e-4), (
            f"Numerical divergence at {num_samples} samples: max diff {max_diff:.2e}"
        )
        assert onnx_emb.shape[-1] == 256, (
            f"Expected 256-dim embedding, got {onnx_emb.shape[-1]}"
        )


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference outputs not generated yet")
def test_reference_outputs_exist():
    """PREP-05: Reference outputs generated for at least one test file."""
    # Find at least one complete set of reference files
    segment_files = list(REFERENCE_DIR.glob("*_segments.json"))
    assert len(segment_files) > 0, "No segment JSON files found in reference_outputs/"

    for seg_json in segment_files:
        stem = seg_json.stem.replace("_segments", "")

        # Check all 4 files exist for this audio
        seg_raw = REFERENCE_DIR / f"{stem}_segmentation_raw.npz"
        binary = REFERENCE_DIR / f"{stem}_binary_activity.npz"
        embeddings = REFERENCE_DIR / f"{stem}_embeddings.npz"

        assert seg_raw.exists(), f"Missing {seg_raw}"
        assert binary.exists(), f"Missing {binary}"
        assert embeddings.exists(), f"Missing {embeddings}"

        # Verify segmentation raw NPZ
        with np.load(str(seg_raw)) as data:
            assert "windows" in data, "Missing 'windows' in segmentation_raw.npz"
            windows = data["windows"]
            assert windows.ndim == 3, f"Expected 3D windows, got {windows.ndim}D"
            assert windows.shape[2] == 7, f"Expected 7 classes, got {windows.shape[2]}"
            # Log-softmax: all values <= 0
            assert float(np.max(windows)) <= 0.0, (
                f"Segmentation values > 0 (not log-softmax): max={float(np.max(windows))}"
            )

        # Verify binary activity NPZ
        with np.load(str(binary)) as data:
            assert "activity" in data, "Missing 'activity' in binary_activity.npz"
            activity = data["activity"]
            assert activity.ndim == 2, f"Expected 2D activity, got {activity.ndim}D"
            assert activity.shape[1] == 3, f"Expected 3 speakers, got {activity.shape[1]}"
            assert activity.dtype == bool, f"Expected bool dtype, got {activity.dtype}"

        # Verify segments JSON
        with open(seg_json, encoding="utf-8") as f:
            seg_data = json.load(f)
        assert "audio_file" in seg_data, "Missing 'audio_file' in segments JSON"
        assert "segments" in seg_data, "Missing 'segments' in segments JSON"
        for seg in seg_data["segments"]:
            assert "start" in seg, "Missing 'start' in segment"
            assert "end" in seg, "Missing 'end' in segment"
            assert "speaker" in seg, "Missing 'speaker' in segment"
            assert isinstance(seg["start"], (int, float)), "start must be numeric"
            assert isinstance(seg["end"], (int, float)), "end must be numeric"

        # Verify embeddings NPZ
        with np.load(str(embeddings)) as data:
            assert "embeddings" in data, "Missing 'embeddings' in embeddings.npz"
            emb = data["embeddings"]
            if emb.shape[0] > 0:
                assert emb.shape[1] == 256, f"Expected 256-dim, got {emb.shape[1]}"
            assert "segment_starts" in data, "Missing 'segment_starts'"
            assert "segment_ends" in data, "Missing 'segment_ends'"

        # Only need to verify one complete set
        break
