#!/usr/bin/env python3
"""Generate three-layer reference outputs for test audio files.

Produces per-audio reference data for Phase 2-3 Rust integration tests:
  Layer 1: Raw segmentation tensors per sliding window (.npz)
  Layer 2: Binary speaker activity after powerset decode (.npz)
  Layer 3: Final diarization segments (.json)
  Embeddings: Per-segment speaker embeddings (.npz)

Usage:
    export HF_TOKEN=your_huggingface_token
    python generate_reference.py [--audio-dir ...] [--output-dir ...] [--model-dir ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return token


def load_pipeline(token: str):
    """Load the full pyannote pipeline."""
    from pyannote.audio import Pipeline

    print("Loading pyannote pipeline ...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    return pipeline


def generate_layer1_segmentation(audio_path: Path, output_dir: Path, pipeline) -> None:
    """Layer 1: Raw segmentation tensors per sliding window.

    Uses Inference with skip_conversion=True to get the raw 7-class
    log-softmax output before powerset decoding.
    """
    from pyannote.audio import Inference

    stem = audio_path.stem
    print(f"  Layer 1: Raw segmentation tensors ...")

    seg_model = pipeline._segmentation.model

    # skip_conversion=True gives raw 7-class log-softmax (before powerset decode)
    # window="sliding" returns per-window results: (num_windows, frames_per_window, 7)
    inference = Inference(
        seg_model,
        window="sliding",
        duration=10.0,
        step=2.5,
        skip_conversion=True,
    )

    output = inference(audio_path)
    windows_array = output.data  # (num_windows, frames_per_window, 7)

    # Read audio to get sample count
    info = sf.info(str(audio_path))
    num_samples = int(info.frames)
    sample_rate = int(info.samplerate)

    out_path = output_dir / f"{stem}_segmentation_raw.npz"
    np.savez(
        str(out_path),
        windows=windows_array,
        window_duration=np.float64(10.0),
        step_duration=np.float64(2.5),
        sample_rate=np.int32(sample_rate),
        num_samples=np.int64(num_samples),
    )
    print(f"    Saved: {out_path}")
    print(f"    Shape: {windows_array.shape} (windows, frames, classes)")


def generate_layer2_binary_activity(audio_path: Path, output_dir: Path, pipeline) -> None:
    """Layer 2: Binary speaker activity after powerset decode.

    Uses default Inference (with powerset conversion), which returns decoded
    per-speaker probability scores (num_windows, frames_per_window, 3) in [0, 1].
    Binarises with threshold > 0.5 and stitches windows into a flat timeline
    using the centre-crop overlap-add region (half the step on each side).
    """
    from pyannote.audio import Inference

    stem = audio_path.stem
    print(f"  Layer 2: Binary speaker activity ...")

    seg_model = pipeline._segmentation.model

    # Default Inference applies powerset decoding; output is (num_windows, frames, 3)
    # with values in [0, 1] (per-speaker probabilities).
    inference = Inference(
        seg_model,
        window="sliding",
        duration=10.0,
        step=2.5,
    )
    segmentation = inference(audio_path)

    # segmentation.data: (num_windows, frames_per_window, 3)
    # segmentation.sliding_window.step: 2.5 s (window step, not frame step)
    per_window = segmentation.data           # (W, F, 3)
    W, F, S = per_window.shape               # windows, frames, speakers

    # Frame duration (in seconds) for one output frame
    win_dur = segmentation.sliding_window.duration   # 10.0 s
    frame_step_s = win_dur / F                        # ~0.016949 s per frame

    info = sf.info(str(audio_path))
    num_samples = int(info.frames)
    sample_rate = int(info.samplerate)

    # Total number of output frames that cover the full audio
    total_frames = int(np.ceil(num_samples / sample_rate / frame_step_s))

    # Centre-crop overlap-add: keep only the central step-width from each window
    # (discard warm-up/cool-down frames at each edge).
    step_s = segmentation.sliding_window.step        # 2.5 s
    margin_s = (win_dur - step_s) / 2.0              # 3.75 s
    margin_frames = int(round(margin_s / frame_step_s))
    keep_frames = int(round(step_s / frame_step_s))

    activity = np.zeros((total_frames, S), dtype=bool)

    for w in range(W):
        # Source slice within the window (centre crop)
        src_start = margin_frames
        src_end = margin_frames + keep_frames
        src_end = min(src_end, F)

        # Destination slice in the output timeline
        dst_start = w * keep_frames
        dst_end = dst_start + (src_end - src_start)
        dst_end = min(dst_end, total_frames)

        n = dst_end - dst_start
        if n <= 0:
            continue

        probs = per_window[w, src_start : src_start + n, :]  # (n, 3)
        activity[dst_start:dst_end, :] = probs > 0.5

    # Frame duration in ms for Rust consumers
    frame_duration_ms = frame_step_s * 1000.0

    out_path = output_dir / f"{stem}_binary_activity.npz"
    np.savez(
        str(out_path),
        activity=activity,
        frame_duration_ms=np.float64(frame_duration_ms),
    )
    print(f"    Saved: {out_path}")
    print(f"    Shape: {activity.shape} (frames, speakers)")
    print(f"    Frame duration: {frame_duration_ms:.3f} ms")


def generate_layer3_and_embeddings(
    audio_path: Path, output_dir: Path, pipeline
) -> None:
    """Layer 3 + Embeddings: run pipeline once, save segments JSON and embeddings NPZ.

    pyannote/speaker-diarization-community-1 returns a DiarizeOutput with:
      .speaker_diarization  -- pyannote Annotation (itertracks)
      .speaker_embeddings   -- np.ndarray (num_unique_speakers, 256), rows in
                               sorted(speakers) order
    """
    stem = audio_path.stem
    print(f"  Layer 3 + Embeddings: running full pipeline ...")

    # Run full pipeline once
    result = pipeline(str(audio_path))
    diarization = result.speaker_diarization      # pyannote Annotation
    spk_embeddings = result.speaker_embeddings    # (num_speakers, 256)

    # --- Layer 3: segments JSON ---
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(float(turn.start), 6),
            "end": round(float(turn.end), 6),
            "speaker": speaker,
        })

    seg_result = {
        "audio_file": audio_path.name,
        "segments": segments,
    }

    json_path = output_dir / f"{stem}_segments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(seg_result, f, indent=2, ensure_ascii=False)

    print(f"    Saved (L3): {json_path}")
    print(f"    Segments: {len(segments)}")
    if segments:
        print(f"    First: {segments[0]}")
        print(f"    Last:  {segments[-1]}")

    # --- Embeddings NPZ ---
    # Build per-segment embedding array: repeat each speaker's centroid embedding
    # for each turn that speaker appears in.
    unique_speakers = sorted(diarization.labels())
    spk_to_idx = {spk: i for i, spk in enumerate(unique_speakers)}

    emb_list: list[np.ndarray] = []
    starts_list: list[float] = []
    ends_list: list[float] = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        idx = spk_to_idx[speaker]
        emb_list.append(spk_embeddings[idx].astype(np.float32))
        starts_list.append(float(turn.start))
        ends_list.append(float(turn.end))

    if emb_list:
        embeddings_array = np.stack(emb_list)          # (num_segments, 256)
    else:
        embeddings_array = np.zeros((0, 256), dtype=np.float32)

    npz_path = output_dir / f"{stem}_embeddings.npz"
    np.savez(
        str(npz_path),
        embeddings=embeddings_array.astype(np.float32),
        segment_starts=np.array(starts_list, dtype=np.float64),
        segment_ends=np.array(ends_list, dtype=np.float64),
    )
    print(f"    Saved (Emb): {npz_path}")
    print(f"    Shape: {embeddings_array.shape} (segments, 256)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate three-layer reference outputs for test audio"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="reference_outputs/preprocessed/",
        help="Directory containing preprocessed WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reference_outputs/",
        help="Directory to save reference outputs",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/",
        help="Base directory for model storage",
    )
    args = parser.parse_args()

    token = get_hf_token()
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline once
    pipeline = load_pipeline(token)

    # Find all WAV files
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No WAV files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(wav_files)} WAV files to process.\n")

    for i, wav_path in enumerate(wav_files, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(wav_files)}] Processing: {wav_path.name}")
        print(f"{'=' * 60}")

        # Layer 1: Raw segmentation
        generate_layer1_segmentation(wav_path, output_dir, pipeline)

        # Layer 2: Binary activity
        generate_layer2_binary_activity(wav_path, output_dir, pipeline)

        # Layer 3 + Embeddings (single pipeline call)
        generate_layer3_and_embeddings(wav_path, output_dir, pipeline)

    print(f"\n{'=' * 60}")
    print(f"Reference generation complete!")
    print(f"  Audio files processed: {len(wav_files)}")
    print(f"  Output directory:      {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
