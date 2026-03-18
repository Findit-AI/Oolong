# pyannote Scripts: ONNX Validation & Reference Output Generation

## Environment Setup

```bash
# Create virtual environment (Python 3.10+)
python3 -m venv .venv-pyannote
source .venv-pyannote/bin/activate
pip install -r scripts/pyannote/requirements.txt

# Set HuggingFace token (required for model access)
export HF_TOKEN=your_token_here
```

## Scripts

### 1. download_models.py

Downloads the pyannote pipeline from HuggingFace Hub.

```bash
python scripts/pyannote/download_models.py
```

### 2. export_onnx.py

Exports PyTorch segmentation and embedding models to ONNX format.

```bash
python scripts/pyannote/export_onnx.py [--model-dir models/]
```

### 3. validate_onnx.py

Validates ONNX models against PyTorch inference. Tests numerical consistency with `np.allclose(atol=1e-4)` across multiple input lengths.

```bash
python scripts/pyannote/validate_onnx.py [--model-dir models/]
```

### 4. generate_reference.py

Generates three-layer reference outputs for all preprocessed test audio files.

```bash
python scripts/pyannote/generate_reference.py \
    [--audio-dir reference_outputs/preprocessed/] \
    [--output-dir reference_outputs/] \
    [--model-dir models/]
```

### 5. preprocess_audio.sh

Converts test audio files to 16kHz mono WAV format.

```bash
bash scripts/pyannote/preprocess_audio.sh
```

### 6. test_exports.py

Pytest suite validating exports and reference outputs.

```bash
cd scripts/pyannote && python -m pytest test_exports.py -v
```

## Reference Output Format

For each audio file `{stem}.wav` in `reference_outputs/preprocessed/`, four reference files are generated in `reference_outputs/`:

### File Naming Convention

```
{audio_stem}_segmentation_raw.npz   -- Layer 1
{audio_stem}_binary_activity.npz    -- Layer 2
{audio_stem}_segments.json          -- Layer 3
{audio_stem}_embeddings.npz         -- Embeddings
```

### Layer 1: Raw Segmentation Tensors (`_segmentation_raw.npz`)

Per-window log-softmax output from the segmentation model with a 10s sliding window and 2.5s step.

| Array Key         | Shape                              | Dtype   | Description                             |
|-------------------|------------------------------------|---------|-----------------------------------------|
| `windows`         | `(num_windows, frames_per_window, 7)` | float32 | Raw log-softmax output per window       |
| `window_duration` | scalar                             | float64 | Window size in seconds (10.0)           |
| `step_duration`   | scalar                             | float64 | Step size in seconds (2.5)              |
| `sample_rate`     | scalar                             | int32   | Audio sample rate (16000)               |
| `num_samples`     | scalar                             | int64   | Total audio length in samples           |

**CRITICAL**: Output values are **log-softmax** (all values <= 0), NOT logits. Use `exp()` to get probabilities, or `argmax` directly for class selection.

**Powerset class mapping** (7 classes, 3 max speakers):

| Index | Active Speakers | Description           |
|-------|----------------|-----------------------|
| 0     | `{}`           | No speaker (silence)  |
| 1     | `{0}`          | Speaker 0 only        |
| 2     | `{1}`          | Speaker 1 only        |
| 3     | `{2}`          | Speaker 2 only        |
| 4     | `{0, 1}`       | Speakers 0 and 1      |
| 5     | `{0, 2}`       | Speakers 0 and 2      |
| 6     | `{1, 2}`       | Speakers 1 and 2      |

Ordering follows `itertools.combinations` (NOT bitmask encoding).

### Layer 2: Binary Speaker Activity (`_binary_activity.npz`)

Frame-level binary voice activity per speaker, decoded from powerset classes.

| Array Key          | Shape               | Dtype   | Description                        |
|--------------------|---------------------|---------|------------------------------------|
| `activity`         | `(total_frames, 3)` | bool    | Per-speaker binary activity        |
| `frame_duration_ms`| scalar              | float64 | Milliseconds per frame (~16.875)   |

### Layer 3: Final Diarization Segments (`_segments.json`)

Full pipeline output with speaker labels.

```json
{
  "audio_file": "filename.wav",
  "segments": [
    {"start": 0.5, "end": 2.3, "speaker": "SPEAKER_00"},
    {"start": 2.5, "end": 5.1, "speaker": "SPEAKER_01"}
  ]
}
```

- `start` / `end`: float64, seconds from audio start
- `speaker`: string, format `SPEAKER_XX`
- No extra fields (no confidence, no overlap flags)

### Embeddings (`_embeddings.npz`)

Per-segment 256-dimensional speaker embeddings.

| Array Key        | Shape               | Dtype   | Description                      |
|------------------|---------------------|---------|----------------------------------|
| `embeddings`     | `(num_segments, 256)` | float32 | Speaker embedding vectors        |
| `segment_starts` | `(num_segments,)`   | float64 | Segment start times in seconds   |
| `segment_ends`   | `(num_segments,)`   | float64 | Segment end times in seconds     |

## Embedding Export Strategy

The embedding model was exported with **Strategy B (fbank input)**. See `models/speaker-diarization-community-1/export_metadata.json` for details.

- **Strategy A (waveform)**: Full model takes raw waveform. Failed during export due to torchaudio fbank tracing limitations.
- **Strategy B (fbank)**: Inner ResNet takes pre-computed fbank features `(batch, num_frames, 80)`. Phase 2 Rust code must compute 80-dim fbank features before calling this model.

## Loading Reference Data in Rust

### NPZ files (using `ndarray-npy`)

```rust
use ndarray::Array2;
use ndarray_npy::NpzReader;
use std::fs::File;

let mut npz = NpzReader::new(File::open("reference_outputs/audio_segmentation_raw.npz")?)?;
let windows: Array3<f32> = npz.by_name("windows")?;
let window_duration: f64 = npz.by_name::<Array0<f64>>("window_duration")?.into_scalar();
```

### JSON files (using `serde_json`)

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct DiarizationOutput {
    audio_file: String,
    segments: Vec<Segment>,
}

#[derive(Deserialize)]
struct Segment {
    start: f64,
    end: f64,
    speaker: String,
}

let data: DiarizationOutput = serde_json::from_reader(File::open("ref.json")?)?;
```

### Frame Index Arithmetic

Segmentation produces ~270.05 samples per frame (non-integer). Always use `f64` for timestamp calculations:

```rust
let frame_duration_seconds = step_duration / frames_per_window as f64;
let timestamp = frame_index as f64 * frame_duration_seconds;
```
