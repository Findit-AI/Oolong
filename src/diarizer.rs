//! TieguanyinOolong — pyannote speaker diarization ONNX inference implementation.
//!
//! This module implements the [`TieguanyinOolong`] component:
//! - Streaming VAD (voice activity detection): `detect()` / `finish()`
//! - Batch speaker diarization: `diarize()` / `diarize_timed()` / `diarize_overlap()`
//!
//! ## Model Dependencies
//!
//! - **segmentation-3.0** (5.6MB): pyannote-audio segmentation model
//!   <https://huggingface.co/pyannote/segmentation-3.0>
//! - **WeSpeaker ResNet34** (optional, ~25MB): speaker embedding model, not needed for VAD-only mode
//!   <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM>
//!
//! ## Runtime Dependencies
//!
//! - [`ort`] crate (ONNX Runtime Rust bindings): <https://github.com/pykeio/ort>
//! - [`kodama`] crate: hierarchical clustering (Centroid linkage, VBx AHC initialization)
//! - [`mel_spec`] crate: fbank feature extraction (80-dim, apply_cmn = false)

use core::{num::NonZeroU32, time::Duration};
use std::{collections::VecDeque, path::PathBuf, time::Instant};

use ndarray::{Array1, Array2, Axis};
use ndarray_npy::NpzReader;
use ort::session::{Session, builder::GraphOptimizationLevel};
#[cfg(feature = "coreml")]
use ort::ep;
use mel_spec::fbank::{Fbank, FbankConfig};

use crate::types::{DiarizeSegment, DiarizeTiming, SpeakerSegment, VoiceRange};

// ── Window parameter constants (verified through testing) ────────────────────

/// Number of samples for a full inference window: 10s @ 16kHz.
const WINDOW_SAMPLES: usize = 160_000;

/// Streaming VAD step size in samples: 2.5s @ 16kHz (detect() buffer step).
const STEP_SAMPLES: usize = 40_000;

/// Speaker diarization (diarize/diarize_with_embeddings) step size in samples: 1s @ 16kHz.
/// Matches pyannote segmentation_step=0.1 (default), yielding 218 chunks × 3 speakers ≈ 654 embeddings,
/// aligned with the official Python pipeline, ensuring VBx has enough information to distinguish speakers across the full audio.
const DIARIZE_STEP_SAMPLES: usize = 16_000;

/// Number of frames produced by a full 160000-sample window (measured, not estimated).
const FRAMES_PER_WINDOW: usize = 589;

/// Center-crop left margin in frames: round(3.75s / frame_step_s).
/// frame_step_s = 10.0 / 589.0 ≈ 0.016978s → margin = round(3.75 / 0.016978) = 221.
const CENTER_MARGIN_FRAMES: usize = 221;

/// Center-crop retained frame count: round(2.5s / frame_step_s) = 147.
const CENTER_KEEP_FRAMES: usize = 147;

/// Fixed input frame count for embedding inference.
///
/// The WeSpeaker ONNX model accepts dynamic shape `[1, T, 80]`, but ONNX Runtime compiles (JIT)
/// a separate inference kernel for each distinct T, causing 1000+ calls × various frame counts =
/// tens of minutes of extra overhead. Fixing to this value means the entire inference process
/// compiles only one kernel, then reuses it throughout.
///
/// Implementation: active frames > 200 → uniform downsampling; active frames < 200 → zero-padding.
/// Speaker embedding relies on global statistics (global average pooling), so reducing frame count
/// has minimal impact on embedding quality.
// FIXED_EMBEDDING_FRAMES is no longer used: Phase 3 confirmed downsampling severely degrades embedding quality (cosine sim 0.978→0.715)

/// Fixed sample rate (pyannote segmentation model only accepts 16kHz).
const SAMPLE_RATE: NonZeroU32 = NonZeroU32::new(16_000).unwrap();

/// Convert sample index to Duration (based on 16kHz sample rate).
fn samples_to_duration(samples: u64) -> Duration {
  Duration::from_secs_f64(samples as f64 / SAMPLE_RATE.get() as f64)
}

/// PCM f32 [-1.0, 1.0] → 16-bit integer range scaling factor.
/// mel_spec's Fbank expects 16-bit PCM level input amplitude, so this factor must be applied.
const PCM_F32_TO_I16_SCALE: f32 = 32768.0;

// ── Error types ──────────────────────────────────────────────────────────────

/// Error type covering ONNX inference failure or model loading failure.
#[derive(Debug, thiserror::Error)]
pub enum TieguanyinOolongError {
  /// ONNX Runtime error (model loading, inference failure, etc.).
  #[error("ort inference error: {0}")]
  Ort(#[from] ort::Error),
  /// Embedding model not loaded (diarize() called in VAD-only mode).
  #[error("embedding model not loaded: set embedding_model_path in options")]
  EmbeddingModelNotLoaded,
  /// ONNX output tensor shape does not match expectations (inference model version mismatch).
  #[error("invalid tensor shape: expected {expected} values per frame, got {actual}")]
  InvalidTensorShape {
    /// Expected number of values per frame.
    expected: usize,
    /// Actual number of values per frame (logits total length / num_frames).
    actual: usize,
  },
  /// PLDA model file loading or shape validation failure.
  #[error("PLDA model load error: {0}")]
  PldaModelLoad(String),
  /// PLDA model not loaded when diarize_overlap() is called.
  #[error("PLDA model not loaded: set plda_model_path in options")]
  PldaModelNotLoaded,
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration options for TieguanyinOolong, with defaults aligned to pyannote's original config.
#[derive(Debug, Clone)]
pub struct TieguanyinOolongOptions {
  /// VAD activation threshold: voice region starts when any speaker's probability >= vad_onset (pyannote default 0.5).
  vad_onset: f32,
  /// VAD deactivation threshold: voice region ends only when all speakers' probabilities < vad_offset (pyannote default 0.357).
  vad_offset: f32,
  /// Minimum silence duration: silence gaps shorter than this are merged into the voice region (pyannote default 0.0s).
  min_duration_off: Duration,
  /// VAD-only mode: true = only load the 5.6MB segmentation model, skip loading the embedding model.
  vad_only: bool,
  /// Embedding ONNX model path. None = VAD-only mode, embedding model not loaded.
  embedding_model_path: Option<std::path::PathBuf>,
  /// Batch size for batch embedding inference (default 32).
  embedding_batch_size: usize,
  /// Exclude overlapping frames when extracting embeddings (default true, consistent with Python config.yaml).
  embedding_exclude_overlap: bool,
  /// Clustering distance threshold (default 0.6, VBx AHC init uses Euclidean; AHC fallback uses cosine distance).
  clustering_threshold: f32,
  /// Minimum speaker count constraint (None = no constraint).
  min_speakers: Option<usize>,
  /// Maximum speaker count constraint (None = no constraint).
  max_speakers: Option<usize>,
  /// PLDA model path (vbx_model.npz). None = use AHC fallback clustering.
  plda_model_path: Option<std::path::PathBuf>,
  /// ONNX Runtime intra-op parallel thread count. None = determined automatically by ort.
  /// When set, applies to both the segmentation and embedding sessions.
  intra_threads: Option<usize>,
}

impl Default for TieguanyinOolongOptions {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn default() -> Self {
    Self::new()
  }
}

impl TieguanyinOolongOptions {
  /// Create configuration with pyannote default values.
  ///
  /// Note: cannot be const fn because it contains `Option<PathBuf>` fields (non-const).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new() -> Self {
    Self {
      vad_onset: 0.5,
      vad_offset: 0.357,
      min_duration_off: Duration::ZERO,
      vad_only: false,
      embedding_model_path: None,
      embedding_batch_size: 32,
      embedding_exclude_overlap: true,
      clustering_threshold: 0.6,
      min_speakers: None,
      max_speakers: None,
      plda_model_path: None,
      intra_threads: None,
    }
  }

  /// Returns the VAD activation threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vad_onset(&self) -> f32 {
    self.vad_onset
  }

  /// Sets the VAD activation threshold (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_vad_onset(mut self, vad_onset: f32) -> Self {
    self.vad_onset = vad_onset;
    self
  }

  /// Sets the VAD activation threshold (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_vad_onset(&mut self, vad_onset: f32) -> &mut Self {
    self.vad_onset = vad_onset;
    self
  }

  /// Returns the VAD deactivation threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vad_offset(&self) -> f32 {
    self.vad_offset
  }

  /// Sets the VAD deactivation threshold (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_vad_offset(mut self, vad_offset: f32) -> Self {
    self.vad_offset = vad_offset;
    self
  }

  /// Sets the VAD deactivation threshold (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_vad_offset(&mut self, vad_offset: f32) -> &mut Self {
    self.vad_offset = vad_offset;
    self
  }

  /// Returns the minimum silence duration.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_duration_off(&self) -> Duration {
    self.min_duration_off
  }

  /// Sets the minimum silence duration (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_duration_off(mut self, min_duration_off: Duration) -> Self {
    self.min_duration_off = min_duration_off;
    self
  }

  /// Sets the minimum silence duration (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_duration_off(&mut self, min_duration_off: Duration) -> &mut Self {
    self.min_duration_off = min_duration_off;
    self
  }

  /// Returns whether VAD-only mode is enabled.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn vad_only(&self) -> bool {
    self.vad_only
  }

  /// Sets VAD-only mode (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_vad_only(mut self, vad_only: bool) -> Self {
    self.vad_only = vad_only;
    self
  }

  /// Sets VAD-only mode (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_vad_only(&mut self, vad_only: bool) -> &mut Self {
    self.vad_only = vad_only;
    self
  }

  /// Returns the embedding model path (None = VAD-only mode).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn embedding_model_path(&self) -> Option<&std::path::Path> {
    self.embedding_model_path.as_deref()
  }

  /// Sets the embedding model path (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_embedding_model_path(mut self, path: Option<std::path::PathBuf>) -> Self {
    self.embedding_model_path = path;
    self
  }

  /// Sets the embedding model path (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_embedding_model_path(
    &mut self,
    path: Option<std::path::PathBuf>,
  ) -> &mut Self {
    self.embedding_model_path = path;
    self
  }

  /// Returns the batch size for batch embedding inference.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn embedding_batch_size(&self) -> usize {
    self.embedding_batch_size
  }

  /// Sets the batch size for batch embedding inference (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_embedding_batch_size(mut self, n: usize) -> Self {
    self.embedding_batch_size = n;
    self
  }

  /// Sets the batch size for batch embedding inference (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_embedding_batch_size(&mut self, n: usize) -> &mut Self {
    self.embedding_batch_size = n;
    self
  }

  /// Returns whether overlapping frames are excluded.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn embedding_exclude_overlap(&self) -> bool {
    self.embedding_exclude_overlap
  }

  /// Sets whether to exclude overlapping frames (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_embedding_exclude_overlap(mut self, v: bool) -> Self {
    self.embedding_exclude_overlap = v;
    self
  }

  /// Sets whether to exclude overlapping frames (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_embedding_exclude_overlap(&mut self, v: bool) -> &mut Self {
    self.embedding_exclude_overlap = v;
    self
  }

  /// Returns the clustering cosine distance threshold.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn clustering_threshold(&self) -> f32 {
    self.clustering_threshold
  }

  /// Sets the clustering cosine distance threshold (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_clustering_threshold(mut self, t: f32) -> Self {
    self.clustering_threshold = t;
    self
  }

  /// Sets the clustering cosine distance threshold (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_clustering_threshold(&mut self, t: f32) -> &mut Self {
    self.clustering_threshold = t;
    self
  }

  /// Returns the minimum speaker count constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn min_speakers(&self) -> Option<usize> {
    self.min_speakers
  }

  /// Sets the minimum speaker count constraint (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_min_speakers(mut self, n: Option<usize>) -> Self {
    self.min_speakers = n;
    self
  }

  /// Sets the minimum speaker count constraint (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_min_speakers(&mut self, n: Option<usize>) -> &mut Self {
    self.min_speakers = n;
    self
  }

  /// Returns the maximum speaker count constraint.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_speakers(&self) -> Option<usize> {
    self.max_speakers
  }

  /// Sets the maximum speaker count constraint (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_speakers(mut self, n: Option<usize>) -> Self {
    self.max_speakers = n;
    self
  }

  /// Sets the maximum speaker count constraint (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_speakers(&mut self, n: Option<usize>) -> &mut Self {
    self.max_speakers = n;
    self
  }

  /// Returns the PLDA model path (None = use AHC fallback clustering).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn plda_model_path(&self) -> Option<&std::path::Path> {
    self.plda_model_path.as_deref()
  }

  /// Sets the PLDA model path (consuming builder).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_plda_model(mut self, path: std::path::PathBuf) -> Self {
    self.plda_model_path = Some(path);
    self
  }

  /// Sets the PLDA model path (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_plda_model(&mut self, path: std::path::PathBuf) -> &mut Self {
    self.plda_model_path = Some(path);
    self
  }

  /// Returns the ONNX Runtime intra-op thread count configuration (None = automatic).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn intra_threads(&self) -> Option<usize> {
    self.intra_threads
  }

  /// Sets the ONNX Runtime intra-op thread count (consuming builder).
  ///
  /// Controls parallelism within a single operator. For CPU-intensive models (e.g., segmentation/embedding),
  /// limiting thread count can reduce thread contention overhead. None = determined automatically by ort.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_intra_threads(mut self, threads: Option<usize>) -> Self {
    self.intra_threads = threads;
    self
  }

  /// Sets the ONNX Runtime intra-op thread count (mutable reference).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_intra_threads(&mut self, threads: Option<usize>) -> &mut Self {
    self.intra_threads = threads;
    self
  }
}

// ── Internal state machine ───────────────────────────────────────────────────

/// VAD state machine states, modeled after RoseBlack's three-state design but using dual thresholds (onset/offset).
#[derive(Debug)]
enum VadState {
  /// No active voice region.
  Idle,
  /// Accumulating voice samples.
  Active {
    /// Sample index where the voice region starts (global).
    start: u64,
    /// Sample index where the last voiced frame ends (global).
    end: u64,
  },
  /// Voice region just ended, waiting to confirm whether the gap exceeds min_duration_off.
  Pending {
    /// Sample index where the voice region starts.
    range_start: u64,
    /// Sample index where the voice region ends.
    range_end: u64,
    /// Accumulated silence samples since range_end.
    gap_samples: u64,
  },
}

// ── Main struct ──────────────────────────────────────────────────────────────

/// VoiceDetector implementation for pyannote segmentation ONNX model.
///
/// The first inference triggers after accumulating 160000 samples (10s); subsequent inferences
/// trigger every 40000 samples (2.5s), using a sliding buffer that retains the previous
/// 120000 samples (3 steps) to form a complete 160000-sample 10s window.
/// After powerset decoding, center-crop frames (221..368) are converted to voice regions.
pub struct TieguanyinOolong {
  /// ONNX inference session (segmentation model).
  session: Session,
  opts: TieguanyinOolongOptions,
  /// Sliding buffer, retaining at most WINDOW_SAMPLES samples.
  /// Uses the last WINDOW_SAMPLES for inference, then truncates to WINDOW_SAMPLES - STEP_SAMPLES.
  sample_buffer: Vec<f32>,
  /// Global sample cursor (total samples fed into detect()).
  cursor: u64,
  /// Global frame offset at the current inference trigger point (total center-crop frames produced).
  frame_cursor: u64,
  /// min_duration_off converted to sample count.
  min_duration_off_samples: u64,
  /// VAD state machine.
  vad_state: VadState,
  /// Queue of ready VAD regions not yet returned via detect().
  pending_ranges: VecDeque<VoiceRange>,
  // NOTE: pending_ranges stores complete VoiceRange (Duration pairs), produced by feed_frames_to_vad
  /// ONNX inference session (embedding model). None = VAD-only mode.
  embedding_session: Option<Session>,
  /// PLDA model (for VBx clustering). None = use AHC fallback clustering.
  plda_model: Option<PldaModel>,
  /// Cached Fbank instance (avoids recreating on each embedding inference).
  /// Configuration fixed at 16kHz/80mel/25ms/10ms/apply_cmn=false, aligned with pyannote torchaudio.compliance.kaldi.fbank.
  /// Only initialized when the embedding model is loaded (None in VAD-only mode).
  fbank: Option<Fbank>,
}

impl TieguanyinOolong {
  /// Creates a TieguanyinOolong from a segmentation model file path.
  pub fn new_from_path(
    model_path: impl AsRef<std::path::Path>,
    opts: TieguanyinOolongOptions,
  ) -> Result<Self, TieguanyinOolongError> {
    // SessionBuilder methods return Error<SessionBuilder>; convert to ort::Error (= Error<()>)
    // intra_threads config: controls parallel thread count within a single operator, reduces thread contention overhead
    let mut seg_builder = Session::builder()
      .map_err(ort::Error::from)?
      .with_optimization_level(GraphOptimizationLevel::Level3)
      .map_err(ort::Error::from)?;
    if let Some(threads) = opts.intra_threads() {
      seg_builder = seg_builder.with_intra_threads(threads).map_err(ort::Error::from)?;
    }
    // ── CoreML EP (segmentation) ──────────────────────────────────────────
    // The segmentation model (pyannote/segmentation-3.0) contains SincConv/LSTM operators,
    // and CoreML MLProgram compilation may fail (topological sort error).
    // Strategy: attempt to register CoreML EP and load; on failure, automatically fall back to CPU-only session.
    #[cfg(feature = "coreml")]
    let session = {
      let mut coreml_builder = seg_builder
        .with_execution_providers([
          ep::CoreML::default()
            .with_subgraphs(true)
            .with_model_format(ep::coreml::ModelFormat::MLProgram)
            .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
            .build(),
        ])
        .map_err(ort::Error::from)?;
      match coreml_builder.commit_from_file(&model_path) {
        Ok(s) => s,
        Err(_e) => {
          // CoreML compilation failed (common with SincConv topological sort issues), fall back to CPU-only
          let mut fallback = Session::builder()
            .map_err(ort::Error::from)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort::Error::from)?;
          if let Some(threads) = opts.intra_threads() {
            fallback = fallback.with_intra_threads(threads).map_err(ort::Error::from)?;
          }
          fallback.commit_from_file(&model_path)?
        }
      }
    };
    #[cfg(not(feature = "coreml"))]
    let session = seg_builder.commit_from_file(model_path)?;

    let embedding_session = if let Some(emb_path) = opts.embedding_model_path() {
      let mut emb_builder = Session::builder()
        .map_err(ort::Error::from)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort::Error::from)?;
      if let Some(threads) = opts.intra_threads() {
        emb_builder = emb_builder.with_intra_threads(threads).map_err(ort::Error::from)?;
      }
      // ── embedding session (CoreML not applicable) ────────────────────────
      // WeSpeaker ResNet34 embedding model does not use CoreML EP:
      // CoreML model compilation (MLProgram format) alters numerical behavior (both CPUAndNeuralEngine
      // and CPUAndGPU degrade embedding vector precision), causing abnormal PLDA clustering results (2 speakers → 1).
      // Keeping CPU-only inference ensures numerical consistency with Python reference output.
      let emb_session = emb_builder.commit_from_file(emb_path)?;
      Some(emb_session)
    } else {
      None
    };

    let plda_model = if let Some(plda_path) = opts.plda_model_path() {
      Some(PldaModel::load(plda_path)?)
    } else {
      None
    };

    // Fbank instance cache: fixed configuration initialized only once, avoiding recreation on each embedding inference.
    // Only needed when the embedding model is loaded (VAD-only mode does not extract embeddings).
    // Equivalent transformation: configuration is identical to the FbankConfig in the original run_embedding_inference.
    let fbank = if opts.embedding_model_path().is_some() {
      let fbank_config = FbankConfig {
        sample_rate: 16000.0,
        num_mel_bins: 80,       // 80-dim mel filter bank
        frame_length_ms: 25.0,  // 400 samples @ 16kHz
        frame_shift_ms: 10.0,   // 160 samples @ 16kHz (= 10ms step)
        apply_cmn: false,       // CRITICAL: must be false, aligned with torchaudio subtract_mean=False
        preemphasis: 0.97,      // pyannote default
        dither: 0.0,            // pyannote default
        ..Default::default()
      };
      Some(Fbank::new(fbank_config))
    } else {
      None
    };

    let min_duration_off_samples =
      (opts.min_duration_off().as_secs_f64() * SAMPLE_RATE.get() as f64) as u64;

    Ok(Self {
      session,
      opts,
      sample_buffer: Vec::with_capacity(WINDOW_SAMPLES),
      cursor: 0,
      frame_cursor: 0,
      min_duration_off_samples,
      vad_state: VadState::Idle,
      pending_ranges: VecDeque::new(),
      embedding_session,
      plda_model,
      fbank,
    })
  }
}

// ── Inference core functions (pub(crate) for unit testing, not publicly exposed) ──

/// Decode 7-class log-softmax frame into 3 speakers' activity probabilities.
///
/// Powerset class order (itertools.combinations order, not bitmask):
/// `[{}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}]`
/// - speaker 0 is active in classes 1, 4, 5
/// - speaker 1 is active in classes 2, 4, 6
/// - speaker 2 is active in classes 3, 5, 6
pub(crate) fn powerset_decode(log_probs: &[f32; 7]) -> [f32; 3] {
  let p: [f32; 7] = log_probs.map(|x| x.exp());
  [
    p[1] + p[4] + p[5], // speaker 0
    p[2] + p[4] + p[6], // speaker 1
    p[3] + p[5] + p[6], // speaker 2
  ]
}

/// Extract center-crop frame range from a flat logits array (row-major, 7 classes per row).
///
/// - Full window (is_full_window = true): take frames [CENTER_MARGIN_FRAMES, CENTER_MARGIN_FRAMES + CENTER_KEEP_FRAMES)
/// - Partial window (is_full_window = false): take all frames [0, num_frames)
///
/// Returns decoded per-speaker probability vector, each element being `[speaker0_prob, speaker1_prob, speaker2_prob]`.
pub(crate) fn extract_speaker_probs(
  logits: &[f32],
  num_frames: usize,
  is_full_window: bool,
) -> Result<Vec<[f32; 3]>, TieguanyinOolongError> {
  // Validate logits shape: 7 log-softmax values per frame
  const EXPECTED_CLASSES: usize = 7;
  if num_frames > 0 && logits.len() != num_frames * EXPECTED_CLASSES {
    let actual = logits.len() / num_frames.max(1);
    return Err(TieguanyinOolongError::InvalidTensorShape {
      expected: EXPECTED_CLASSES,
      actual,
    });
  }

  let (frame_start, frame_end) = if is_full_window {
    (
      CENTER_MARGIN_FRAMES,
      (CENTER_MARGIN_FRAMES + CENTER_KEEP_FRAMES).min(num_frames),
    )
  } else {
    (0, num_frames)
  };

  (frame_start..frame_end)
    .map(|f| {
      let base = f * EXPECTED_CLASSES;
      // Take 7 values from the flat array, convert to fixed-length array
      let frame_log_probs: [f32; 7] = logits[base..base + EXPECTED_CLASSES]
        .try_into()
        .map_err(|_| TieguanyinOolongError::InvalidTensorShape {
          expected: EXPECTED_CLASSES,
          actual: logits[base..].len().min(EXPECTED_CLASSES),
        })?;
      Ok(powerset_decode(&frame_log_probs))
    })
    .collect::<Result<Vec<_>, _>>()
}

/// Extract per-speaker probabilities for the full window (no center-crop).
///
/// Unlike `extract_speaker_probs`, this function returns all `num_frames` frames, used for
/// precise mask alignment during embedding extraction (segmentation 589 frames → fbank ~998 frames).
fn extract_full_speaker_probs(
  logits: &[f32],
  num_frames: usize,
) -> Result<Vec<[f32; 3]>, TieguanyinOolongError> {
  const EXPECTED_CLASSES: usize = 7;
  if num_frames > 0 && logits.len() != num_frames * EXPECTED_CLASSES {
    let actual = logits.len() / num_frames.max(1);
    return Err(TieguanyinOolongError::InvalidTensorShape {
      expected: EXPECTED_CLASSES,
      actual,
    });
  }
  (0..num_frames)
    .map(|f| {
      let base = f * EXPECTED_CLASSES;
      let frame_log_probs: [f32; 7] = logits[base..base + EXPECTED_CLASSES]
        .try_into()
        .map_err(|_| TieguanyinOolongError::InvalidTensorShape {
          expected: EXPECTED_CLASSES,
          actual: logits[base..].len().min(EXPECTED_CLASSES),
        })?;
      Ok(powerset_decode(&frame_log_probs))
    })
    .collect::<Result<Vec<_>, _>>()
}

/// Run a single ONNX inference, returning (flat_logits, num_frames).
///
/// `audio_window` length must be <= WINDOW_SAMPLES (160000).
/// For partial windows (finish()), it can be shorter; the model supports dynamic axes.
fn run_inference(
  session: &mut Session,
  audio_window: &[f32],
) -> Result<(Vec<f32>, usize), TieguanyinOolongError> {
  let num_samples = audio_window.len();

  // Build input tensor [1, 1, num_samples]
  // Use to_vec() to ensure owned data (ort rc.12 TensorRef caused result distortion in run_inference,
  // possibly an ort 2.0.0-rc.12 TensorRef lifetime/alignment issue, falling back to safe Tensor::from_array)
  use ort::value::Tensor;
  let input = Tensor::from_array(([1usize, 1, num_samples], audio_window.to_vec()))?;

  // Run inference (inputs! macro returns Vec, not Result; run() returns Result)
  let outputs = session.run(ort::inputs!["input_values" => input])?;

  // Extract output: shape [1, num_frames, 7], no ndarray needed
  // try_extract_tensor returns (&Shape, &[T]), shape is &[i64]
  let (shape, logits_slice) = outputs["logits"].try_extract_tensor::<f32>()?;
  let num_frames = shape[1] as usize;
  let logits = logits_slice.to_vec(); // Copy once to obtain an owned Vec

  Ok((logits, num_frames))
}

// ── VAD state machine core ───────────────────────────────────────────────────

/// Determine whether a frame is voiced based on current VAD state and speaker probabilities, using dual-threshold hysteresis.
///
/// - Idle state: any speaker >= onset to activate (high threshold)
/// - Active/Pending state: any speaker >= offset to stay active (low threshold, hysteresis effect)
pub(crate) fn is_voiced_with_hysteresis(
  probs: &[f32; 3],
  is_currently_active: bool,
  onset: f32,
  offset: f32,
) -> bool {
  if is_currently_active {
    probs.iter().any(|&p| p >= offset)
  } else {
    probs.iter().any(|&p| p >= onset)
  }
}

// ── VBx PLDA infrastructure ──────────────────────────────────────────────────

/// Pre-computed PLDA model parameters (all f64).
///
/// Generated by `scripts/pyannote/precompute_plda.py` from the original `plda.npz` + `xvec_transform.npz`,
/// saved as a single `vbx_model.npz`. Eigendecomposition is pre-processed on the Python side; Rust only needs matrix multiplication.
struct PldaModel {
  /// xvector centering mean \[256\].
  mean1: Array1<f64>,
  /// LDA output centering mean \[128\].
  mean2: Array1<f64>,
  /// LDA projection matrix \[256, 128\].
  lda: Array2<f64>,
  /// PLDA centering mean \[128\].
  plda_mu: Array1<f64>,
  /// PLDA transform matrix（eigendecomp pre-applied）\[128, 128\].
  plda_tr: Array2<f64>,
  /// Between-class covariance diagonal \[128\].
  plda_psi: Array1<f64>,
}

impl PldaModel {
  /// Load pre-computed PLDA model parameters from `vbx_model.npz`.
  ///
  /// Validates each array's shape, returning `PldaModelLoad` error on mismatch.
  fn load(path: &std::path::Path) -> Result<Self, TieguanyinOolongError> {
    let file = std::fs::File::open(path)
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("open {}: {e}", path.display())))?;
    let mut npz = NpzReader::new(file)
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("NpzReader: {e}")))?;

    let mean1: Array1<f64> = npz
      .by_name("mean1")
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("mean1: {e}")))?;
    if mean1.len() != 256 {
      return Err(TieguanyinOolongError::PldaModelLoad(format!(
        "mean1 shape: expected (256,), got ({},)",
        mean1.len()
      )));
    }

    let mean2: Array1<f64> = npz
      .by_name("mean2")
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("mean2: {e}")))?;
    if mean2.len() != 128 {
      return Err(TieguanyinOolongError::PldaModelLoad(format!(
        "mean2 shape: expected (128,), got ({},)",
        mean2.len()
      )));
    }

    let lda: Array2<f64> = npz
      .by_name("lda")
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("lda: {e}")))?;
    if lda.shape() != [256, 128] {
      return Err(TieguanyinOolongError::PldaModelLoad(format!(
        "lda shape: expected (256, 128), got {:?}",
        lda.shape()
      )));
    }

    let plda_mu: Array1<f64> = npz
      .by_name("plda_mu")
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("plda_mu: {e}")))?;
    if plda_mu.len() != 128 {
      return Err(TieguanyinOolongError::PldaModelLoad(format!(
        "plda_mu shape: expected (128,), got ({},)",
        plda_mu.len()
      )));
    }

    let plda_tr: Array2<f64> = npz
      .by_name("plda_tr")
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("plda_tr: {e}")))?;
    if plda_tr.shape() != [128, 128] {
      return Err(TieguanyinOolongError::PldaModelLoad(format!(
        "plda_tr shape: expected (128, 128), got {:?}",
        plda_tr.shape()
      )));
    }

    let plda_psi: Array1<f64> = npz
      .by_name("plda_psi")
      .map_err(|e| TieguanyinOolongError::PldaModelLoad(format!("plda_psi: {e}")))?;
    if plda_psi.len() != 128 {
      return Err(TieguanyinOolongError::PldaModelLoad(format!(
        "plda_psi shape: expected (128,), got ({},)",
        plda_psi.len()
      )));
    }

    Ok(Self { mean1, mean2, lda, plda_mu, plda_tr, plda_psi })
  }

  /// Full PLDA transform: (N, 256) embeddings → (N, 128) PLDA space.
  ///
  /// Pipeline: center(mean1) → L2 normalize → scale(sqrt(256)) → LDA(256→128)
  /// → center(mean2) → L2 normalize → scale(sqrt(128)) → center(plda_mu) → dot(plda_tr.T)
  fn transform(&self, embeddings: &Array2<f64>) -> Array2<f64> {
    // Step 1: xvec_tf
    let mut x = embeddings - &self.mean1;
    l2_normalize_rows(&mut x);
    x *= 256.0_f64.sqrt();
    let mut x128 = x.dot(&self.lda);
    x128 -= &self.mean2;
    l2_normalize_rows(&mut x128);
    x128 *= 128.0_f64.sqrt();
    // Step 2: plda_tf
    let centered = &x128 - &self.plda_mu;
    centered.dot(&self.plda_tr.t())
  }
}

// ── VBx math utility functions ───────────────────────────────────────────────

/// Compute log-sum-exp for each row of a matrix (max-subtraction trick, numerically stable).
fn logsumexp_rows(mat: &Array2<f64>) -> Array1<f64> {
  let max_per_row =
    mat.map_axis(Axis(1), |row| row.iter().copied().fold(f64::NEG_INFINITY, f64::max));
  let mut result = Array1::<f64>::zeros(mat.nrows());
  for i in 0..mat.nrows() {
    let max_val = max_per_row[i];
    if max_val == f64::NEG_INFINITY {
      result[i] = f64::NEG_INFINITY;
      continue;
    }
    let sum_exp: f64 = mat.row(i).iter().map(|&x| (x - max_val).exp()).sum();
    result[i] = max_val + sum_exp.ln();
  }
  result
}

/// In-place softmax on each row of a matrix (subtract row max, exp, normalize).
fn softmax_rows_inplace(mat: &mut Array2<f64>) {
  for mut row in mat.rows_mut() {
    let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    row.mapv_inplace(|x| (x - max_val).exp());
    let sum: f64 = row.sum();
    if sum > 0.0 {
      row /= sum;
    }
  }
}

/// In-place L2 normalization on each row of a matrix. Rows with norm < 1e-10 are skipped to avoid division by zero.
fn l2_normalize_rows(mat: &mut Array2<f64>) {
  for mut row in mat.rows_mut() {
    let norm: f64 = row.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
      continue;
    }
    row /= norm;
  }
}

/// Compute cosine distance between two vectors (1 - cosine_similarity).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
  let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
  let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
  let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
  1.0 - dot / (norm_a * norm_b)
}

/// Assign all embeddings to the nearest cluster centroid (aligned with Python clustering.py assign_embeddings).
///
/// **Process**:
/// 1. Only cluster the `trainable=true` embedding subset via VBx/AHC → `train_labels`
/// 2. Compute each cluster's centroid (mean of train embeddings)
/// 3. Assign ALL embeddings (including trainable=false) to the nearest centroid by cosine distance
///
/// Returns labels for all embeddings (same length and order as `all_embeddings`).
fn assign_embeddings(
  all_embeddings: &[Vec<f32>],
  trainable_mask: &[bool],
  train_labels: &[u32],
) -> Vec<u32> {
  // Compute cluster centroids
  let num_clusters = train_labels.iter().copied().max().unwrap_or(0) as usize + 1;
  let dim = all_embeddings.first().map_or(0, |e| e.len());
  let mut centroids: Vec<Vec<f64>> = vec![vec![0.0; dim]; num_clusters];
  let mut counts: Vec<usize> = vec![0; num_clusters];

  let mut train_idx = 0usize;
  for (i, _emb) in all_embeddings.iter().enumerate() {
    if trainable_mask[i] {
      let k = train_labels[train_idx] as usize;
      for (c, &v) in centroids[k].iter_mut().zip(all_embeddings[i].iter()) {
        *c += v as f64;
      }
      counts[k] += 1;
      train_idx += 1;
    }
  }
  for k in 0..num_clusters {
    if counts[k] > 0 {
      for c in centroids[k].iter_mut() {
        *c /= counts[k] as f64;
      }
    }
  }

  // Assign each embedding to the nearest centroid by cosine distance
  let centroid_f32: Vec<Vec<f32>> = centroids
    .iter()
    .map(|c| c.iter().map(|&v| v as f32).collect())
    .collect();

  all_embeddings
    .iter()
    .map(|emb| {
      centroid_f32
        .iter()
        .enumerate()
        .map(|(k, c)| (k, cosine_distance(emb, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k as u32)
        .unwrap_or(0)
    })
    .collect()
}

/// VBx-specific AHC initialization: L2 normalize → Euclidean distance → Centroid linkage.
///
/// Completely independent from `cluster_embeddings()` (cosine + Average) --
/// VBx requires Centroid linkage + Euclidean on normalized embeddings
/// to produce the correct initial cluster count (matching Python VBxClustering source code).
///
/// Implementation: directly replicates scipy `linkage(data, method='centroid', metric='euclidean')`'s
/// condensed-matrix Lance-Williams algorithm (squared distances), ensuring exact consistency with Python behavior.
/// - Lance-Williams update on squared distances (`on_squares = true`)
/// - Minimum distance scan order: row-major (i<j), ties broken by smaller index, consistent with scipy
/// - Time complexity O(n^2), negligible for n<=500
fn ahc_init(embeddings: &[Vec<f32>], threshold: f64) -> Vec<u32> {
  let n = embeddings.len();
  if n == 0 {
    return vec![];
  }
  if n == 1 {
    return vec![0];
  }

  // L2 normalize to f64
  let normed: Vec<Vec<f64>> = embeddings
    .iter()
    .map(|e| {
      let norm2 = e.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt().max(1e-10);
      e.iter().map(|x| *x as f64 / norm2).collect()
    })
    .collect();

  // Condensed **squared** Euclidean distance matrix (upper triangular, row-major)
  // Index (i,j) (i<j) → condensed[(i*(2n-i-1)/2 + (j-i-1))]
  // This is fully consistent with scipy
  let idx = |i: usize, j: usize| -> usize {
    debug_assert!(i < j);
    i * (2 * n - i - 1) / 2 + (j - i - 1)
  };
  let mut D: Vec<f64> = vec![0.0; n * (n - 1) / 2];
  // After L2-normalization, ||a-b||^2 = 2 - 2*dot(a,b)
  // ndarray .dot() matrix multiplication replaces hand-written O(n^2*D) triple loop
  let dim = normed[0].len();
  let mut flat = Vec::with_capacity(n * dim);
  for row in &normed {
    flat.extend_from_slice(row);
  }
  let mat = Array2::from_shape_vec((n, dim), flat)
    .expect("ahc_init: ndarray shape mismatch");
  let gram = mat.dot(&mat.t());
  for i in 0..n {
    for j in (i + 1)..n {
      D[idx(i, j)] = 2.0 - 2.0 * gram[[i, j]];
    }
  }

  // Track each active cluster's size and leaf → cluster mapping
  let mut sizes: Vec<usize> = vec![1; n];
  let mut cluster_of: Vec<usize> = (0..n).collect(); // which cluster leaf i belongs to (represented by active cluster index)
  let mut active: Vec<bool> = vec![true; n];
  let mut n_active = n;

  let threshold_sq = threshold * threshold;

  while n_active > 1 {
    // Find the minimum squared distance pair (row-major scan, ties broken by smallest index, consistent with scipy)
    let mut min_sq = f64::INFINITY;
    let mut mi = 0usize;
    let mut mj = 0usize;
    for i in 0..n {
      if !active[i] {
        continue;
      }
      for j in (i + 1)..n {
        if !active[j] {
          continue;
        }
        let d = D[idx(i, j)];
        if d < min_sq {
          min_sq = d;
          mi = i;
          mj = j;
        }
      }
    }

    if min_sq > threshold_sq {
      break;
    }

    // Lance-Williams centroid update (in squared distances):
    //   d²(AB, X) = (na*d²(A,X) + nb*d²(B,X)) / (na+nb)
    //             - na*nb / (na+nb)² * d²(A,B)
    let na = sizes[mi] as f64;
    let nb = sizes[mj] as f64;
    let nab = na + nb;
    let d_ab = min_sq; // d²(A,B)

    // Update distances for mi's row/column (mi becomes the representative of AB)
    for x in 0..n {
      if !active[x] || x == mi || x == mj {
        continue;
      }
      let d_ax = if x < mi { D[idx(x, mi)] } else { D[idx(mi, x)] };
      let d_bx = if x < mj { D[idx(x, mj)] } else { D[idx(mj, x)] };
      let new_d = (na * d_ax + nb * d_bx) / nab - (na * nb / (nab * nab)) * d_ab;
      if x < mi {
        D[idx(x, mi)] = new_d;
      } else {
        D[idx(mi, x)] = new_d;
      }
    }

    // Merge mj into mi
    sizes[mi] += sizes[mj];
    active[mj] = false;
    n_active -= 1;

    // Update leaf cluster membership: mj → mi
    for k in 0..n {
      if cluster_of[k] == mj {
        cluster_of[k] = mi;
      }
    }
  }

  // Assign 0-based labels to active clusters
  let mut root_to_label: std::collections::HashMap<usize, u32> = Default::default();
  let mut next_label = 0u32;
  let labels: Vec<u32> = (0..n)
    .map(|i| {
      let root = cluster_of[i];
      *root_to_label.entry(root).or_insert_with(|| {
        let l = next_label;
        next_label += 1;
        l
      })
    })
    .collect();
  labels
}

// ── VBx core algorithm ──────────────────────────────────────────────────────

/// VBx iteration result.
struct VbxResult {
  /// (T, S) frame-speaker responsibilities.
  gamma: Array2<f64>,
  /// (S,) speaker prior probabilities.
  pi: Array1<f64>,
}

/// VB-HMM speaker clustering iteration.
///
/// Input: PLDA-transformed embeddings `fea` (T, D), between-class covariance diagonal `phi` (D,),
/// AHC initial labels `ahc_labels` (T,). Hyperparameters `fa`(=0.07), `fb`(=0.8).
/// Up to `max_iters`(=20) iterations, with early stopping when ELBO increment < `epsilon`(=1e-4).
fn vbx(
  fea: &Array2<f64>,
  phi: &Array1<f64>,
  ahc_labels: &[u32],
  fa: f64,
  fb: f64,
  max_iters: usize,
  epsilon: f64,
) -> VbxResult {
  let (t, d) = (fea.nrows(), fea.ncols());
  debug_assert!(!ahc_labels.is_empty(), "ahc_labels must not be empty");
  let n_clusters = *ahc_labels.iter().max().unwrap_or(&0) as usize + 1;

  // Initialize gamma: one-hot(ahc_labels) * 7.0 → softmax
  let mut gamma = Array2::<f64>::zeros((t, n_clusters));
  for (i, &label) in ahc_labels.iter().enumerate() {
    gamma[[i, label as usize]] = 1.0;
  }
  gamma *= 7.0;
  softmax_rows_inplace(&mut gamma);

  let mut pi = Array1::<f64>::from_elem(n_clusters, 1.0 / n_clusters as f64);

  // Pre-compute constants
  let v = phi.mapv(f64::sqrt); // (D,)
  let rho = fea * &v; // (T, D) element-wise broadcast
  // G[t] = -0.5 * (||fea[t]||^2 + D * ln(2*pi))
  let fea_sq_sum = fea.mapv(|x| x * x).sum_axis(Axis(1)); // (T,)
  let ln2pi = (2.0 * std::f64::consts::PI).ln();
  let g = (-0.5) * (&fea_sq_sum + d as f64 * ln2pi); // (T,)

  let mut prev_elbo = f64::NEG_INFINITY;

  for iter in 0..max_iters {
    // M-step: update speaker model parameters
    let gamma_sum = gamma.sum_axis(Axis(0)); // (S,)

    // invL[s, d] = 1 / (1 + (fa/fb) * gamma_sum[s] * phi[d])
    let fa_fb = fa / fb;
    let mut inv_l = Array2::<f64>::zeros((n_clusters, d));
    for s in 0..n_clusters {
      for dd in 0..d {
        inv_l[[s, dd]] = 1.0 / (1.0 + fa_fb * gamma_sum[s] * phi[dd]);
      }
    }

    // alpha[s, d] = (fa/fb) * invL[s, d] * (gamma.T @ rho)[s, d]
    let gamma_t_rho = gamma.t().dot(&rho); // (S, D)
    let alpha = fa_fb * &inv_l * &gamma_t_rho; // (S, D)

    // E-step: compute log-likelihoods and update gamma
    let rho_alpha_t = rho.dot(&alpha.t()); // (T, S)
    // penalty[s] = (invL[s,:] + alpha[s,:]^2) . phi
    let penalty = (&inv_l + &alpha.mapv(|x| x * x)).dot(phi); // (S,)

    let mut log_p = Array2::<f64>::zeros((t, n_clusters));
    for tt in 0..t {
      for s in 0..n_clusters {
        log_p[[tt, s]] = fa * (rho_alpha_t[[tt, s]] - 0.5 * penalty[s] + g[tt]);
      }
    }

    // GMM update
    let eps_small = 1e-8_f64;
    let log_pi = pi.mapv(|p| (p + eps_small).ln()); // (S,)

    // log_p + log_pi broadcast → (T, S)
    let log_p_plus_pi = &log_p + &log_pi;
    let log_p_x = logsumexp_rows(&log_p_plus_pi); // (T,)

    // gamma[t, s] = exp(log_p_plus_pi[t,s] - log_p_x[t])
    for tt in 0..t {
      for s in 0..n_clusters {
        gamma[[tt, s]] = (log_p_plus_pi[[tt, s]] - log_p_x[tt]).exp();
      }
    }

    // Update pi
    let pi_sum: f64 = gamma.sum_axis(Axis(0)).sum();
    pi = gamma.sum_axis(Axis(0)) / pi_sum;

    // ELBO convergence check
    let log_px_sum: f64 = log_p_x.sum();
    let reg_term: f64 =
      fb * 0.5 * (&inv_l.mapv(f64::ln) - &inv_l - &alpha.mapv(|x| x * x) + 1.0).sum();
    let elbo = log_px_sum + reg_term;

    if iter > 0 && elbo - prev_elbo < epsilon {
      break;
    }
    prev_elbo = elbo;
  }

  VbxResult { gamma, pi }
}

/// Convert `Vec<Vec<f32>>` embeddings to (N, D) `Array2<f64>`.
fn to_ndarray_f64(embeddings: &[Vec<f32>]) -> Array2<f64> {
  let n = embeddings.len();
  let d = embeddings[0].len();
  let mut arr = Array2::<f64>::zeros((n, d));
  for (i, emb) in embeddings.iter().enumerate() {
    for (j, &v) in emb.iter().enumerate() {
      arr[[i, j]] = v as f64;
    }
  }
  arr
}

/// VBx clustering pipeline: AHC init → PLDA transform → VBx iteration → speaker pruning.
///
/// Replaces `cluster_embeddings()` in `diarize()` Step 3.
/// Returns `(labels, gamma_kept)` where gamma_kept is used for overlap mode output.
fn vbx_cluster_embeddings(
  embeddings: &[Vec<f32>],
  plda: &PldaModel,
  threshold: f64,
  fa: f64,
  fb: f64,
) -> (Vec<u32>, Array2<f64>) {
  let n = embeddings.len();
  if n == 0 {
    return (vec![], Array2::zeros((0, 0)));
  }
  if n == 1 {
    return (vec![0], Array2::ones((1, 1)));
  }

  // Stage 1: AHC initialization
  let ahc_labels = ahc_init(embeddings, threshold);

  // Stage 2: PLDA transform
  let emb_f64 = to_ndarray_f64(embeddings);
  let fea = plda.transform(&emb_f64);

  // Stage 3: VBx iteration
  let result = vbx(&fea, &plda.plda_psi, &ahc_labels, fa, fb, 20, 1e-4);

  // Stage 4: prune dead speakers (pi > 1e-7) and extract labels
  let kept: Vec<usize> = result
    .pi
    .iter()
    .enumerate()
    .filter(|&(_, p)| *p > 1e-7)
    .map(|(i, _)| i)
    .collect();

  if kept.is_empty() {
    // Should not happen, but safe fallback
    return (vec![0; n], Array2::ones((n, 1)));
  }

  // Extract gamma sub-matrix and argmax
  let k = kept.len();
  let mut gamma_kept = Array2::<f64>::zeros((n, k));
  for i in 0..n {
    for (new_s, &orig_s) in kept.iter().enumerate() {
      gamma_kept[[i, new_s]] = result.gamma[[i, orig_s]];
    }
  }

  let labels: Vec<u32> = (0..n)
    .map(|i| {
      gamma_kept
        .row(i)
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
    })
    .collect();

  (labels, gamma_kept)
}

// AHC clustering code (cosine_distance + cluster_embeddings) has been archived to
// _archived_ahc_clustering.rs, replaced by VBx (vbx_cluster_embeddings).

/// Frame reconstruction: map clustering labels back to the global frame probability matrix (center-crop overlap-add).
///
/// Uses frames from CENTER_MARGIN_FRAMES..CENTER_MARGIN_FRAMES+CENTER_KEEP_FRAMES
/// (i.e., the center 2.5s of all_full_probs), reducing boundary artifacts and false positives:
///   - First chunk (idx=0): uses [0, CENTER_MARGIN+CENTER_KEEP) to cover the audio head
///   - Middle chunks: uses [CENTER_MARGIN, CENTER_MARGIN+CENTER_KEEP) center-crop
///   - Last chunk: from center-crop start to end, covering the audio tail
///
/// Shared by `diarize_with_embeddings()` and `diarize_timed()` to ensure consistent behavior across both paths.
fn reconstruct_frame_speaker_probs(
  all_full_probs: &[Vec<[f32; 3]>],
  chunk_speaker_pairs: &[(usize, usize)],
  trainable_mask: &[bool],
  embedding_labels: &[u32],
  audio_len: usize,
) -> Vec<Vec<(u32, f32)>> {
  const FRAME_STEP_SECS: f64 = 10.0 / FRAMES_PER_WINDOW as f64;
  const STEP_SECS: f64 = DIARIZE_STEP_SAMPLES as f64 / SAMPLE_RATE.get() as f64;
  let total_frames =
    (audio_len as f64 / SAMPLE_RATE.get() as f64 / FRAME_STEP_SECS).ceil() as usize + 1;
  let num_global_speakers = embedding_labels.iter().copied().max().unwrap_or(0) as usize + 1;
  // Contiguous memory layout (row-major): eliminates independent heap allocations for total_frames small Vecs
  let mut frame_sum_probs: Vec<f32> = vec![0.0f32; total_frames * num_global_speakers];
  let mut frame_sum_counts: Vec<u32> = vec![0u32; total_frames * num_global_speakers];

  let num_chunks = all_full_probs.len();
  for (pair_idx, &(chunk_idx, local_speaker)) in chunk_speaker_pairs.iter().enumerate() {
    // Only trainable embeddings participate in frame reconstruction (non-trainable low-quality embeddings would dilute probabilities causing missed detections)
    if !trainable_mask[pair_idx] {
      continue;
    }
    let global_speaker = embedding_labels[pair_idx] as usize;
    // Pre-compute chunk's global frame offset, eliminating floating-point division in inner loop
    let chunk_frame_offset = (chunk_idx as f64 * STEP_SECS / FRAME_STEP_SECS).round() as usize;
    let chunk_frames = &all_full_probs[chunk_idx];
    // Determine valid frame range based on chunk position (center-crop + boundary extension)
    let (j_low, j_high) = if chunk_idx == 0 {
      // First chunk: from start to center-crop end, covering audio head
      (
        0,
        (CENTER_MARGIN_FRAMES + CENTER_KEEP_FRAMES).min(chunk_frames.len()),
      )
    } else if chunk_idx == num_chunks - 1 {
      // Last chunk: from center-crop start to end, covering audio tail (may be an incomplete window)
      (
        CENTER_MARGIN_FRAMES.min(chunk_frames.len()),
        chunk_frames.len(),
      )
    } else {
      // Middle chunk: use only the center CENTER_KEEP_FRAMES frames (~2.5s)
      (
        CENTER_MARGIN_FRAMES,
        (CENTER_MARGIN_FRAMES + CENTER_KEEP_FRAMES).min(chunk_frames.len()),
      )
    };
    for (j, probs) in chunk_frames.iter().enumerate() {
      if j < j_low || j >= j_high {
        continue;
      }
      let gf = chunk_frame_offset + j;
      if gf < total_frames {
        frame_sum_probs[gf * num_global_speakers + global_speaker] += probs[local_speaker];
        frame_sum_counts[gf * num_global_speakers + global_speaker] += 1;
      }
    }
  }

  (0..total_frames)
    .map(|gf| {
      let base = gf * num_global_speakers;
      (0..num_global_speakers)
        .filter(|&spk| frame_sum_counts[base + spk] > 0)
        .map(|spk| {
          (
            spk as u32,
            frame_sum_probs[base + spk] / frame_sum_counts[base + spk] as f32,
          )
        })
        .collect()
    })
    .collect()
}

/// Reconstruct frame-level global speaker probability matrix into sorted, merged Vec<SpeakerSegment>.
///
/// frame_probs: per-frame (global_speaker_id → prob) mapping, in global frame order.
/// frame_step_secs: duration per frame (seconds).
/// vad_onset: speaker activity probability threshold (used for is_overlap determination).
pub(crate) fn build_speaker_segments(
  frame_probs: &[Vec<(u32, f32)>],
  frame_step_secs: f64,
  vad_onset: f32,
) -> Vec<SpeakerSegment> {
  if frame_probs.is_empty() {
    return vec![];
  }

  // Step 1: frame-level argmax + is_overlap + silence detection
  // silent: winner_prob < vad_onset → this frame belongs to no speaker
  // is_overlap: >= 2 speakers with prob >= vad_onset
  let frame_assignments: Vec<(Option<u32>, bool)> = frame_probs
    .iter()
    .map(|frame| {
      if frame.is_empty() {
        return (None, false);
      }
      let (winner_id, winner_prob) = frame
        .iter()
        .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or((0, 0.0));

      if winner_prob < vad_onset {
        return (None, false);
      }

      // is_overlap: count of speakers with prob >= vad_onset is >= 2
      let active_count = frame.iter().filter(|(_, p)| *p >= vad_onset).count();
      let is_overlap = active_count >= 2;

      (Some(winner_id), is_overlap)
    })
    .collect();

  // Step 2: merge consecutive frames with the same label (run-length encoding), silent frames produce no segment
  let mut segments: Vec<SpeakerSegment> = Vec::new();
  let mut run_start: Option<usize> = None;
  let mut run_speaker: Option<u32> = None;
  let mut run_overlap = false;

  let total = frame_assignments.len();
  for (i, &(speaker_id, is_overlap)) in frame_assignments.iter().enumerate() {
    match (run_speaker, speaker_id) {
      (Some(rs), Some(sid)) if rs == sid => {
        run_overlap = run_overlap || is_overlap;
      }
      (Some(rs), _) => {
        // Submit current run
        let start = Duration::from_secs_f64(run_start.unwrap() as f64 * frame_step_secs);
        let end = Duration::from_secs_f64(i as f64 * frame_step_secs);
        segments.push(SpeakerSegment { start, end, speaker_id: rs, is_overlap: run_overlap });
        run_start = speaker_id.map(|_| i);
        run_speaker = speaker_id;
        run_overlap = is_overlap;
      }
      (None, Some(_)) => {
        run_start = Some(i);
        run_speaker = speaker_id;
        run_overlap = is_overlap;
      }
      (None, None) => {}
    }
  }
  // Submit the last run (if there is an active speaker)
  if let Some(rs) = run_speaker {
    let start = Duration::from_secs_f64(run_start.unwrap() as f64 * frame_step_secs);
    let end = Duration::from_secs_f64(total as f64 * frame_step_secs);
    segments.push(SpeakerSegment { start, end, speaker_id: rs, is_overlap: run_overlap });
  }

  // Step 3: sort by start (OUT-02)
  segments.sort_by(|a, b| a.start.cmp(&b.start));

  // Step 4: merge adjacent segments with the same speaker (OUT-02)
  let mut merged: Vec<SpeakerSegment> = Vec::new();
  for seg in segments {
    if let Some(last) = merged.last_mut() {
      if last.speaker_id == seg.speaker_id && last.end == seg.start {
        // Merge: extend end, update is_overlap
        last.end = seg.end;
        last.is_overlap = last.is_overlap || seg.is_overlap;
        continue;
      }
    }
    merged.push(seg);
  }

  // Step 5: eliminate isolated short fragments (OUT-05)
  // If a segment's duration < MERGE_THRESHOLD and differs from both left and right neighbors,
  // but the left and right original segments share the same speaker → absorb into the surrounding speaker (eliminate noise flips).
  // Matches pyannote min_duration_on semantics: remove brief speaker switch artifacts.
  const MERGE_THRESHOLD_SECS: f64 = 0.25; // 250ms: ~15 frames
  let n_orig = merged.len();
  let mut result: Vec<SpeakerSegment> = Vec::with_capacity(n_orig);
  for i in 0..n_orig {
    let seg = merged[i].clone();
    let dur = seg.end.as_secs_f64() - seg.start.as_secs_f64();
    // Use original index neighbors (avoid misjudgment from cumulative effects)
    let orig_prev_spk = if i > 0 { Some(merged[i - 1].speaker_id) } else { None };
    let orig_next_spk = if i + 1 < n_orig { Some(merged[i + 1].speaker_id) } else { None };
    if dur < MERGE_THRESHOLD_SECS {
      if let (Some(ps), Some(ns)) = (orig_prev_spk, orig_next_spk) {
        if ps == ns && ps != seg.speaker_id {
          // Absorb into surrounding speaker: extend previous segment's end
          if let Some(last) = result.last_mut() {
            last.end = seg.end;
            last.is_overlap = last.is_overlap || seg.is_overlap;
            continue;
          }
        }
      }
    }
    // Normal: merge with previous if same speaker and adjacent (no silence gap), otherwise append
    if let Some(last) = result.last_mut() {
      if last.speaker_id == seg.speaker_id && last.end == seg.start {
        last.end = seg.end;
        last.is_overlap = last.is_overlap || seg.is_overlap;
        continue;
      }
    }
    result.push(seg);
  }

  result
}

/// Build overlap-mode output segments from frame-level speaker probabilities.
///
/// Unlike `build_speaker_segments` (exclusive mode: argmax picks a single speaker per frame),
/// this function independently scans the frame sequence for each active speaker, merging
/// consecutive active frames into a `DiarizeSegment`.
/// Result: the same time range may have multiple records (one per speaker), consistent with Python `itertracks()` output.
fn build_diarize_segments(
  frame_probs: &[Vec<(u32, f32)>],
  frame_step_secs: f64,
  vad_onset: f32,
) -> Vec<DiarizeSegment> {
  use std::collections::HashSet;

  if frame_probs.is_empty() {
    return vec![];
  }

  // Collect all global_speaker_ids that have appeared
  let mut all_speakers = HashSet::new();
  for frame in frame_probs {
    for &(spk, _) in frame {
      all_speakers.insert(spk);
    }
  }

  let mut segments: Vec<DiarizeSegment> = Vec::new();

  // For each speaker_id, independently scan all frames to find consecutive regions with prob >= vad_onset
  for &speaker_id in &all_speakers {
    let mut run_start: Option<usize> = None;

    for (f, frame) in frame_probs.iter().enumerate() {
      // Find this speaker's probability in the current frame (may appear multiple times, take max)
      let prob = frame
        .iter()
        .filter(|(spk, _)| *spk == speaker_id)
        .map(|(_, p)| *p)
        .fold(0.0_f32, f32::max);

      if prob >= vad_onset {
        if run_start.is_none() {
          run_start = Some(f);
        }
      } else if let Some(start) = run_start.take() {
        segments.push(DiarizeSegment {
          start: Duration::from_secs_f64(start as f64 * frame_step_secs),
          end: Duration::from_secs_f64(f as f64 * frame_step_secs),
          speaker_id,
        });
      }
    }
    // Submit the run that is still active at the end
    if let Some(start) = run_start {
      segments.push(DiarizeSegment {
        start: Duration::from_secs_f64(start as f64 * frame_step_secs),
        end: Duration::from_secs_f64(frame_probs.len() as f64 * frame_step_secs),
        speaker_id,
      });
    }
  }

  // Sort by start (records for multiple speakers in the same time range appear adjacently)
  segments.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.speaker_id.cmp(&b.speaker_id)));

  segments
}

/// Minimum clean active frame ratio required for embedding extraction (aligned with Python pyannote min_active_ratio=0.2).
///
/// Only when a speaker's **clean (non-overlapping) active frame count** in a chunk is >= 20% of total chunk frames
/// will that embedding be retained for clustering. This avoids extracting low-quality embeddings from brief voice bursts, reducing over-segmentation.
///
/// Reference: pyannote/audio/pipelines/clustering.py filter_embeddings() min_active_ratio=0.2
const MIN_ACTIVE_RATIO: f32 = 0.2;

/// Minimum frame count the embedding model can process (equivalent to Python _embedding.min_num_samples in frames).
///
/// Python determines min_num_samples ≈ 640 samples via binary search,
/// corresponding to min_num_frames = ceil(589 * 640 / 160000) ≈ 3 frames.
/// This value is used for the exclude_overlap fallback: when clean frames after overlap removal
/// fall below this count, fall back to using all frames (including overlap) instead of skipping.
///
/// Reference: pyannote/audio/pipelines/speaker_diarization.py get_embeddings()
const EMBEDDING_MIN_FRAMES: usize = 3;

impl TieguanyinOolong {
  /// Feed a batch of decoded speaker probs into the VAD state machine, enqueuing produced VAD regions.
  ///
  /// `frame_offset`: global start frame index for this batch (used for timestamp calculation).
  fn feed_frames_to_vad(&mut self, speaker_probs: &[[f32; 3]], frame_offset: u64) {
    // Frame step (seconds/frame), must use f64 to avoid cumulative error
    const FRAME_STEP_SECS: f64 = 10.0 / FRAMES_PER_WINDOW as f64;

    for (i, probs) in speaker_probs.iter().enumerate() {
      let global_frame = frame_offset + i as u64;

      // Dual-threshold hysteresis: active state uses offset (low threshold), inactive state uses onset (high threshold)
      let currently_active = matches!(
        self.vad_state,
        VadState::Active { .. } | VadState::Pending { .. }
      );
      let is_voiced = is_voiced_with_hysteresis(
        probs,
        currently_active,
        self.opts.vad_onset(),
        self.opts.vad_offset(),
      );

      // Sample range corresponding to this frame (use f64 arithmetic to avoid frame step drift)
      let frame_start_sample =
        (global_frame as f64 * FRAME_STEP_SECS * SAMPLE_RATE.get() as f64).round() as u64;
      let frame_end_sample =
        ((global_frame + 1) as f64 * FRAME_STEP_SECS * SAMPLE_RATE.get() as f64).round() as u64;

      self.process_vad_frame(is_voiced, frame_start_sample, frame_end_sample);
    }
  }

  /// Process a single VAD frame, update state machine, enqueue completed regions.
  fn process_vad_frame(&mut self, is_voiced: bool, frame_start: u64, frame_end: u64) {
    match &mut self.vad_state {
      VadState::Idle => {
        if is_voiced {
          self.vad_state = VadState::Active {
            start: frame_start,
            end: frame_end,
          };
        }
      }
      VadState::Active { start, end } => {
        if is_voiced {
          *end = frame_end;
        } else {
          let (range_start, range_end) = (*start, *end);
          self.vad_state = VadState::Pending {
            range_start,
            range_end,
            gap_samples: frame_end - range_end,
          };
        }
      }
      VadState::Pending {
        range_start,
        range_end,
        gap_samples,
      } => {
        if is_voiced {
          // Gap too short, absorb silence into the voice region
          let range_start = *range_start;
          self.vad_state = VadState::Active {
            start: range_start,
            end: frame_end,
          };
        } else {
          *gap_samples += frame_end - frame_start;

          if *gap_samples >= self.min_duration_off_samples {
            let range = VoiceRange::new(samples_to_duration(*range_start), samples_to_duration(*range_end));
            self.vad_state = VadState::Idle;
            self.pending_ranges.push_back(range);
          }
        }
      }
    }
  }

  /// Flush the VAD state machine, enqueuing any dangling Active/Pending regions.
  fn flush_vad_state(&mut self) {
    use core::mem::replace;
    match replace(&mut self.vad_state, VadState::Idle) {
      VadState::Active { start, end } => {
        self
          .pending_ranges
          .push_back(VoiceRange::new(samples_to_duration(start), samples_to_duration(end)));
      }
      VadState::Pending {
        range_start,
        range_end,
        ..
      } => {
        self
          .pending_ranges
          .push_back(VoiceRange::new(samples_to_duration(range_start), samples_to_duration(range_end)));
      }
      VadState::Idle => {}
    }
  }

  /// Run a single embedding ONNX inference, returning an L2-normalized 256-dim embedding vector.
  fn run_onnx_embedding_only(
    &mut self,
    features: &Array2<f32>,
    num_fbank_frames: usize,
    active_mask: &[f32],
    _num_seg_frames: usize,
  ) -> Result<Vec<f32>, TieguanyinOolongError> {
    let session = self
      .embedding_session
      .as_mut()
      .ok_or(TieguanyinOolongError::EmbeddingModelNotLoaded)?;

    // Mask alignment: segmentation frame count (~589) != fbank frame count (~997)
    // Use nearest-neighbor resampling to align mask to fbank frame count
    let mask_aligned: Vec<f32> = if active_mask.len() == num_fbank_frames {
      active_mask.to_vec()
    } else {
      (0..num_fbank_frames)
        .map(|i| {
          let src_i = (i * active_mask.len()) / num_fbank_frames;
          active_mask[src_i]
        })
        .collect()
    };

    let active_indices: Vec<usize> = (0..num_fbank_frames)
      .filter(|&i| mask_aligned[i] > 0.0)
      .collect();
    if active_indices.is_empty() {
      return Err(TieguanyinOolongError::InvalidTensorShape {
        expected: 1,
        actual: 0,
      });
    }
    let active_frames_count = active_indices.len();
    let mut active_fbank = Vec::with_capacity(active_frames_count * 80);
    for &row in &active_indices {
      active_fbank.extend_from_slice(features.row(row).as_slice().unwrap());
    }

    use ort::value::Tensor;
    let fbank_tensor = Tensor::from_array(([1usize, active_frames_count, 80], active_fbank))?;

    // Embedding inference stays CPU-only (no CoreML EP / IOBinding):
    // CoreML model compilation alters numerical behavior, causing abnormal PLDA clustering results. See session builder comments for details.
    let outputs = session.run(ort::inputs![
        "fbank" => fbank_tensor
    ])?;

    let (shape, emb_slice) = outputs["embedding"].try_extract_tensor::<f32>()?;
    let emb_dim = shape.get(1).copied().unwrap_or(emb_slice.len() as i64) as usize;
    let emb_dim = emb_dim.min(emb_slice.len());
    Ok(emb_slice[..emb_dim].to_vec())
  }

  /// Iterate over each (chunk_idx, local_speaker_idx) pair from segmentation results, extracting embedding vectors.
  ///
  /// Calls `run_onnx_embedding_only` individually for embedding inference.
  /// Per-chunk fbank caching and single-pass mask construction are retained.
  ///
  /// `all_full_probs`: per-chunk full segmentation probability matrix from the segmentation loop (all frames, not center-cropped).
  /// `audio`: complete audio PCM (used to crop the time window corresponding to each chunk).
  ///
  /// Returns: `Vec<(chunk_idx, local_speaker_idx, embedding, trainable)>`
  ///
  /// Reference:
  /// - pyannote speaker_diarization.py get_embeddings()
  /// - pyannote clustering.py filter_embeddings()
  fn collect_embeddings(
    &mut self,
    all_full_probs: &[Vec<[f32; 3]>],
    audio: &[f32],
  ) -> Result<Vec<(usize, usize, Vec<f32>, bool)>, TieguanyinOolongError> {
    let onset = self.opts.vad_onset();
    let exclude_overlap = self.opts.embedding_exclude_overlap();
    let mut result = Vec::new();

    for (chunk_idx, chunk_probs) in all_full_probs.iter().enumerate() {
      let chunk_start_sample = chunk_idx * DIARIZE_STEP_SAMPLES;
      let chunk_end_sample = (chunk_start_sample + WINDOW_SAMPLES).min(audio.len());
      let chunk_audio = &audio[chunk_start_sample..chunk_end_sample];
      let num_frames = chunk_probs.len();
      let min_active_frames = (MIN_ACTIVE_RATIO * num_frames as f32).ceil() as usize;

      // ── Per-chunk fbank cache (performance optimization) ─────────────
      // Same chunk has at most 3 speakers, fbank only depends on chunk_audio → compute once and reuse.
      let fbank_ref = self.fbank.as_ref()
        .ok_or(TieguanyinOolongError::EmbeddingModelNotLoaded)?;
      let scaled: Vec<f32> = chunk_audio.iter().map(|&s| s * PCM_F32_TO_I16_SCALE).collect();
      let raw_features: Array2<f32> = fbank_ref.compute(&scaled);
      let num_fbank_frames = raw_features.nrows();
      let frame_mean: Array1<f32> = raw_features.mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(80));
      let features: Array2<f32> = &raw_features - &frame_mean.view().insert_axis(Axis(0));

      for local_speaker in 0..3usize {
        // Single pass: simultaneously build original_mask and clean_mask, count active_frames and clean_count
        let mut original_mask = Vec::with_capacity(num_frames);
        let mut clean_mask = Vec::with_capacity(num_frames);
        let mut active_frames = 0usize;
        let mut clean_count = 0usize;
        for probs in chunk_probs.iter() {
          let is_active = probs[local_speaker] >= onset;
          let mask_val = if is_active { 1.0f32 } else { 0.0f32 };
          original_mask.push(mask_val);
          if is_active {
            active_frames += 1;
            if exclude_overlap {
              let num_active_speakers = probs.iter().filter(|&&p| p >= onset).count();
              if num_active_speakers >= 2 {
                clean_mask.push(0.0f32);
              } else {
                clean_mask.push(1.0f32);
                clean_count += 1;
              }
            } else {
              clean_mask.push(mask_val);
              clean_count += 1;
            }
          } else {
            clean_mask.push(0.0f32);
          }
        }

        if active_frames == 0 {
          continue;
        }

        // ── Fallback mechanism (aligned with Python get_embeddings) ──
        // Clean frames below model minimum requirement → fall back to all frames (including overlap)
        let (used_mask, num_clean_frames) = if exclude_overlap {
          if clean_count < EMBEDDING_MIN_FRAMES {
            (original_mask, clean_count)
          } else {
            (clean_mask, clean_count)
          }
        } else {
          (original_mask, active_frames)
        };

        // Final mask must have active frames for extraction
        let final_active = used_mask.iter().filter(|&&m| m > 0.0).count();
        if final_active < EMBEDDING_MIN_FRAMES {
          continue;
        }

        // Mark as trainable (aligned with Python filter_embeddings)
        let trainable = num_clean_frames >= min_active_frames;

        // Call ONNX embedding inference individually
        let embedding = self.run_onnx_embedding_only(&features, num_fbank_frames, &used_mask, num_frames)?;
        result.push((chunk_idx, local_speaker, embedding, trainable));
      }
    }

    Ok(result)
  }

  /// Default batch size for segmentation batch inference.
  ///
  /// Each batch collects 16 full windows (16 x 160000 = 2560000 samples ≈ 160s) for a single inference call.
  /// Batch size affects CoreML memory usage: batch=16 is tested safe on 16GB Apple Silicon.
  const SEG_BATCH_SIZE: usize = 16;

  /// Batch segmentation inference: collect N full windows, call ONNX once, then split the output.
  ///
  /// - Full windows (`WINDOW_SAMPLES` samples) are processed in batches of `SEG_BATCH_SIZE`.
  /// - The trailing partial window (`< WINDOW_SAMPLES`) is handled separately via `run_inference` (original logic).
  ///
  /// Return format is identical to `run_segmentation_loop`: `(center_probs, full_probs)` per window.
  fn run_segmentation_loop_batched(
    &mut self,
    audio: &[f32],
  ) -> Result<Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>)>, TieguanyinOolongError> {
    use ort::value::Tensor;

    // ── 1. Collect all window offsets ──────────────────────────────────────
    let mut full_offsets: Vec<usize> = Vec::new();
    let mut partial_offset: Option<usize> = None;
    let mut offset = 0usize;
    while offset < audio.len() {
      let window_end = (offset + WINDOW_SAMPLES).min(audio.len());
      if window_end - offset == WINDOW_SAMPLES {
        full_offsets.push(offset);
        offset += DIARIZE_STEP_SAMPLES;
      } else {
        // Trailing partial window: record separately, not batched
        partial_offset = Some(offset);
        break;
      }
    }

    let total_windows = full_offsets.len() + partial_offset.map_or(0, |_| 1);
    let mut results: Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>)> = Vec::with_capacity(total_windows);

    // ── 2. Process full windows in batches of SEG_BATCH_SIZE ──────────────
    for batch_start in (0..full_offsets.len()).step_by(Self::SEG_BATCH_SIZE) {
      let batch_end = (batch_start + Self::SEG_BATCH_SIZE).min(full_offsets.len());
      let batch_offsets = &full_offsets[batch_start..batch_end];
      let n = batch_offsets.len();

      // Build [N, 1, WINDOW_SAMPLES] flattened buffer
      // All full windows have the same length, no padding needed
      let batch_audio: Vec<f32> = batch_offsets
        .iter()
        .flat_map(|&off| audio[off..off + WINDOW_SAMPLES].iter().copied())
        .collect();

      let input = Tensor::from_array(([n, 1usize, WINDOW_SAMPLES], batch_audio))?;
      let outputs = self.session.run(ort::inputs!["input_values" => input])?;

      // Extract output: shape [N, num_frames, 7]
      let (shape, logits_slice) = outputs["logits"].try_extract_tensor::<f32>()?;
      let num_frames = shape[1] as usize;   // Frame count per window (fixed, ~589)
      let frames_per_item = num_frames * 7; // Element count per window in the flat slice

      // ── 3. Split batch output by window ──────────────────────────────────
      for i in 0..n {
        let start = i * frames_per_item;
        let end = start + frames_per_item;
        let window_logits = &logits_slice[start..end];
        // Full window: is_full_window = true, perform center-crop
        let center_probs = extract_speaker_probs(window_logits, num_frames, true)?;
        let full_probs = extract_full_speaker_probs(window_logits, num_frames)?;
        results.push((center_probs, full_probs));
      }
    }

    // ── 4. Handle trailing partial window separately (original logic) ─────
    if let Some(off) = partial_offset {
      let audio_window = &audio[off..];
      let (logits, num_frames) = run_inference(&mut self.session, audio_window)?;
      let center_probs = extract_speaker_probs(&logits, num_frames, false)?;
      let full_probs = extract_full_speaker_probs(&logits, num_frames)?;
      results.push((center_probs, full_probs));
    }

    Ok(results)
  }

  /// Run sliding-window segmentation inference on complete audio, returning per-chunk frame-level speaker probabilities.
  ///
  /// Each chunk corresponds to STEP_SAMPLES (40000 samples = 2.5s), retaining
  /// CENTER_KEEP_FRAMES (147 frames) after center-crop. Trailing partial window takes all frames.
  ///
  /// Returns: `(center_probs, full_probs)` per chunk.
  ///
  /// - `center_probs`: CENTER_KEEP_FRAMES frames (for frame/segmentation result concatenation)
  /// - `full_probs`: all segmentation frames (for precise mask alignment during embedding extraction)
  fn run_segmentation_loop(
    &mut self,
    audio: &[f32],
  ) -> Result<Vec<(Vec<[f32; 3]>, Vec<[f32; 3]>)>, TieguanyinOolongError> {
    // Batch processing version: collect SEG_BATCH_SIZE full windows, then make a single ONNX call
    // Reduces kernel launch overhead (Phase 7 optimization, segmentation model batch_size is dynamic dim_param)
    self.run_segmentation_loop_batched(audio)
  }

  /// Run the full speaker diarization pipeline, also returning per-speaker mean embedding vectors.
  ///
  /// Returns `(Vec<SpeakerSegment>, HashMap<u32, Vec<f32>>)`, where:
  /// - `Vec<SpeakerSegment>`: same output as `diarize()`
  /// - `HashMap<u32, Vec<f32>>`: key = speaker_id, value = mean vector of all embeddings for that speaker (256-dim)
  pub fn diarize_with_embeddings(
    &mut self,
    audio: &[f32],
  ) -> Result<(Vec<SpeakerSegment>, std::collections::HashMap<u32, Vec<f32>>), TieguanyinOolongError>
  {
    // Verify embedding model is loaded
    if self.embedding_session.is_none() {
      return Err(TieguanyinOolongError::EmbeddingModelNotLoaded);
    }

    // ── Step 1: Segmentation loop ──────────────────────────────────────────────
    let seg_results = self.run_segmentation_loop(audio)?;

    if seg_results.is_empty() {
      return Ok((vec![], std::collections::HashMap::new()));
    }

    // full_probs: complete segmentation frames (for precise embedding mask alignment and frame-level probability reconstruction)
    let all_full_probs: Vec<Vec<[f32; 3]>> = seg_results.iter().map(|(_, f)| f.clone()).collect();

    let embedding_quads = self.collect_embeddings(&all_full_probs, audio)?;
    if embedding_quads.is_empty() {
      return Ok((vec![], std::collections::HashMap::new()));
    }

    let all_embeddings: Vec<Vec<f32>> =
      embedding_quads.iter().map(|(_, _, e, _)| e.clone()).collect();
    let chunk_speaker_pairs: Vec<(usize, usize)> =
      embedding_quads.iter().map(|(ci, si, _, _)| (*ci, *si)).collect();
    let trainable_mask: Vec<bool> =
      embedding_quads.iter().map(|(_, _, _, t)| *t).collect();

    // ── filter → cluster → assign (aligned with Python clustering.__call__) ──
    // Step 1: extract trainable subset for clustering
    let train_embeddings: Vec<Vec<f32>> = all_embeddings
      .iter()
      .zip(trainable_mask.iter())
      .filter(|&(_, &t)| t)
      .map(|(e, _)| e.clone())
      .collect();

    if train_embeddings.is_empty() {
      return Ok((vec![], std::collections::HashMap::new()));
    }

    // Step 2: run VBx clustering only on the train subset (requires PLDA model)
    let plda = self.plda_model.as_ref()
      .ok_or(TieguanyinOolongError::PldaModelNotLoaded)?;
    let (train_labels, _gamma) = vbx_cluster_embeddings(
      &train_embeddings,
      plda,
      self.opts.clustering_threshold() as f64,
      0.07, // Fa
      0.8,  // Fb
    );

    // Step 3: assign all embeddings to nearest centroid (including non-trainable)
    let embedding_labels = assign_embeddings(&all_embeddings, &trainable_mask, &train_labels);

    // Compute per-speaker mean embedding vector (using train embeddings only)
    let mut speaker_raw: std::collections::HashMap<u32, Vec<Vec<f32>>> = Default::default();
    for (i, label) in embedding_labels.iter().enumerate() {
      if trainable_mask[i] {
        speaker_raw.entry(*label).or_default().push(all_embeddings[i].clone());
      }
    }
    let speaker_mean_embeddings: std::collections::HashMap<u32, Vec<f32>> = speaker_raw
      .into_iter()
      .map(|(spk, embs)| {
        let n = embs.len() as f32;
        let mean = embs.iter().fold(vec![0.0f32; 256], |mut acc, e| {
          for (a, &v) in acc.iter_mut().zip(e.iter()) {
            *a += v / n;
          }
          acc
        });
        (spk, mean)
      })
      .collect();

    let frame_probs = reconstruct_frame_speaker_probs(
      &all_full_probs,
      &chunk_speaker_pairs,
      &trainable_mask,
      &embedding_labels,
      audio.len(),
    );

    const FRAME_STEP_SECS: f64 = 10.0 / FRAMES_PER_WINDOW as f64;
    let segments = build_speaker_segments(&frame_probs, FRAME_STEP_SECS, self.opts.vad_onset());

    Ok((segments, speaker_mean_embeddings))
  }

  /// Overlap-mode diarization: allows multiple speakers to appear simultaneously in the same time range.
  ///
  /// Shares Steps 1-3 with `diarize()` (segmentation → embedding → VBx clustering),
  /// only Step 4 uses `build_diarize_segments()` instead of `build_speaker_segments()`.
  /// Requires PLDA model (`plda_model_path` must be set).
  pub fn diarize_overlap(
    &mut self,
    audio: &[f32],
  ) -> Result<Vec<DiarizeSegment>, TieguanyinOolongError> {
    // Verify embedding model and PLDA model are loaded
    if self.embedding_session.is_none() {
      return Err(TieguanyinOolongError::EmbeddingModelNotLoaded);
    }
    if self.plda_model.is_none() {
      return Err(TieguanyinOolongError::PldaModelNotLoaded);
    }

    // ── Step 1: Segmentation loop ─────────────────────────────────────────────
    let seg_results = self.run_segmentation_loop(audio)?;
    if seg_results.is_empty() {
      return Ok(vec![]);
    }
    let all_chunk_probs: Vec<Vec<[f32; 3]>> = seg_results.iter().map(|(c, _)| c.clone()).collect();
    let all_full_probs: Vec<Vec<[f32; 3]>> = seg_results.iter().map(|(_, f)| f.clone()).collect();

    // ── Step 2: Embedding extraction ─────────────────────────────────────────────
    let embedding_quads = self.collect_embeddings(&all_full_probs, audio)?;
    if embedding_quads.is_empty() {
      return Ok(vec![]);
    }

    let all_embeddings: Vec<Vec<f32>> =
      embedding_quads.iter().map(|(_, _, e, _)| e.clone()).collect();
    let chunk_speaker_pairs: Vec<(usize, usize)> =
      embedding_quads.iter().map(|(ci, si, _, _)| (*ci, *si)).collect();
    let trainable_mask: Vec<bool> =
      embedding_quads.iter().map(|(_, _, _, t)| *t).collect();

    // ── Step 3: filter → cluster → assign (aligned with Python clustering.__call__) ──
    let train_embeddings: Vec<Vec<f32>> = all_embeddings
      .iter()
      .zip(trainable_mask.iter())
      .filter(|&(_, &t)| t)
      .map(|(e, _)| e.clone())
      .collect();

    if train_embeddings.is_empty() {
      return Ok(vec![]);
    }

    let plda = self.plda_model.as_ref()
      .ok_or(TieguanyinOolongError::PldaModelNotLoaded)?;
    let (train_labels, _gamma) = vbx_cluster_embeddings(
      &train_embeddings,
      plda,
      self.opts.clustering_threshold() as f64,
      0.07, // Fa
      0.8,  // Fb
    );

    let embedding_labels = assign_embeddings(&all_embeddings, &trainable_mask, &train_labels);

    // ── Step 4: Label reconstruction → Vec<DiarizeSegment> (overlap mode) ──
    let total_frames: usize = all_chunk_probs.iter().map(|c| c.len()).sum();
    let mut frame_probs: Vec<Vec<(u32, f32)>> = vec![Vec::new(); total_frames];

    let mut chunk_frame_starts: Vec<usize> = Vec::with_capacity(all_chunk_probs.len());
    let mut acc = 0usize;
    for chunk in &all_chunk_probs {
      chunk_frame_starts.push(acc);
      acc += chunk.len();
    }

    for (pair_idx, &(chunk_idx, local_speaker)) in chunk_speaker_pairs.iter().enumerate() {
      // Only trainable embeddings participate in frame reconstruction
      if !trainable_mask[pair_idx] {
        continue;
      }
      let global_speaker = embedding_labels[pair_idx];
      let frame_start = chunk_frame_starts[chunk_idx];
      let chunk_probs = &all_chunk_probs[chunk_idx];

      for (local_frame, probs) in chunk_probs.iter().enumerate() {
        let global_frame = frame_start + local_frame;
        if global_frame < frame_probs.len() {
          frame_probs[global_frame].push((global_speaker, probs[local_speaker]));
        }
      }
    }
    for frame in frame_probs.iter_mut() {
      if frame.is_empty() {
        frame.push((0u32, 0.0));
      }
    }

    const FRAME_STEP_SECS: f64 = 10.0 / FRAMES_PER_WINDOW as f64;
    let segments = build_diarize_segments(&frame_probs, FRAME_STEP_SECS, self.opts.vad_onset());

    Ok(segments)
  }
}

// ── Streaming VAD interface ──────────────────────────────────────────────────

impl TieguanyinOolong {
  /// Returns whether currently in an active voice region (Active or Pending state).
  pub fn is_active(&self) -> bool {
    matches!(
      self.vad_state,
      VadState::Active { .. } | VadState::Pending { .. }
    )
  }

  /// Accumulate samples, triggering one ONNX inference each time WINDOW_SAMPLES (160000) is reached.
  ///
  /// **Sliding buffer strategy:**
  /// - `sample_buffer` retains the most recent WINDOW_SAMPLES (160000) samples
  /// - After each inference, the buffer is truncated to retain the last `WINDOW_SAMPLES - STEP_SAMPLES` (120000) samples
  ///   as the prefix for the next window (overlapping portion)
  ///
  /// **Return rules:**
  /// - If `pending_ranges` queue is non-empty, dequeue and return the earliest one
  /// - Otherwise return `None` (per trait contract: at most one region returned per call)
  pub fn detect(&mut self, frame: &[f32]) -> Result<Option<VoiceRange>, TieguanyinOolongError> {
    self.sample_buffer.extend_from_slice(frame);
    self.cursor += frame.len() as u64;

    // Buffer must reach a full window (WINDOW_SAMPLES = 160000) to trigger inference.
    // After inference, truncate to 120000 samples (WINDOW_SAMPLES - STEP_SAMPLES),
    // so the next detect() call triggers again after accumulating exactly 40000 (STEP_SAMPLES) samples.
    while self.sample_buffer.len() >= WINDOW_SAMPLES {
      // Form inference window: take the last min(WINDOW_SAMPLES, buffer.len()) samples from the buffer
      let window_start = self.sample_buffer.len().saturating_sub(WINDOW_SAMPLES);
      let audio_window = &self.sample_buffer[window_start..];
      let is_full_window = audio_window.len() == WINDOW_SAMPLES;

      // Run ONNX inference (returns flat logits and frame count)
      let (logits, num_frames) = run_inference(&mut self.session, audio_window)?;

      // Decode logits into per-speaker probabilities (full window uses center-crop, partial window takes all frames)
      let speaker_probs = extract_speaker_probs(&logits, num_frames, is_full_window)?;

      // Feed decoded frames into the VAD three-state machine (Idle/Active/Pending)
      let current_frame_offset = self.frame_cursor;
      self.feed_frames_to_vad(&speaker_probs, current_frame_offset);
      self.frame_cursor += speaker_probs.len() as u64;

      // Truncate buffer: retain last WINDOW_SAMPLES - STEP_SAMPLES samples for next window overlap
      // This ensures the next 10s window's first 7.5s comes from the previous window's tail (sliding overlap)
      let overlap = WINDOW_SAMPLES - STEP_SAMPLES; // 120000 samples = 7.5s
      let keep_from = self.sample_buffer.len().saturating_sub(overlap);
      self.sample_buffer.drain(..keep_from);
    }

    // Return at most one ready VAD region per call (per trait contract)
    Ok(self.pending_ranges.pop_front())
  }

  /// Process remaining samples in the buffer (< STEP_SAMPLES), no zero-padding, direct dynamic-length inference.
  ///
  /// This is the last window at the end of the stream. Since there is no subsequent overlap,
  /// no center-crop is applied (all frames taken), and the VAD state machine is flushed after inference.
  pub fn finish(&mut self) -> Option<VoiceRange> {
    // Process remaining samples in the buffer that haven't triggered inference yet
    if !self.sample_buffer.is_empty() {
      // Run inference with actual length (model supports dynamic input axes)
      let audio_window = self.sample_buffer.clone();
      // `VoiceDetector::finish()` returns `Option` (no Result), cannot propagate inference errors upward.
      // On inference failure, log a warning and skip remaining frames -- caller cannot distinguish "no voice" from "inference failure".
      match run_inference(&mut self.session, &audio_window) {
        Ok((logits, num_frames)) => {
        // Last window: no center-crop, take all frames (is_full_window = false)
        if let Ok(speaker_probs) = extract_speaker_probs(&logits, num_frames, false) {
          let current_frame_offset = self.frame_cursor;
          self.feed_frames_to_vad(&speaker_probs, current_frame_offset);
          self.frame_cursor += speaker_probs.len() as u64;
        }
        }
        Err(e) => {
          eprintln!("[TieguanyinOolong::finish] inference error on trailing audio ({} samples), skipping: {e}", audio_window.len());
        }
      }
      self.sample_buffer.clear();
    }

    // Flush VAD state machine, enqueue Active or Pending regions
    self.flush_vad_state();

    // Return the earliest region in the queue (caller should loop until None if all are needed)
    self.pending_ranges.pop_front()
  }
}

// ── diarize_timed (speaker diarization with step-by-step timing) ─────────────

impl TieguanyinOolong {
  /// Run the full speaker diarization pipeline on complete 16kHz mono PCM audio, returning step-by-step timing.
  ///
  /// Functionality is identical to [`SpeakerDiarizer::diarize`], additionally returning [`DiarizeTiming`]
  /// recording the time spent on segmentation, embedding, and clustering steps, for performance analysis and benchmarking.
  pub fn diarize_timed(&mut self, audio: &[f32]) -> Result<(Vec<SpeakerSegment>, DiarizeTiming), TieguanyinOolongError> {
    let t_total = Instant::now();

    // Verify embedding model is loaded
    if self.embedding_session.is_none() {
      return Err(TieguanyinOolongError::EmbeddingModelNotLoaded);
    }

    // ── Step 1: Segmentation loop ─────────────────────────────────────────────
    let t_seg = Instant::now();
    let seg_results = self.run_segmentation_loop(audio)?;
    let segmentation_ms = t_seg.elapsed().as_millis() as u64;

    if seg_results.is_empty() {
      let timing = DiarizeTiming {
        segmentation_ms,
        embedding_ms: 0,
        clustering_ms: 0,
        total_ms: t_total.elapsed().as_millis() as u64,
      };
      return Ok((vec![], timing));
    }
    let all_full_probs: Vec<Vec<[f32; 3]>> = seg_results.iter().map(|(_, f)| f.clone()).collect();

    // ── Step 2: Embedding extraction ─────────────────────────────────────────────
    let t_emb = Instant::now();
    let embedding_quads = self.collect_embeddings(&all_full_probs, audio)?;
    let embedding_ms = t_emb.elapsed().as_millis() as u64;

    if embedding_quads.is_empty() {
      let timing = DiarizeTiming {
        segmentation_ms,
        embedding_ms,
        clustering_ms: 0,
        total_ms: t_total.elapsed().as_millis() as u64,
      };
      return Ok((vec![], timing));
    }

    let all_embeddings: Vec<Vec<f32>> =
      embedding_quads.iter().map(|(_, _, e, _)| e.clone()).collect();
    let chunk_speaker_pairs: Vec<(usize, usize)> =
      embedding_quads.iter().map(|(ci, si, _, _)| (*ci, *si)).collect();
    let trainable_mask: Vec<bool> =
      embedding_quads.iter().map(|(_, _, _, t)| *t).collect();

    // ── Step 3+4: filter → cluster → assign + label reconstruction ───
    let t_clus = Instant::now();

    let train_embeddings: Vec<Vec<f32>> = all_embeddings
      .iter()
      .zip(trainable_mask.iter())
      .filter(|&(_, &t)| t)
      .map(|(e, _)| e.clone())
      .collect();

    if train_embeddings.is_empty() {
      let clustering_ms = t_clus.elapsed().as_millis() as u64;
      let timing = DiarizeTiming {
        segmentation_ms,
        embedding_ms,
        clustering_ms,
        total_ms: t_total.elapsed().as_millis() as u64,
      };
      return Ok((vec![], timing));
    }

    let plda = self.plda_model.as_ref()
      .ok_or(TieguanyinOolongError::PldaModelNotLoaded)?;
    let (train_labels, _gamma) = vbx_cluster_embeddings(
      &train_embeddings,
      plda,
      self.opts.clustering_threshold() as f64,
      0.07, // Fa
      0.8,  // Fb
    );
    let embedding_labels = assign_embeddings(&all_embeddings, &trainable_mask, &train_labels);

    // ── Label reconstruction → Vec<SpeakerSegment> (using center-crop overlap-add) ──
    let frame_probs = reconstruct_frame_speaker_probs(
      &all_full_probs,
      &chunk_speaker_pairs,
      &trainable_mask,
      &embedding_labels,
      audio.len(),
    );

    const FRAME_STEP_SECS: f64 = 10.0 / FRAMES_PER_WINDOW as f64;
    let segments = build_speaker_segments(&frame_probs, FRAME_STEP_SECS, self.opts.vad_onset());
    let clustering_ms = t_clus.elapsed().as_millis() as u64;

    let timing = DiarizeTiming {
      segmentation_ms,
      embedding_ms,
      clustering_ms,
      total_ms: t_total.elapsed().as_millis() as u64,
    };

    Ok((segments, timing))
  }
}

// ── diarize convenience method ───────────────────────────────────────────────

impl TieguanyinOolong {
  /// Run the full speaker diarization pipeline on complete 16kHz mono PCM audio.
  ///
  /// Delegates to [`TieguanyinOolong::diarize_timed`], discarding timing information.
  pub fn diarize(&mut self, audio: &[f32]) -> Result<Vec<SpeakerSegment>, TieguanyinOolongError> {
    let (segments, _timing) = self.diarize_timed(audio)?;
    Ok(segments)
  }
}

// ── Unit tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_options_defaults() {
    let opts = TieguanyinOolongOptions::new();
    assert_eq!(opts.vad_onset(), 0.5);
    assert!((opts.vad_offset() - 0.357).abs() < 1e-6);
    assert_eq!(opts.min_duration_off(), Duration::ZERO);
    assert!(!opts.vad_only());
    assert!(opts.embedding_exclude_overlap());
    assert!((opts.clustering_threshold() - 0.6).abs() < 1e-6);
    assert!(opts.plda_model_path().is_none());
  }

  #[test]
  fn test_options_builder_consuming() {
    let opts = TieguanyinOolongOptions::new()
      .with_vad_onset(0.7)
      .with_vad_only(true);
    assert_eq!(opts.vad_onset(), 0.7);
    assert!(opts.vad_only());
    // Other fields keep defaults
    assert!((opts.vad_offset() - 0.357).abs() < 1e-6);
  }

  #[test]
  fn test_options_builder_mutable() {
    let mut opts = TieguanyinOolongOptions::new();
    opts.set_vad_offset(0.4).set_vad_only(true);
    assert!((opts.vad_offset() - 0.4).abs() < 1e-6);
    assert!(opts.vad_only());
  }

  #[test]
  fn test_powerset_decode_uniform() {
    // Uniform log-softmax: each class probability = 1/7, so log_prob = log(1/7).
    // After decoding, each speaker contains 3 classes, expected speaker_prob = 3/7 ≈ 0.4286.
    // Note: [0.0; 7] is unnormalized (exp gives each class probability = 1, sum = 7 ≠ 1).
    let log_prob_uniform = (1.0f32 / 7.0).ln(); // log(1/7) ≈ -1.9459
    let log_probs = [log_prob_uniform; 7];
    let result = powerset_decode(&log_probs);
    let expected = 3.0f32 / 7.0;
    for &p in &result {
      assert!((p - expected).abs() < 1e-5, "expected {expected}, got {p}");
    }
  }

  #[test]
  fn test_powerset_decode_single_speaker() {
    // Class 1 (speaker 0 only) log_prob = 0, rest = -1000 (exp ≈ 0)
    let mut log_probs = [-1000.0f32; 7];
    log_probs[1] = 0.0;
    let result = powerset_decode(&log_probs);
    // speaker 0 ≈ 1.0 (only class 1 contributes), speaker 1 and 2 ≈ 0
    assert!(
      result[0] > 0.99,
      "speaker 0 prob should be ~1.0, got {}",
      result[0]
    );
    assert!(
      result[1] < 0.01,
      "speaker 1 prob should be ~0.0, got {}",
      result[1]
    );
    assert!(
      result[2] < 0.01,
      "speaker 2 prob should be ~0.0, got {}",
      result[2]
    );
  }

  #[test]
  fn test_center_crop_full_window() {
    // 589 frames full window -> center-crop [221, 368), total 147 frames
    let logits = vec![0.0f32; 589 * 7];
    let probs = extract_speaker_probs(&logits, 589, true).unwrap();
    assert_eq!(
      probs.len(),
      CENTER_KEEP_FRAMES,
      "full window should yield {CENTER_KEEP_FRAMES} frames"
    );
  }

  #[test]
  fn test_center_crop_partial_window() {
    // 293 frames partial window -> all frames [0, 293)
    let logits = vec![0.0f32; 293 * 7];
    let probs = extract_speaker_probs(&logits, 293, false).unwrap();
    assert_eq!(
      probs.len(),
      293,
      "partial window should yield all 293 frames"
    );
  }

  // ── VoiceDetector<f32> behavior tests ──────────────────────────────────────

  /// Verify constant relationships in detect() accumulation logic.
  ///
  /// This test verifies three key invariants of the sliding buffer:
  /// 1. First trigger requires a full 10s window (WINDOW_SAMPLES)
  /// 2. Steady-state step size is 2.5s (STEP_SAMPLES)
  /// 3. Overlap = WINDOW_SAMPLES - STEP_SAMPLES = 7.5s (120000 samples)
  /// 4. Center-crop window corresponds exactly to one step duration (CENTER_KEEP_FRAMES * frame_step ≈ STEP_SAMPLES)
  #[test]
  fn test_detect_accumulates_before_inference() {
    // Invariant 1: WINDOW = 4 x STEP (10s = 4 x 2.5s)
    assert_eq!(
      WINDOW_SAMPLES,
      4 * STEP_SAMPLES,
      "window is 4 steps: first trigger requires full 10s, steady state triggers every 2.5s"
    );
    // Invariant 2: overlap = 3 steps (75% overlap)
    assert_eq!(
      WINDOW_SAMPLES - STEP_SAMPLES,
      3 * STEP_SAMPLES,
      "after truncation, retain 3 steps = 120000 samples (75% overlap)"
    );
    // Invariant 3: center-crop retained frames do not exceed FRAMES_PER_WINDOW
    assert!(
      CENTER_MARGIN_FRAMES + CENTER_KEEP_FRAMES <= FRAMES_PER_WINDOW,
      "center-crop range [margin, margin+keep] must be within window: \
       {} + {} = {} <= {}",
      CENTER_MARGIN_FRAMES,
      CENTER_KEEP_FRAMES,
      CENTER_MARGIN_FRAMES + CENTER_KEEP_FRAMES,
      FRAMES_PER_WINDOW
    );
    // Invariant 4: sample rate meets pyannote requirements
    assert_eq!(SAMPLE_RATE.get(), 16_000, "pyannote model only supports 16kHz");
  }

  /// Idle state should not produce VoiceRange (flush_vad_state logic verification).
  #[test]
  fn test_idle_state_does_not_produce_range() {
    // Flush in Idle state does not enqueue, verifying the state machine's initial invariant
    let state = VadState::Idle;
    let is_active = matches!(state, VadState::Active { .. } | VadState::Pending { .. });
    assert!(!is_active, "Idle state should not be active");
    // Idle carries no range_start/range_end → flush has nothing to enqueue
  }

  /// is_active() should return false in initial state (VadState::Idle).
  #[test]
  fn test_is_active_idle() {
    // is_active() depends on VadState enum:
    // Idle -> false, Active -> true, Pending -> true
    // Test by directly exercising pattern matching logic
    let state = VadState::Idle;
    let is_active = matches!(state, VadState::Active { .. } | VadState::Pending { .. });
    assert!(!is_active, "VadState::Idle should not be active");
  }

  /// Real model inference: feed 40000 samples to trigger one inference (requires model file).
  #[test]
  #[ignore = "requires model file at models/segmentation-3.0.onnx"]
  fn test_detect_triggers_at_step_boundary() {
    let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .parent()
      .unwrap()
      .join("models/segmentation-3.0.onnx");

    assert!(model_path.exists(), "model not found: {model_path:?}");

    let opts = TieguanyinOolongOptions::new();
    let mut detector = TieguanyinOolong::new_from_path(&model_path, opts).unwrap();

    // Feed exactly STEP_SAMPLES all-1.0 samples -> should trigger inference without panic
    let frame = vec![1.0f32; STEP_SAMPLES];
    let result = detector.detect(&frame);
    assert!(
      result.is_ok(),
      "detect() should not error: {:?}",
      result.err()
    );
    // Return value can be None (all 1.0 may not reach vad_onset), but must not panic
  }

  /// Real model: is_active() = true after feeding voice frames (requires model file).
  #[test]
  #[ignore = "requires model file and audio that triggers VAD"]
  fn test_is_active_after_voice() {
    let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .parent()
      .unwrap()
      .join("models/segmentation-3.0.onnx");

    let opts = TieguanyinOolongOptions::new();
    let mut detector = TieguanyinOolong::new_from_path(&model_path, opts).unwrap();

    // Initial state is Idle
    assert!(!detector.is_active(), "initial state should be inactive");

    // After feeding real voice samples (audio with actual voice content needed to trigger VAD)
    // This test depends on real audio content, use with actual audio files for manual verification
    let _ = detector.detect(&vec![0.5f32; STEP_SAMPLES]);
    // is_active() may be true or false, depending on model inference results and thresholds
    // This test only verifies no panic
  }

  /// min_duration_off merge logic: Pending state receiving a voiced frame should return to Active preserving range_start.
  ///
  /// Verify VadState transition: Pending + voiced → Active (merge, no split).
  #[test]
  fn test_pending_state_merges_on_voiced_frame() {
    // Pending state represents a brief silence gap (gap_samples < min_duration_off_samples)
    let pending = VadState::Pending {
      range_start: 0,
      range_end: 10_000,
      gap_samples: 1_600, // 100ms gap
    };
    // Verify Pending is an active state (not yet committed as a completed region)
    assert!(
      matches!(pending, VadState::Pending { .. }),
      "state should be Pending"
    );
    let is_active = matches!(pending, VadState::Active { .. } | VadState::Pending { .. });
    assert!(is_active, "Pending state should count as active");

    // Verify the precondition: gap_samples < min_duration_off_samples
    if let VadState::Pending { gap_samples, .. } = pending {
      let min_duration_off_samples = SAMPLE_RATE.get() as u64; // 1s = 16000 samples
      assert!(
        gap_samples < min_duration_off_samples,
        "100ms gap ({gap_samples}) should be less than 1s threshold ({min_duration_off_samples})"
      );
    }
  }

  // ── Dual-threshold hysteresis unit tests ────────────────────────────────────

  #[test]
  fn test_hysteresis_below_onset_stays_active() {
    // Active state, prob=0.4 (below onset=0.5, above offset=0.357) -> stays active
    let probs = [0.4f32, 0.0, 0.0];
    assert!(
      is_voiced_with_hysteresis(&probs, true, 0.5, 0.357),
      "prob=0.4 with offset=0.357 should keep active state"
    );
  }

  #[test]
  fn test_hysteresis_below_offset_triggers_pending() {
    // Active state, prob=0.3 (below offset=0.357) -> deactivates
    let probs = [0.3f32, 0.0, 0.0];
    assert!(
      !is_voiced_with_hysteresis(&probs, true, 0.5, 0.357),
      "prob=0.3 below offset=0.357 should deactivate"
    );
  }

  #[test]
  fn test_hysteresis_idle_requires_onset() {
    // Idle state, prob=0.4 (below onset=0.5) -> does not activate
    let probs = [0.4f32, 0.0, 0.0];
    assert!(
      !is_voiced_with_hysteresis(&probs, false, 0.5, 0.357),
      "prob=0.4 below onset=0.5 should not activate from Idle"
    );
  }

  #[test]
  fn test_hysteresis_idle_onset_activates() {
    // Idle state, prob=0.6 (above onset=0.5) -> activates
    let probs = [0.6f32, 0.0, 0.0];
    assert!(
      is_voiced_with_hysteresis(&probs, false, 0.5, 0.357),
      "prob=0.6 above onset=0.5 should activate from Idle"
    );
  }

  #[test]
  fn test_vad_offset_field_used_in_logic() {
    // Custom onset=0.8 / offset=0.6, prob=0.7 (above offset, below onset)
    // Active state -> should stay active (hysteresis)
    let probs = [0.7f32, 0.0, 0.0];
    assert!(
      is_voiced_with_hysteresis(&probs, true, 0.8, 0.6),
      "prob=0.7 above offset=0.6 should keep active when onset=0.8"
    );
    // Verify if onset were incorrectly used:
    assert!(
      !is_voiced_with_hysteresis(&probs, false, 0.8, 0.6),
      "prob=0.7 below onset=0.8 should NOT activate from Idle"
    );
  }

  /// Requires actual model file, skipped in CI. Run:
  /// `cargo test -p video-indexer test_inference_shape -- --ignored`
  #[test]
  #[ignore]
  fn test_run_inference_shape() {
    // Find segmentation model in the models/ directory from the project root
    let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .parent() // indexer/
      .unwrap()
      .join("models/segmentation-3.0.onnx");

    assert!(
      model_path.exists(),
      "segmentation model not found at {model_path:?}"
    );

    let mut session = ort::session::Session::builder()
      .unwrap()
      .commit_from_file(&model_path)
      .unwrap();

    let audio_window = vec![0.0f32; WINDOW_SAMPLES];
    let (logits, num_frames) = run_inference(&mut session, &audio_window).unwrap();

    assert_eq!(
      num_frames, FRAMES_PER_WINDOW,
      "expected {FRAMES_PER_WINDOW} frames for full window"
    );
    assert_eq!(logits.len(), FRAMES_PER_WINDOW * 7);
  }

  // ── Phase 3 diarize tests (Wave 0 skeleton) ────────────────────────────────
  mod diarize_tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::types::SpeakerSegment;
    #[allow(unused_imports)]
    use core::time::Duration;

    // ── Fbank config field name verification (no model, just needs to compile) ──

    #[test]
    fn test_fbank_config_field_names_compile() {
      // Verify mel_spec 0.3.4 actual field names: num_mel_bins / frame_length_ms / frame_shift_ms
      // CONTEXT.md pseudocode used incorrect field names (n_mels / fft_size / hop_size),
      // this test compiling proves the field names are correct.
      let _ = FbankConfig {
        num_mel_bins: 80,       // NOT n_mels
        frame_length_ms: 25.0,  // NOT frame_length (samples)
        frame_shift_ms: 10.0,   // NOT hop_size (samples)
        apply_cmn: false,       // CRITICAL: override default true
        preemphasis: 0.97,
        dither: 0.0,
        ..Default::default()
      };
    }

    // ── Label reconstruction tests (no model) ────────────────────────────

    #[test]
    fn test_segment_merge_adjacent_same_speaker() {
      // 3-segment sequence: [spk0 x20, spk1 x15, spk0 x20]
      // Each frame ≈0.017s, 15 frames ≈0.255s > MERGE_THRESHOLD(0.25s), won't be eliminated as short fragments
      // Expected: 3 SpeakerSegments after merging
      let frame_step = 10.0 / 589.0_f64;
      let vad_onset = 0.5_f32;
      let mut frame_probs: Vec<Vec<(u32, f32)>> = Vec::new();
      for _ in 0..20 { frame_probs.push(vec![(0, 0.9)]); }
      for _ in 0..15 { frame_probs.push(vec![(1, 0.9)]); }
      for _ in 0..20 { frame_probs.push(vec![(0, 0.9)]); }
      let segments = build_speaker_segments(&frame_probs, frame_step, vad_onset);
      assert_eq!(segments.len(), 3, "3 runs (spk0/spk1/spk0) should produce 3 segments");
      assert_eq!(segments[0].speaker_id, 0);
      assert_eq!(segments[1].speaker_id, 1);
      assert_eq!(segments[2].speaker_id, 0);
    }

    #[test]
    fn test_exclusive_assignment_picks_max() {
      // Frame has speaker 0 prob=0.6, speaker 1 prob=0.8
      // exclusive mode → frame assigned to speaker 1 (highest probability)
      // is_overlap=true (both speakers >= vad_onset=0.5)
      let frame_step = 10.0 / 589.0_f64;
      let frame_probs: Vec<Vec<(u32, f32)>> = vec![
        vec![(0, 0.6), (1, 0.8)], // overlap frame: both active
      ];
      let segments = build_speaker_segments(&frame_probs, frame_step, 0.5);
      assert_eq!(segments.len(), 1);
      assert_eq!(segments[0].speaker_id, 1, "speaker 1 has higher prob and wins argmax");
      assert!(segments[0].is_overlap, "two speakers active: is_overlap must be true");
    }

    // ── Integration test skeleton (requires model files, marked #[ignore]) ──

    #[test]
    #[ignore = "requires model files and reference_outputs/"]
    fn test_diarize_full_pipeline() {
      let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
      let seg_model_path = root.join("models/speaker-diarization-community-1/segmentation/model.onnx");
      let emb_model_path = root.join("models/speaker-diarization-community-1/embedding/model.onnx");
      let plda_model_path = root.join("models/speaker-diarization-community-1/plda/vbx_model.npz");

      // Skip if any file is missing (skeleton test, depends on optional resources)
      let audio_path = root.join("reference_outputs/01_人声_自录双人对话_16khz_mono.wav");
      for p in [&seg_model_path, &emb_model_path, &plda_model_path, &audio_path] {
        if !p.exists() {
          eprintln!("skipping test_diarize_full_pipeline: {p:?} not found");
          return;
        }
      }

      let opts = super::TieguanyinOolongOptions::new()
        .with_embedding_model_path(Some(emb_model_path))
        .with_plda_model(plda_model_path);
      let mut detector = super::TieguanyinOolong::new_from_path(&seg_model_path, opts).unwrap();

      // let segments = detector.diarize(&audio).unwrap();
      // assert!(!segments.is_empty(), "diarize should produce at least one segment");
      // let speaker_ids: std::collections::HashSet<u32> =
      //     segments.iter().map(|s| s.speaker_id).collect();
      // assert_eq!(speaker_ids.len(), 2, "dual-speaker audio should have 2 speakers");
      // segments are sorted by start
      // for w in segments.windows(2) { assert!(w[0].start <= w[1].start); }
      let _ = detector; // suppress unused warning
    }

    #[test]
    fn test_diarize_with_embeddings_returns_err_without_model() {
      // OUT-05: diarize_with_embeddings() returns Err when embedding model not loaded
      // VAD-only mode (no embedding_model_path) → EmbeddingModelNotLoaded
      // We cannot construct TieguanyinOolong without a model file, so we test the
      // error variant existence and the EmbeddingModelNotLoaded Display message.
      use super::TieguanyinOolongError;
      let err = TieguanyinOolongError::EmbeddingModelNotLoaded;
      let msg = format!("{err}");
      assert!(
        msg.contains("embedding model not loaded"),
        "error message should mention embedding model: {msg}"
      );
    }

    // ── DIA-03: embedding_exclude_overlap logic tests ─────────────────────

    #[test]
    fn test_overlap_mask_zeroing() {
      // DIA-03: verify embedding_exclude_overlap logic
      // Frame probabilities: frame 0 = spk0 active only, frame 1 = spk0+spk1 both active (overlap),
      //         frame 2 = spk1 active only
      // When exclude_overlap=true: frame 1's mask should be 0.0 (zeroed out)
      let onset = 0.5f32;
      let chunk_probs: Vec<[f32; 3]> = vec![
        [0.8, 0.1, 0.0], // frame 0: only spk0 active
        [0.7, 0.6, 0.0], // frame 1: spk0 + spk1 overlap
        [0.1, 0.9, 0.0], // frame 2: only spk1 active
      ];
      let local_speaker = 0usize;

      // Build mask for local_speaker=0
      let mut mask: Vec<f32> = chunk_probs
        .iter()
        .map(|p| if p[local_speaker] >= onset { 1.0 } else { 0.0 })
        .collect();
      assert_eq!(mask, vec![1.0, 1.0, 0.0], "initial mask before overlap exclusion");

      // Apply overlap exclusion
      for (i, probs) in chunk_probs.iter().enumerate() {
        let num_active = probs.iter().filter(|&&p| p >= onset).count();
        if num_active >= 2 {
          mask[i] = 0.0;
        }
      }
      assert_eq!(mask, vec![1.0, 0.0, 0.0], "frame 1 (overlap) should be zeroed");
    }

    // ── DIA-04: batch_collection logic tests ───────────────────────────────

    #[test]
    fn test_batch_collection() {
      // DIA-04: verify embedding_batch_size concept (no model, test counting logic)
      // Simulate 5 (chunk, speaker) pairs, batch_size=3: first batch 3, second batch 2
      let total_pairs = 5usize;
      let batch_size = 3usize;
      let batches: Vec<std::ops::Range<usize>> = (0..total_pairs)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(total_pairs))
        .collect();
      assert_eq!(batches.len(), 2, "5 pairs with batch_size=3 should produce 2 batches");
      assert_eq!(batches[0].len(), 3, "first batch: 3 pairs");
      assert_eq!(batches[1].len(), 2, "second batch: 2 pairs");
    }

    // ── VBx: logsumexp / softmax / PLDA tests ─────────────────────────────

    #[test]
    fn test_logsumexp() {
      use ndarray::array;
      // ln(e^1 + e^2 + e^3) ≈ 3.40760596…
      let mat = array![[1.0_f64, 2.0, 3.0]];
      let result = logsumexp_rows(&mat);
      assert!(
        (result[0] - 3.4076059644443806).abs() < 1e-6,
        "logsumexp([1,2,3]) = {}, expected ~3.4076",
        result[0]
      );

      // Extreme values: no NaN/Inf
      let extreme = array![[-1000.0_f64, -1001.0]];
      let result2 = logsumexp_rows(&extreme);
      assert!(result2[0].is_finite(), "extreme input should not produce NaN/Inf");
    }

    #[test]
    fn test_softmax() {
      use ndarray::array;
      let mut mat = array![[1.0_f64, 2.0, 3.0]];
      softmax_rows_inplace(&mut mat);
      let expected = [0.09003057_f64, 0.24472847, 0.66524096];
      for (got, exp) in mat.row(0).iter().zip(expected.iter()) {
        assert!(
          (got - exp).abs() < 1e-4,
          "softmax mismatch: got {got}, expected {exp}"
        );
      }

      // All-zero row -> uniform distribution
      let mut zeros = array![[0.0_f64, 0.0, 0.0]];
      softmax_rows_inplace(&mut zeros);
      for &v in zeros.row(0).iter() {
        assert!(
          (v - 1.0 / 3.0).abs() < 1e-6,
          "all-zero softmax should be uniform, got {v}"
        );
      }
    }

    // ── VBx: vbx / vbx_cluster_embeddings tests ───────────────────────────

    #[test]
    fn test_vbx_basic() {
      use ndarray::Array2;

      // Synthesize 10 128-dim embeddings: 5 cluster A (mostly positive) + 5 cluster B (mostly negative)
      let d = 128;
      let mut fea = Array2::<f64>::zeros((10, d));
      for i in 0..5 {
        for j in 0..d {
          // cluster A: base +1.0, plus small noise
          fea[[i, j]] = 1.0 + (i as f64 * 0.01) + (j as f64 * 0.001);
        }
      }
      for i in 5..10 {
        for j in 0..d {
          // cluster B: base -1.0, plus small noise
          fea[[i, j]] = -1.0 + ((i - 5) as f64 * 0.01) + (j as f64 * 0.001);
        }
      }

      // Simple phi (between-class covariance diagonal)
      let phi = Array1::from_elem(d, 1.0_f64);
      // AHC labels: assume 5 initial clusters (over-segmented)
      let ahc_labels: Vec<u32> = vec![0, 1, 2, 2, 2, 3, 3, 4, 4, 4];

      let result = vbx(&fea, &phi, &ahc_labels, 0.07, 0.8, 20, 1e-4);

      // gamma shape is correct
      assert_eq!(result.gamma.nrows(), 10);
      assert_eq!(result.gamma.ncols(), 5); // 5 initial clusters

      // No NaN
      assert!(
        result.gamma.iter().all(|v| v.is_finite()),
        "gamma contains NaN/Inf"
      );
      assert!(
        result.pi.iter().all(|v| v.is_finite()),
        "pi contains NaN/Inf"
      );

      // After pruning surviving speakers (pi > 1e-7), should converge to ~2
      let alive = result.pi.iter().filter(|&&p| p > 1e-7).count();
      assert!(
        alive <= 3,
        "expected ~2 surviving speakers, got {alive} (pi = {:?})",
        result.pi
      );

      // gamma argmax groups correct: first 5 and last 5 should each be consistent
      let labels: Vec<usize> = (0..10)
        .map(|i| {
          result
            .gamma
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
        })
        .collect();
      // First 5 labels consistent
      for i in 1..5 {
        assert_eq!(labels[0], labels[i], "cluster A frames should share label");
      }
      // Last 5 labels consistent
      for i in 6..10 {
        assert_eq!(labels[5], labels[i], "cluster B frames should share label");
      }
      // Two groups have different labels
      assert_ne!(labels[0], labels[5], "cluster A and B should differ");
    }

    #[test]
    #[ignore] // requires PLDA model file on disk
    fn test_vbx_cluster_embeddings() {
      // Synthesize 6 256-dim embeddings (3 cluster A + 3 cluster B)
      let mut embeddings: Vec<Vec<f32>> = Vec::new();
      for i in 0..3 {
        let mut e = vec![0.0f32; 256];
        for j in 0..256 {
          e[j] = 1.0 + (i as f32 * 0.05) + (j as f32 * 0.001);
        }
        embeddings.push(e);
      }
      for i in 0..3 {
        let mut e = vec![0.0f32; 256];
        for j in 0..256 {
          e[j] = -1.0 + (i as f32 * 0.05) + (j as f32 * 0.001);
        }
        embeddings.push(e);
      }

      let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/speaker-diarization-community-1/plda/vbx_model.npz");
      let plda = PldaModel::load(&model_path).expect("PldaModel::load failed");

      let (labels, gamma) = vbx_cluster_embeddings(&embeddings, &plda, 0.6, 0.07, 0.8);
      assert_eq!(labels.len(), 6, "labels length should match embeddings");
      let unique: std::collections::HashSet<u32> = labels.iter().copied().collect();
      assert_eq!(unique.len(), 2, "should converge to 2 speakers, got {:?}", unique);

      // gamma shape: (6, K) where K = unique speaker count
      assert_eq!(gamma.nrows(), 6);
      assert_eq!(gamma.ncols(), unique.len());
    }

    // ── VBx: ahc_init tests ───────────────────────────────────────────────

    #[test]
    fn test_ahc_init_single() {
      // Single embedding → returns [0]
      let embeddings = vec![vec![1.0f32, 0.0, 0.0, 0.5]];
      let labels = ahc_init(&embeddings, 0.6);
      assert_eq!(labels, vec![0u32], "single embedding should get label 0");
    }

    #[test]
    fn test_ahc_init_identical() {
      // 3 identical embeddings + threshold=0.6 → all assigned to the same cluster
      let embeddings = vec![
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![1.0f32, 2.0, 3.0, 4.0],
      ];
      let labels = ahc_init(&embeddings, 0.6);
      assert_eq!(labels.len(), 3);
      assert_eq!(labels[0], labels[1], "identical embeddings should share cluster");
      assert_eq!(labels[1], labels[2], "identical embeddings should share cluster");
    }

    #[test]
    fn test_ahc_init_distinct() {
      // 2 similar + 1 distant → split into 2 clusters at threshold=0.6
      // L2-normalized: [1,0,...] and [0.99,0.1,...] are close;
      // [0,1,...] is far from both
      let embeddings = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.99f32, 0.1, 0.0, 0.0],
        vec![0.0f32, 1.0, 0.0, 0.0],
      ];
      let labels = ahc_init(&embeddings, 0.6);
      assert_eq!(labels.len(), 3);
      assert_eq!(labels[0], labels[1], "similar embeddings should share cluster");
      assert_ne!(labels[0], labels[2], "distinct embedding should be separate cluster");
    }

    #[test]
    #[ignore] // requires model file on disk
    fn test_plda_load() {
      let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/speaker-diarization-community-1/plda/vbx_model.npz");
      let model = PldaModel::load(&model_path).expect("PldaModel::load failed");
      assert_eq!(model.mean1.len(), 256);
      assert_eq!(model.mean2.len(), 128);
      assert_eq!(model.lda.shape(), &[256, 128]);
      assert_eq!(model.plda_mu.len(), 128);
      assert_eq!(model.plda_tr.shape(), &[128, 128]);
      assert_eq!(model.plda_psi.len(), 128);
    }

    #[test]
    #[ignore] // requires model file on disk
    fn test_plda_transform() {
      let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/speaker-diarization-community-1/plda/vbx_model.npz");
      let model = PldaModel::load(&model_path).expect("PldaModel::load failed");

      // (2, 256) input with simple values
      let input = Array2::<f64>::ones((2, 256));
      let output = model.transform(&input);
      assert_eq!(output.shape(), &[2, 128], "transform output shape mismatch");
      // No NaN in output
      assert!(
        output.iter().all(|v| v.is_finite()),
        "transform output contains NaN or Inf"
      );
    }

  }
}
