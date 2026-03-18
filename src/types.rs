//! Public data types: output structures for speaker diarization and VAD.

use core::time::Duration;

/// A speaker-attributed audio segment (exclusive-mode output).
///
/// Each segment is assigned to exactly one speaker (argmax). The `is_overlap` flag
/// indicates whether the segment falls within a multi-speaker overlap region.
/// Used for STT alignment and other scenarios that require a unique speaker label.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegment {
  /// Segment start time (inclusive).
  pub start: Duration,
  /// Segment end time (exclusive).
  pub end: Duration,
  /// Zero-based speaker ID assigned by the clustering algorithm. Always 0 in VAD-only mode.
  pub speaker_id: u32,
  /// Whether this segment falls within a multi-speaker overlap region (2+ speakers active).
  pub is_overlap: bool,
}

/// A speaker-attributed audio segment (overlap-mode output).
///
/// Compatible with the Python pyannote output format: when multiple speakers are
/// active in the same time span, one record is emitted per speaker in parallel.
/// There is no `is_overlap` field because overlap is inherently represented by
/// the co-occurrence of multiple records.
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizeSegment {
  /// Segment start time (inclusive).
  pub start: Duration,
  /// Segment end time (exclusive).
  pub end: Duration,
  /// Zero-based speaker ID.
  pub speaker_id: u32,
}

/// Step-by-step timing information, returned by `diarize_timed()`.
#[derive(Debug, Clone, Copy)]
pub struct DiarizeTiming {
  /// Segmentation step duration (ms): sliding-window ONNX inference + powerset decoding.
  pub segmentation_ms: u64,
  /// Embedding extraction step duration (ms): per-segment WeSpeaker inference.
  pub embedding_ms: u64,
  /// Clustering step duration (ms): VBx (PLDA + VB-HMM) + label reconstruction.
  pub clustering_ms: u64,
  /// End-to-end total duration (ms).
  pub total_ms: u64,
}

/// A voice activity interval (VAD output).
///
/// Half-open interval `[start, end)`, with times represented as `Duration`.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceRange {
  /// Interval start time (inclusive).
  pub start: Duration,
  /// Interval end time (exclusive).
  pub end: Duration,
}

impl VoiceRange {
  /// Create a new voice activity interval.
  pub fn new(start: Duration, end: Duration) -> Self {
    Self { start, end }
  }

  /// Duration of the interval.
  pub fn duration(&self) -> Duration {
    self.end.saturating_sub(self.start)
  }
}

impl core::fmt::Display for VoiceRange {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "[{:.3}s, {:.3}s)",
      self.start.as_secs_f64(),
      self.end.as_secs_f64()
    )
  }
}
