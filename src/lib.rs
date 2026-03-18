//! # oolong-diarization
//!
//! A Rust implementation of pyannote speaker diarization and voice activity detection (VAD).
//!
//! Runs pyannote segmentation + WeSpeaker embedding models via ONNX Runtime,
//! performs VBx clustering (PLDA + VB-HMM), and outputs speaker-labeled time intervals.
//!
//! ## Quick Start
//!
//! ### VAD-only (voice activity detection only)
//!
//! ```rust,no_run
//! use oolong_diarization::{TieguanyinOolong, TieguanyinOolongOptions};
//!
//! let opts = TieguanyinOolongOptions::new().with_vad_only(true);
//! let mut detector = TieguanyinOolong::new_from_path("segmentation/model.onnx", opts).unwrap();
//!
//! let pcm: Vec<f32> = vec![0.0; 16000]; // 1s silence
//! if let Some(range) = detector.detect(&pcm).unwrap() {
//!     println!("Voice: {range}");
//! }
//! ```
//!
//! ### Speaker Diarization
//!
//! ```rust,no_run
//! use oolong_diarization::{TieguanyinOolong, TieguanyinOolongOptions};
//!
//! let opts = TieguanyinOolongOptions::new()
//!     .with_embedding_model_path(Some("embedding/model.onnx".into()))
//!     .with_plda_model("plda/vbx_model.npz".into());
//! let mut diarizer = TieguanyinOolong::new_from_path("segmentation/model.onnx", opts).unwrap();
//!
//! let pcm: &[f32] = &[/* 16kHz mono f32 PCM */];
//! let segments = diarizer.diarize(pcm).unwrap();
//! for seg in &segments {
//!     println!("{:?}..{:?} speaker_{}", seg.start, seg.end, seg.speaker_id);
//! }
//! ```

pub mod types;
mod diarizer;

// Re-export primary public types
pub use diarizer::{TieguanyinOolong, TieguanyinOolongOptions, TieguanyinOolongError};
pub use types::{SpeakerSegment, DiarizeSegment, DiarizeTiming, VoiceRange};
