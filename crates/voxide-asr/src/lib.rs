//! Speech-to-text.
//!
//! [`Transcriber`] is fed 16 kHz mono frames and emits an [`Utterance`] when
//! it decides one has ended. Backends live behind the trait so the pipeline
//! can be driven by [`MockTranscriber`] in tests, and so a second engine
//! (Whisper, for instance) is an addition rather than a rewrite.

pub mod mock;

#[cfg(feature = "vosk")]
pub mod vosk_backend;

pub use mock::MockTranscriber;

#[cfg(feature = "vosk")]
pub use vosk_backend::VoskTranscriber;

/// A recognised stretch of speech.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    pub text: String,
    /// Backend-reported confidence in `0.0..=1.0`, where available.
    pub confidence: f32,
}

impl Utterance {
    pub fn new(text: impl Into<String>, confidence: f32) -> Self {
        Self {
            text: text.into(),
            confidence,
        }
    }

    /// True when there is no usable text.
    ///
    /// Recognisers emit empty strings between utterances, and Vosk in
    /// particular emits `[unk]` for audio it cannot place. Neither should ever
    /// reach the matcher.
    pub fn is_blank(&self) -> bool {
        let t = self.text.trim();
        t.is_empty() || t == "[unk]"
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("speech model not found at {0}")]
    ModelNotFound(String),

    #[error("failed to initialise the speech model: {0}")]
    Init(String),

    #[error("speech backend `{0}` is not compiled into this build")]
    Unsupported(&'static str),
}

/// Converts audio frames into text.
pub trait Transcriber: Send {
    /// Feeds one frame. Returns `Some` only when an utterance has completed.
    fn accept(&mut self, frame: &[i16]) -> Option<Utterance>;

    /// Returns whatever has been decoded so far, ending the current utterance.
    ///
    /// Called when the pipeline's silence timer fires, so a trailing phrase is
    /// not stranded inside the decoder waiting for an endpoint the recogniser
    /// has not detected.
    fn flush(&mut self) -> Option<Utterance>;

    /// Discards decoder state between utterances.
    fn reset(&mut self);

    fn backend(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_detection_covers_empty_and_unknown() {
        assert!(Utterance::new("", 1.0).is_blank());
        assert!(Utterance::new("   ", 1.0).is_blank());
        assert!(Utterance::new("[unk]", 1.0).is_blank());
        assert!(!Utterance::new("run the tests", 1.0).is_blank());
    }
}
