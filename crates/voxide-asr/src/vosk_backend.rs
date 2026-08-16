//! Vosk speech recognition.
//!
//! Compiled only with the `vosk` feature, because it links `libvosk`, a native
//! library the user installs separately. Like the microphone backend, this is
//! verified by CI rather than by the default test suite.

use crate::{AsrError, Transcriber, Utterance};
use std::path::Path;

/// Streaming recogniser over a Vosk acoustic model.
pub struct VoskTranscriber {
    recognizer: vosk::Recognizer,
    /// Kept alive because the recogniser borrows from it internally.
    _model: vosk::Model,
    backend_label: &'static str,
}

impl VoskTranscriber {
    /// Loads the model in `model_dir`.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self, AsrError> {
        let dir = model_dir.as_ref();
        if !dir.is_dir() {
            return Err(AsrError::ModelNotFound(dir.display().to_string()));
        }

        let model = vosk::Model::new(dir.to_string_lossy().as_ref()).ok_or_else(|| {
            AsrError::Init(format!("vosk rejected the model at {}", dir.display()))
        })?;

        let mut recognizer = vosk::Recognizer::new(&model, voxide_audio::SAMPLE_RATE as f32)
            .ok_or_else(|| AsrError::Init("could not create a vosk recognizer".to_owned()))?;

        // Alternatives give a confidence figure to report; word timings and
        // partial words cost decoding time voxide has no use for.
        recognizer.set_max_alternatives(1);
        recognizer.set_words(false);
        recognizer.set_partial_words(false);

        tracing::info!(model = %dir.display(), "vosk model loaded");

        Ok(Self {
            recognizer,
            _model: model,
            backend_label: "vosk",
        })
    }

    fn take_result(&mut self) -> Option<Utterance> {
        let result = self.recognizer.final_result();
        let best = result.multiple()?.alternatives.into_iter().next()?;
        let utterance = Utterance::new(best.text.trim(), best.confidence);
        (!utterance.is_blank()).then_some(utterance)
    }
}

impl Transcriber for VoskTranscriber {
    fn accept(&mut self, frame: &[i16]) -> Option<Utterance> {
        match self.recognizer.accept_waveform(frame) {
            // Vosk decided the utterance ended.
            Ok(vosk::DecodingState::Finalized) => self.take_result(),
            Ok(_) => None,
            Err(e) => {
                tracing::error!(error = %e, "vosk failed to accept audio");
                None
            }
        }
    }

    fn flush(&mut self) -> Option<Utterance> {
        self.take_result()
    }

    fn reset(&mut self) {
        self.recognizer.reset();
    }

    fn backend(&self) -> &'static str {
        self.backend_label
    }
}

/// Finds a Vosk model directory under `root`.
///
/// A directory counts as a model when it contains the `am` or `graph`
/// subdirectory that every Vosk model ships.
pub fn find_model(root: impl AsRef<Path>) -> Option<std::path::PathBuf> {
    let root = root.as_ref();
    if is_model_dir(root) {
        return Some(root.to_path_buf());
    }

    let mut candidates: Vec<_> = std::fs::read_dir(root)
        .ok()?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| is_model_dir(p))
        .collect();
    // Deterministic pick when several models are installed.
    candidates.sort();
    candidates.into_iter().next()
}

fn is_model_dir(path: &Path) -> bool {
    path.is_dir() && (path.join("am").is_dir() || path.join("graph").is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_directory_is_reported_as_not_found() {
        let err = VoskTranscriber::open("/nonexistent/voxide/model").unwrap_err();
        assert!(matches!(err, AsrError::ModelNotFound(_)));
    }

    #[test]
    fn model_detection_requires_a_known_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_model(tmp.path()).is_none());

        let model = tmp.path().join("vosk-model-small-en-us");
        std::fs::create_dir_all(model.join("am")).unwrap();
        assert_eq!(find_model(tmp.path()).as_deref(), Some(model.as_path()));
    }
}
