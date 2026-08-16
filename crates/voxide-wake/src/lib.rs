//! Deciding when voxide should start listening for a command.
//!
//! Two strategies ship today, and neither needs a model or a native library:
//!
//! - [`AlwaysOn`] — every frame counts as awake. This is the push-to-talk and
//!   `voxide transcribe` path, where the user has already signalled intent.
//! - [`TranscriptSpotter`] — looks for the wake word in recognised *text*.
//!   Since the recogniser is running anyway, this costs nothing extra and
//!   works with any speech backend.
//!
//! An acoustic detector (Rustpotter or similar) that fires before recognition
//! would cut latency and let the recogniser stay idle until needed. It slots
//! in behind [`WakeDetector`] when that trade is worth its dependency weight.

/// Decides whether the assistant is awake.
pub trait WakeDetector: Send {
    /// Called per audio frame. `true` means the wake word just fired.
    ///
    /// Text-based detectors ignore this and always return `false`.
    fn accept_audio(&mut self, _frame: &[i16]) -> bool {
        false
    }

    /// Called with each recognised utterance.
    ///
    /// Returns what the user actually asked for, with the wake word removed.
    fn accept_text(&mut self, text: &str) -> WakeOutcome;

    fn reset(&mut self) {}

    fn name(&self) -> &'static str;
}

/// What an utterance meant, given the current wake state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WakeOutcome {
    /// Not awake and no wake word present; ignore this utterance entirely.
    Ignored,
    /// The wake word was spoken with nothing after it. Acknowledge and listen.
    WokeUp,
    /// A command to act on. For "jarvis, run the tests" this carries only
    /// "run the tests".
    Command(String),
}

/// Treats every utterance as a command. No wake word.
pub struct AlwaysOn;

impl WakeDetector for AlwaysOn {
    fn accept_text(&mut self, text: &str) -> WakeOutcome {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            WakeOutcome::Ignored
        } else {
            WakeOutcome::Command(trimmed.to_owned())
        }
    }

    fn name(&self) -> &'static str {
        "always-on"
    }
}

/// Trims and reduces every internal whitespace run to a single space.
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for word in s.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

/// Spots the wake word inside recognised text.
///
/// Speech recognisers mangle names, so several spellings can be registered for
/// one wake word and any of them fires it.
pub struct TranscriptSpotter {
    variants: Vec<String>,
    /// Once awake, follow-up commands need no wake word until the session ends.
    awake: bool,
}

impl TranscriptSpotter {
    /// `variants` are matched case-insensitively.
    pub fn new(variants: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            variants: variants
                .into_iter()
                .map(|v| v.into().to_lowercase())
                .filter(|v| !v.is_empty())
                .collect(),
            awake: false,
        }
    }

    pub fn is_awake(&self) -> bool {
        self.awake
    }

    /// Ends the session, so the wake word is required again.
    pub fn sleep(&mut self) {
        self.awake = false;
    }

    /// Removes the first wake-word occurrence, returning the remainder.
    fn strip(&self, lowered: &str, original: &str) -> Option<String> {
        for variant in &self.variants {
            if let Some(at) = lowered.find(variant.as_str()) {
                let mut rest = String::with_capacity(original.len());
                rest.push_str(&original[..at]);
                rest.push(' ');
                rest.push_str(&original[at + variant.len()..]);

                // Leading punctuation is what remains of "voxide, run ..."
                // once the name is gone.
                let trimmed = rest
                    .trim()
                    .trim_start_matches(|c: char| !c.is_alphanumeric());
                // Removing a word from the middle leaves a gap; without this,
                // "okay voxide run" becomes "okay   run".
                return Some(collapse_whitespace(trimmed));
            }
        }
        None
    }
}

impl WakeDetector for TranscriptSpotter {
    fn accept_text(&mut self, text: &str) -> WakeOutcome {
        let original = text.trim();
        if original.is_empty() {
            return WakeOutcome::Ignored;
        }
        let lowered = original.to_lowercase();

        match self.strip(&lowered, original) {
            Some(rest) => {
                self.awake = true;
                if rest.is_empty() {
                    WakeOutcome::WokeUp
                } else {
                    // "jarvis run the tests" in one breath: act immediately
                    // rather than making the user say it twice.
                    WakeOutcome::Command(rest)
                }
            }
            None if self.awake => WakeOutcome::Command(original.to_owned()),
            None => WakeOutcome::Ignored,
        }
    }

    fn reset(&mut self) {
        self.awake = false;
    }

    fn name(&self) -> &'static str {
        "transcript-spotter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spotter() -> TranscriptSpotter {
        TranscriptSpotter::new(["voxide", "vox side", "oxide"])
    }

    #[test]
    fn always_on_passes_everything_through() {
        let mut w = AlwaysOn;
        assert_eq!(
            w.accept_text("run the tests"),
            WakeOutcome::Command("run the tests".into())
        );
        assert_eq!(w.accept_text("  "), WakeOutcome::Ignored);
    }

    #[test]
    fn ignores_speech_before_the_wake_word() {
        let mut w = spotter();
        assert_eq!(w.accept_text("run the tests"), WakeOutcome::Ignored);
        assert!(!w.is_awake());
    }

    #[test]
    fn bare_wake_word_wakes_up() {
        let mut w = spotter();
        assert_eq!(w.accept_text("voxide"), WakeOutcome::WokeUp);
        assert!(w.is_awake());
    }

    /// The case that makes voice control feel natural: name and command in a
    /// single breath, rather than wake-pause-speak.
    #[test]
    fn wake_word_and_command_in_one_utterance() {
        let mut w = spotter();
        assert_eq!(
            w.accept_text("voxide run the tests"),
            WakeOutcome::Command("run the tests".into())
        );
    }

    #[test]
    fn strips_punctuation_after_the_wake_word() {
        let mut w = spotter();
        assert_eq!(
            w.accept_text("Voxide, run the tests"),
            WakeOutcome::Command("run the tests".into())
        );
    }

    #[test]
    fn accepts_a_mishearing_variant() {
        let mut w = spotter();
        assert_eq!(
            w.accept_text("vox side what changed"),
            WakeOutcome::Command("what changed".into())
        );
    }

    #[test]
    fn follow_up_needs_no_wake_word_once_awake() {
        let mut w = spotter();
        w.accept_text("voxide");
        assert_eq!(
            w.accept_text("run the tests"),
            WakeOutcome::Command("run the tests".into())
        );
    }

    #[test]
    fn sleeping_requires_the_wake_word_again() {
        let mut w = spotter();
        w.accept_text("voxide");
        w.sleep();
        assert_eq!(w.accept_text("run the tests"), WakeOutcome::Ignored);
    }

    #[test]
    fn reset_clears_the_awake_state() {
        let mut w = spotter();
        w.accept_text("voxide");
        w.reset();
        assert!(!w.is_awake());
    }

    #[test]
    fn wake_word_mid_sentence_still_fires() {
        let mut w = spotter();
        assert_eq!(
            w.accept_text("okay voxide run the tests"),
            WakeOutcome::Command("okay run the tests".into())
        );
    }

    #[test]
    fn empty_variant_list_never_wakes() {
        let mut w = TranscriptSpotter::new(Vec::<String>::new());
        assert_eq!(w.accept_text("anything"), WakeOutcome::Ignored);
    }
}
