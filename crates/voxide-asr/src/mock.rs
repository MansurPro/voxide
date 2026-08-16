//! A scripted transcriber for tests.

use crate::{Transcriber, Utterance};

/// Emits pre-programmed utterances on a schedule measured in frames.
///
/// This is what lets the whole pipeline — VAD, wake word, matching, execution —
/// be tested end to end with no speech model, no native libraries, and no
/// audio hardware, while still exercising the real frame-by-frame control
/// flow rather than a shortcut around it.
pub struct MockTranscriber {
    /// `(frames_to_wait, text)`, consumed in order.
    script: std::collections::VecDeque<(usize, String)>,
    frames_seen: usize,
    next_at: Option<usize>,
    /// Frames handed to `accept`, for assertions about how much audio the
    /// pipeline actually routed here.
    pub frames_accepted: usize,
}

impl MockTranscriber {
    /// Emits each `text` after its `frames` count has elapsed since the
    /// previous emission.
    pub fn new(script: impl IntoIterator<Item = (usize, impl Into<String>)>) -> Self {
        let script: std::collections::VecDeque<(usize, String)> = script
            .into_iter()
            .map(|(frames, text)| (frames, text.into()))
            .collect();

        let mut me = Self {
            script,
            frames_seen: 0,
            next_at: None,
            frames_accepted: 0,
        };
        me.arm();
        me
    }

    /// Emits `text` once, after a single frame.
    pub fn once(text: impl Into<String>) -> Self {
        Self::new([(1usize, text.into())])
    }

    /// Never emits anything.
    pub fn silent() -> Self {
        Self::new(Vec::<(usize, String)>::new())
    }

    fn arm(&mut self) {
        self.next_at = self
            .script
            .front()
            .map(|(frames, _)| self.frames_seen + frames);
    }
}

impl Transcriber for MockTranscriber {
    fn accept(&mut self, _frame: &[i16]) -> Option<Utterance> {
        self.frames_seen += 1;
        self.frames_accepted += 1;

        if self.next_at? > self.frames_seen {
            return None;
        }

        let (_, text) = self.script.pop_front()?;
        self.arm();
        Some(Utterance::new(text, 1.0))
    }

    fn flush(&mut self) -> Option<Utterance> {
        // Anything still scheduled is treated as decoded-but-unemitted.
        let (_, text) = self.script.pop_front()?;
        self.arm();
        Some(Utterance::new(text, 1.0))
    }

    fn reset(&mut self) {
        self.frames_seen = 0;
        self.arm();
    }

    fn backend(&self) -> &'static str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: [i16; 4] = [0; 4];

    #[test]
    fn emits_after_the_scheduled_frame_count() {
        let mut t = MockTranscriber::new([(3usize, "hello")]);
        assert_eq!(t.accept(&FRAME), None);
        assert_eq!(t.accept(&FRAME), None);
        assert_eq!(
            t.accept(&FRAME),
            Some(Utterance::new("hello", 1.0)),
            "should fire on the third frame"
        );
        assert_eq!(t.accept(&FRAME), None, "should not repeat");
    }

    #[test]
    fn emits_several_utterances_in_order() {
        let mut t = MockTranscriber::new([(1usize, "first"), (2usize, "second")]);
        assert_eq!(t.accept(&FRAME).unwrap().text, "first");
        assert_eq!(t.accept(&FRAME), None);
        assert_eq!(t.accept(&FRAME).unwrap().text, "second");
    }

    #[test]
    fn once_fires_immediately() {
        let mut t = MockTranscriber::once("go");
        assert_eq!(t.accept(&FRAME).unwrap().text, "go");
    }

    #[test]
    fn silent_never_emits() {
        let mut t = MockTranscriber::silent();
        for _ in 0..100 {
            assert_eq!(t.accept(&FRAME), None);
        }
        assert_eq!(t.flush(), None);
    }

    #[test]
    fn flush_releases_a_pending_utterance_early() {
        let mut t = MockTranscriber::new([(1000usize, "late")]);
        assert_eq!(t.accept(&FRAME), None);
        assert_eq!(t.flush().unwrap().text, "late");
    }

    #[test]
    fn counts_frames_accepted() {
        let mut t = MockTranscriber::silent();
        for _ in 0..7 {
            t.accept(&FRAME);
        }
        assert_eq!(t.frames_accepted, 7);
    }

    #[test]
    fn reset_restarts_the_countdown() {
        let mut t = MockTranscriber::new([(2usize, "x")]);
        t.accept(&FRAME);
        t.reset();
        assert_eq!(t.accept(&FRAME), None, "countdown should have restarted");
        assert_eq!(t.accept(&FRAME).unwrap().text, "x");
    }
}
