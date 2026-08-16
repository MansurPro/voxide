//! The listening loop: audio frames in, matched commands out.
//!
//! This crate deliberately performs **no I/O of its own**. It does not print,
//! execute, or play sounds; it emits [`PipelineEvent`] and lets the caller
//! decide. That separation is what makes the loop testable: a test drives it
//! with a [`WavSource`](voxide_audio::WavSource) and a
//! [`MockTranscriber`](voxide_asr::MockTranscriber) and asserts on the event
//! sequence, with no microphone, no speech model, and nothing executed.

use voxide_asr::Transcriber;
use voxide_audio::{AudioError, AudioSource, Preprocessor, Vad};
use voxide_core::Slots;
use voxide_intent::Matcher;
use voxide_wake::{WakeDetector, WakeOutcome};

/// Something the pipeline observed. Ordered as it happens.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineEvent {
    /// Voice activity began.
    SpeechStarted,
    /// Voice activity ended after this many frames.
    SpeechEnded { frames: usize },
    /// The recogniser produced text.
    Transcribed { text: String, confidence: f32 },
    /// The wake word fired with no command attached.
    WokeUp,
    /// Text was recognised but no wake word was active, so it was dropped.
    Ignored { text: String },
    /// A command was resolved and should be executed.
    Matched {
        id: String,
        score: f32,
        slots: Slots,
        text: String,
    },
    /// Text was addressed to voxide but matched nothing above the threshold.
    NoMatch {
        text: String,
        best: Option<(String, f32)>,
    },
    /// The audio source ran out.
    SourceExhausted,
}

/// Tunables for the loop.
#[derive(Debug, Clone)]
pub struct Config {
    /// Minimum match score to act on.
    pub threshold: f32,
    /// Silence after speech before the recogniser is flushed.
    pub silence_timeout_secs: f32,
    /// Audio retained before speech is detected, so onsets are not clipped.
    pub preroll_secs: f32,
    /// Apply gain normalisation before recognition.
    pub gain: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            threshold: voxide_intent::DEFAULT_THRESHOLD,
            // Long enough to survive a mid-sentence pause, short enough that a
            // finished command does not feel like it hung.
            silence_timeout_secs: 1.2,
            preroll_secs: 0.5,
            gain: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Buffering into pre-roll, waiting for voice.
    Idle,
    /// Voice detected; frames are going to the recogniser.
    Listening,
}

/// Wires an audio source, recogniser, wake detector, and matcher together.
pub struct Pipeline<'a> {
    source: Box<dyn AudioSource + 'a>,
    transcriber: Box<dyn Transcriber + 'a>,
    wake: Box<dyn WakeDetector + 'a>,
    matcher: &'a dyn Matcher,
    preprocessor: Preprocessor,
    preroll: voxide_audio::pipeline::PreRoll,
    config: Config,
    state: State,
    silence_frames: usize,
    speech_frames: usize,
    frame: Vec<i16>,
}

impl<'a> Pipeline<'a> {
    pub fn new(
        source: Box<dyn AudioSource + 'a>,
        transcriber: Box<dyn Transcriber + 'a>,
        wake: Box<dyn WakeDetector + 'a>,
        matcher: &'a dyn Matcher,
        vad: Box<dyn Vad>,
        config: Config,
    ) -> Self {
        let frame_len = source.frame_len();
        Self {
            preprocessor: Preprocessor::new(vad, config.gain),
            preroll: voxide_audio::pipeline::PreRoll::new(config.preroll_secs),
            frame: vec![0; frame_len],
            source,
            transcriber,
            wake,
            matcher,
            state: State::Idle,
            silence_frames: 0,
            speech_frames: 0,
            config,
        }
    }

    fn silence_limit(&self) -> usize {
        voxide_audio::frames_for(self.config.silence_timeout_secs)
    }

    /// Advances by one frame, appending anything observed to `events`.
    ///
    /// Returns `false` once the source is exhausted.
    pub fn step(&mut self, events: &mut Vec<PipelineEvent>) -> Result<bool, AudioError> {
        let n = self.source.next_frame(&mut self.frame)?;
        if n == 0 {
            // Do not strand a phrase inside the decoder at end of input.
            if self.state == State::Listening
                && let Some(utterance) = self.transcriber.flush()
            {
                self.handle_utterance(&utterance, events);
            }
            events.push(PipelineEvent::SourceExhausted);
            return Ok(false);
        }

        let processed = self.preprocessor.process(&self.frame[..n]);
        let is_voice = processed.is_voice;
        // Copy out so the preprocessor's borrow ends before `self` is used
        // mutably below. The copy is one frame, not per-sample work.
        let samples: Vec<i16> = processed.samples.to_vec();

        match self.state {
            State::Idle => {
                self.preroll.push(&samples);
                if is_voice {
                    self.state = State::Listening;
                    self.silence_frames = 0;
                    self.speech_frames = 0;
                    events.push(PipelineEvent::SpeechStarted);

                    // Replay buffered audio so the utterance onset is not lost.
                    for buffered in self.preroll.drain() {
                        if let Some(u) = self.transcriber.accept(&buffered) {
                            self.handle_utterance(&u, events);
                        }
                    }
                }
            }
            State::Listening => {
                self.speech_frames += 1;

                if let Some(u) = self.transcriber.accept(&samples) {
                    self.handle_utterance(&u, events);
                }

                if is_voice {
                    self.silence_frames = 0;
                } else {
                    self.silence_frames += 1;
                    if self.silence_frames >= self.silence_limit() {
                        if let Some(u) = self.transcriber.flush() {
                            self.handle_utterance(&u, events);
                        }
                        events.push(PipelineEvent::SpeechEnded {
                            frames: self.speech_frames,
                        });
                        self.end_utterance();
                    }
                }
            }
        }

        Ok(true)
    }

    fn end_utterance(&mut self) {
        self.state = State::Idle;
        self.silence_frames = 0;
        self.speech_frames = 0;
        self.transcriber.reset();
        self.preprocessor.reset();
        self.preroll.clear();
    }

    fn handle_utterance(
        &mut self,
        utterance: &voxide_asr::Utterance,
        events: &mut Vec<PipelineEvent>,
    ) {
        if utterance.is_blank() {
            return;
        }

        events.push(PipelineEvent::Transcribed {
            text: utterance.text.clone(),
            confidence: utterance.confidence,
        });

        let command_text = match self.wake.accept_text(&utterance.text) {
            WakeOutcome::Ignored => {
                events.push(PipelineEvent::Ignored {
                    text: utterance.text.clone(),
                });
                return;
            }
            WakeOutcome::WokeUp => {
                events.push(PipelineEvent::WokeUp);
                return;
            }
            WakeOutcome::Command(text) => text,
        };

        let ranked = self.matcher.rank(&command_text, 1);
        match ranked.first() {
            Some(m) if m.score >= self.config.threshold => {
                let slots = m
                    .via
                    .as_deref()
                    .map(|phrase| voxide_intent::extract_slots(phrase, &command_text))
                    .unwrap_or_default();

                events.push(PipelineEvent::Matched {
                    id: m.id.clone(),
                    score: m.score,
                    slots,
                    text: command_text,
                });
            }
            other => events.push(PipelineEvent::NoMatch {
                text: command_text,
                best: other.map(|m| (m.id.clone(), m.score)),
            }),
        }
    }

    /// Runs until the source is exhausted, collecting every event.
    ///
    /// Only terminates for a finite source, so this is for files and tests;
    /// a live daemon calls [`Pipeline::step`] in its own loop.
    pub fn run_to_end(&mut self) -> Result<Vec<PipelineEvent>, AudioError> {
        let mut events = Vec::new();
        while self.step(&mut events)? {}
        Ok(events)
    }

    pub fn describe_source(&self) -> String {
        self.source.describe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxide_asr::MockTranscriber;
    use voxide_audio::vad::{AlwaysVoice, EnergyVad};
    use voxide_audio::{FRAME_LEN, WavSource};
    use voxide_core::CommandSet;
    use voxide_core::pack::{Action, Command, LoadedCommand, Phrases};
    use voxide_intent::LexicalMatcher;
    use voxide_wake::{AlwaysOn, TranscriptSpotter};

    fn command(id: &str, phrases: &[&str], slots: Vec<voxide_core::SlotDef>) -> LoadedCommand {
        LoadedCommand {
            command: Command {
                id: id.to_owned(),
                description: String::new(),
                phrases: Phrases::Flat(phrases.iter().map(|s| (*s).to_owned()).collect()),
                slots,
                action: Action::Shell {
                    run: "true".into(),
                    cwd: None,
                    timeout_ms: 1000,
                },
                chain: false,
            },
            pack_name: "test".into(),
            pack_dir: ".".into(),
        }
    }

    fn command_set() -> CommandSet {
        CommandSet::from_commands(vec![
            command("cargo.test", &["run the tests"], vec![]),
            command("git.status", &["what changed"], vec![]),
            command(
                "git.checkout",
                &["checkout {branch}"],
                vec![voxide_core::SlotDef {
                    name: "branch".into(),
                    entity: "branch name".into(),
                    optional: false,
                    default: None,
                }],
            ),
        ])
    }

    /// Loud audio, so the energy VAD reports speech.
    fn loud(frames: usize) -> Vec<i16> {
        (0..frames * FRAME_LEN)
            .map(|i| if i % 2 == 0 { 8000 } else { -8000 })
            .collect()
    }

    fn silence(frames: usize) -> Vec<i16> {
        vec![0; frames * FRAME_LEN]
    }

    fn matched(events: &[PipelineEvent]) -> Vec<(&str, &Slots)> {
        events
            .iter()
            .filter_map(|e| match e {
                PipelineEvent::Matched { id, slots, .. } => Some((id.as_str(), slots)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn end_to_end_wav_to_matched_command() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut audio = loud(5);
        audio.extend(silence(60));

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(audio)),
            Box::new(MockTranscriber::new([(2usize, "run the tests")])),
            Box::new(AlwaysOn),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert_eq!(
            matched(&events).first().map(|(id, _)| *id),
            Some("cargo.test")
        );
        assert!(events.contains(&PipelineEvent::SourceExhausted));
    }

    #[test]
    fn emits_speech_started_and_ended_around_an_utterance() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut audio = loud(5);
        audio.extend(silence(80));

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(audio)),
            Box::new(MockTranscriber::silent()),
            Box::new(AlwaysOn),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert!(events.contains(&PipelineEvent::SpeechStarted));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::SpeechEnded { .. })),
            "silence timeout never closed the utterance: {events:?}"
        );
    }

    #[test]
    fn extracts_slots_from_the_winning_phrase() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut audio = loud(5);
        audio.extend(silence(60));

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(audio)),
            Box::new(MockTranscriber::new([(2usize, "checkout release")])),
            Box::new(AlwaysOn),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        let hits = matched(&events);
        assert_eq!(hits[0].0, "git.checkout");
        assert_eq!(
            hits[0].1.get("branch").map(|v| v.as_str().into_owned()),
            Some("release".to_owned())
        );
    }

    #[test]
    fn speech_without_the_wake_word_is_ignored() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut audio = loud(5);
        audio.extend(silence(60));

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(audio)),
            Box::new(MockTranscriber::new([(2usize, "run the tests")])),
            Box::new(TranscriptSpotter::new(["voxide"])),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert!(
            matched(&events).is_empty(),
            "should not have acted: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Ignored { .. }))
        );
    }

    #[test]
    fn wake_word_with_command_in_one_breath_is_acted_on() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut audio = loud(5);
        audio.extend(silence(60));

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(audio)),
            Box::new(MockTranscriber::new([(2usize, "voxide run the tests")])),
            Box::new(TranscriptSpotter::new(["voxide"])),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert_eq!(
            matched(&events).first().map(|(id, _)| *id),
            Some("cargo.test")
        );
    }

    #[test]
    fn unmatched_speech_reports_the_near_miss() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut audio = loud(5);
        audio.extend(silence(60));

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(audio)),
            Box::new(MockTranscriber::new([(
                2usize,
                "completely unrelated words",
            )])),
            Box::new(AlwaysOn),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PipelineEvent::NoMatch { .. })),
            "{events:?}"
        );
    }

    #[test]
    fn silence_alone_produces_no_speech_events() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(silence(40))),
            Box::new(MockTranscriber::silent()),
            Box::new(AlwaysOn),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert_eq!(events, vec![PipelineEvent::SourceExhausted]);
    }

    /// A trailing phrase must not be stranded in the decoder when the audio
    /// simply stops, which is exactly what happens at the end of a WAV file.
    #[test]
    fn pending_text_is_flushed_at_end_of_input() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(loud(3))),
            Box::new(MockTranscriber::new([(10_000usize, "what changed")])),
            Box::new(AlwaysOn),
            &matcher,
            Box::new(AlwaysVoice),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert_eq!(
            matched(&events).first().map(|(id, _)| *id),
            Some("git.status"),
            "{events:?}"
        );
    }

    #[test]
    fn blank_transcriptions_are_dropped() {
        let set = command_set();
        let matcher = LexicalMatcher::new(&set, "en");

        let mut audio = loud(5);
        audio.extend(silence(60));

        let mut p = Pipeline::new(
            Box::new(WavSource::from_samples(audio)),
            Box::new(MockTranscriber::new([(2usize, "[unk]")])),
            Box::new(AlwaysOn),
            &matcher,
            Box::new(EnergyVad::new(100.0, 0)),
            Config::default(),
        );

        let events = p.run_to_end().unwrap();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PipelineEvent::Transcribed { .. })),
            "recogniser filler reached the matcher: {events:?}"
        );
    }
}
