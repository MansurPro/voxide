//! Per-frame preprocessing and the pre-roll buffer.

use crate::vad::{Vad, VadDecision};

/// A frame after preprocessing.
///
/// `samples` borrows the preprocessor's internal buffer. That is deliberate:
/// allocating a fresh `Vec` per frame, 31 times a second, for the entire time
/// the daemon is listening, is pure waste.
#[derive(Debug)]
pub struct ProcessedFrame<'a> {
    pub samples: &'a [i16],
    pub is_voice: bool,
    pub confidence: f32,
}

/// Normalises level and classifies voice activity.
///
/// Note that [`ProcessedFrame::samples`] is the *processed* audio and callers
/// are expected to pass it downstream. Computing a cleaned signal and then
/// feeding the raw frame to the recogniser anyway is an easy mistake to make
/// and silently wastes the whole preprocessing stage.
pub struct Preprocessor {
    vad: Box<dyn Vad>,
    gain: Option<GainNormalizer>,
    buffer: Vec<i16>,
}

impl Preprocessor {
    pub fn new(vad: Box<dyn Vad>, gain: bool) -> Self {
        Self {
            vad,
            gain: gain.then(GainNormalizer::new),
            buffer: Vec::new(),
        }
    }

    pub fn process(&mut self, frame: &[i16]) -> ProcessedFrame<'_> {
        self.buffer.clear();
        self.buffer.extend_from_slice(frame);

        if let Some(gain) = &mut self.gain {
            gain.apply(&mut self.buffer);
        }

        let VadDecision {
            is_voice,
            confidence,
        } = self.vad.detect(&self.buffer);

        ProcessedFrame {
            samples: &self.buffer,
            is_voice,
            confidence,
        }
    }

    pub fn reset(&mut self) {
        self.vad.reset();
        if let Some(gain) = &mut self.gain {
            gain.reset();
        }
    }
}

/// Smoothly scales quiet input up toward a target level.
struct GainNormalizer {
    current: f32,
}

const TARGET_RMS: f32 = 3_000.0;
const MIN_GAIN: f32 = 0.5;
const MAX_GAIN: f32 = 3.0;
/// Weight of the previous gain when smoothing. High, because stepping gain
/// abruptly between frames is audible as pumping and upsets the recogniser.
const SMOOTHING: f32 = 0.9;

impl GainNormalizer {
    fn new() -> Self {
        Self { current: 1.0 }
    }

    fn apply(&mut self, frame: &mut [i16]) {
        let rms = crate::vad::EnergyVad::rms(frame);
        if rms > 0.0 {
            let target = (TARGET_RMS / rms).clamp(MIN_GAIN, MAX_GAIN);
            self.current = SMOOTHING * self.current + (1.0 - SMOOTHING) * target;
        }

        for s in frame.iter_mut() {
            let scaled = f32::from(*s) * self.current;
            *s = scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
        }
    }

    fn reset(&mut self) {
        self.current = 1.0;
    }
}

/// Fixed-capacity buffer of recent frames.
///
/// Speech always starts slightly before a detector notices it. Keeping a short
/// history means the beginning of an utterance can be replayed into the
/// recogniser rather than clipped off.
pub struct PreRoll {
    frames: std::collections::VecDeque<Vec<i16>>,
    capacity: usize,
}

impl PreRoll {
    pub fn new(seconds: f32) -> Self {
        Self {
            frames: std::collections::VecDeque::new(),
            capacity: crate::frames_for(seconds).max(1),
        }
    }

    pub fn push(&mut self, frame: &[i16]) {
        if self.frames.len() == self.capacity {
            // Reuse the evicted allocation instead of freeing and reallocating.
            let mut recycled = self.frames.pop_front().unwrap_or_default();
            recycled.clear();
            recycled.extend_from_slice(frame);
            self.frames.push_back(recycled);
        } else {
            self.frames.push_back(frame.to_vec());
        }
    }

    /// Removes and returns everything buffered.
    pub fn drain(&mut self) -> Vec<Vec<i16>> {
        self.frames.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::{AlwaysVoice, EnergyVad};

    fn tone(amplitude: i16, len: usize) -> Vec<i16> {
        (0..len)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    #[test]
    fn preprocessor_reports_voice_activity() {
        let mut p = Preprocessor::new(Box::new(EnergyVad::new(100.0, 0)), false);
        assert!(p.process(&tone(1000, 512)).is_voice);
        assert!(!p.process(&[0; 512]).is_voice);
    }

    #[test]
    fn without_gain_samples_pass_through_unchanged() {
        let mut p = Preprocessor::new(Box::new(AlwaysVoice), false);
        let input = tone(1000, 512);
        assert_eq!(p.process(&input).samples, input.as_slice());
    }

    /// Guards the mistake this type is documented against: the processed
    /// audio must actually differ from the input when gain is enabled, and it
    /// must be what the caller receives.
    #[test]
    fn gain_actually_reaches_the_output() {
        let mut p = Preprocessor::new(Box::new(AlwaysVoice), true);
        let quiet = tone(100, 512);

        let mut last = 0i16;
        for _ in 0..200 {
            last = p.process(&quiet).samples[0];
        }

        assert!(
            last > 150,
            "quiet input was not amplified; got {last} from 100"
        );
    }

    #[test]
    fn gain_does_not_overflow_on_full_scale_input() {
        let mut p = Preprocessor::new(Box::new(AlwaysVoice), true);
        for _ in 0..50 {
            let out = p.process(&tone(i16::MAX, 512));
            assert!(out.samples.iter().all(|s| *s != 0), "clipped to zero");
        }
    }

    #[test]
    fn reset_clears_vad_state() {
        let mut p = Preprocessor::new(Box::new(EnergyVad::new(100.0, 5)), false);
        p.process(&tone(1000, 512));
        p.reset();
        assert!(!p.process(&[0; 512]).is_voice);
    }

    #[test]
    fn preroll_evicts_oldest_beyond_capacity() {
        let mut r = PreRoll::new(0.1);
        let cap = r.capacity();

        for i in 0..cap + 5 {
            r.push(&[i as i16; 4]);
        }
        assert_eq!(r.len(), cap);

        let drained = r.drain();
        assert_eq!(drained.len(), cap);
        // Oldest surviving frame is the (cap+5-cap)=5th pushed.
        assert_eq!(drained[0][0], 5);
        assert!(r.is_empty());
    }

    #[test]
    fn preroll_capacity_is_never_zero() {
        assert!(PreRoll::new(0.0).capacity() >= 1);
    }

    #[test]
    fn preroll_clear_empties_without_draining() {
        let mut r = PreRoll::new(1.0);
        r.push(&[1, 2, 3]);
        r.clear();
        assert!(r.is_empty());
    }
}
