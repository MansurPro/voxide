//! Voice activity detection.

/// Whether a frame carries speech, and how confident that call is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadDecision {
    pub is_voice: bool,
    /// Roughly `0.0..=1.0`.
    pub confidence: f32,
}

/// Classifies a frame as speech or silence.
pub trait Vad: Send {
    fn detect(&mut self, frame: &[i16]) -> VadDecision;

    /// Clears any accumulated state between utterances.
    fn reset(&mut self);
}

/// RMS-energy threshold detection.
///
/// Cheap and adequate in a quiet room, which covers most desk setups. It will
/// mistake steady background noise for speech; a spectral or neural detector
/// slots in behind the same trait when that matters.
pub struct EnergyVad {
    threshold: f32,
    /// Frames of sub-threshold audio required before declaring silence.
    /// Without this, natural pauses between words end an utterance early.
    hangover_frames: usize,
    remaining_hangover: usize,
}

/// Full-scale is 32767, so this sits near -50 dBFS.
pub const DEFAULT_ENERGY_THRESHOLD: f32 = 100.0;

impl EnergyVad {
    pub fn new(threshold: f32, hangover_frames: usize) -> Self {
        Self {
            threshold,
            hangover_frames,
            remaining_hangover: 0,
        }
    }

    /// Default threshold with roughly 300 ms of hangover.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_ENERGY_THRESHOLD, crate::frames_for(0.3))
    }

    pub fn rms(frame: &[i16]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }
        // Accumulate in f64: 512 samples squared at full scale overflows f32
        // precision well before it overflows range, and the drift is visible.
        let sum: f64 = frame.iter().map(|s| f64::from(*s).powi(2)).sum();
        (sum / frame.len() as f64).sqrt() as f32
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl Vad for EnergyVad {
    fn detect(&mut self, frame: &[i16]) -> VadDecision {
        let rms = Self::rms(frame);
        let confidence = (rms / (self.threshold * 2.0)).clamp(0.0, 1.0);

        if rms > self.threshold {
            self.remaining_hangover = self.hangover_frames;
            return VadDecision {
                is_voice: true,
                confidence,
            };
        }

        if self.remaining_hangover > 0 {
            self.remaining_hangover -= 1;
            return VadDecision {
                is_voice: true,
                confidence,
            };
        }

        VadDecision {
            is_voice: false,
            confidence,
        }
    }

    fn reset(&mut self) {
        self.remaining_hangover = 0;
    }
}

/// Treats every frame as speech. Used with push-to-talk, where the user has
/// already signalled intent and gating on energy would only add latency.
pub struct AlwaysVoice;

impl Vad for AlwaysVoice {
    fn detect(&mut self, _frame: &[i16]) -> VadDecision {
        VadDecision {
            is_voice: true,
            confidence: 1.0,
        }
    }
    fn reset(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(amplitude: i16, len: usize) -> Vec<i16> {
        (0..len)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    #[test]
    fn rms_of_silence_is_zero() {
        assert_eq!(EnergyVad::rms(&[0; 512]), 0.0);
    }

    #[test]
    fn rms_of_a_square_wave_is_its_amplitude() {
        assert!((EnergyVad::rms(&tone(1000, 512)) - 1000.0).abs() < 1.0);
    }

    #[test]
    fn rms_of_an_empty_frame_is_zero_not_nan() {
        assert_eq!(EnergyVad::rms(&[]), 0.0);
    }

    #[test]
    fn loud_frames_are_voice_and_quiet_ones_are_not() {
        let mut vad = EnergyVad::new(100.0, 0);
        assert!(vad.detect(&tone(1000, 512)).is_voice);
        assert!(!vad.detect(&[0; 512]).is_voice);
    }

    /// A pause between words must not end the utterance.
    #[test]
    fn hangover_bridges_a_short_pause() {
        let mut vad = EnergyVad::new(100.0, 3);

        assert!(vad.detect(&tone(1000, 512)).is_voice);
        for i in 0..3 {
            assert!(
                vad.detect(&[0; 512]).is_voice,
                "silence frame {i} ended the utterance early"
            );
        }
        assert!(!vad.detect(&[0; 512]).is_voice, "hangover never expired");
    }

    #[test]
    fn hangover_rearms_on_new_speech() {
        let mut vad = EnergyVad::new(100.0, 2);
        vad.detect(&tone(1000, 512));
        vad.detect(&[0; 512]);
        vad.detect(&tone(1000, 512));
        // Hangover was reset by the second burst.
        assert!(vad.detect(&[0; 512]).is_voice);
        assert!(vad.detect(&[0; 512]).is_voice);
        assert!(!vad.detect(&[0; 512]).is_voice);
    }

    #[test]
    fn reset_clears_the_hangover() {
        let mut vad = EnergyVad::new(100.0, 5);
        vad.detect(&tone(1000, 512));
        vad.reset();
        assert!(!vad.detect(&[0; 512]).is_voice);
    }

    #[test]
    fn confidence_is_bounded() {
        let mut vad = EnergyVad::new(100.0, 0);
        let loud = vad.detect(&tone(i16::MAX, 512));
        assert!((0.0..=1.0).contains(&loud.confidence));
        assert_eq!(loud.confidence, 1.0);
        assert_eq!(vad.detect(&[0; 512]).confidence, 0.0);
    }

    #[test]
    fn always_voice_never_gates() {
        let mut vad = AlwaysVoice;
        assert!(vad.detect(&[0; 512]).is_voice);
    }
}
