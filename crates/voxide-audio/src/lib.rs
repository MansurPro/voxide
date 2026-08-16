//! Audio capture and preprocessing.
//!
//! Everything downstream of this crate assumes one format: **16 kHz, mono,
//! signed 16-bit PCM**, delivered in fixed-size frames. Sources are
//! responsible for converting to it, so no later stage has to care whether the
//! samples came from a microphone at 48 kHz stereo or from a WAV fixture.
//!
//! The central abstraction is [`AudioSource`]. Having a trait here rather than
//! a hardcoded capture path is what makes the rest of voxide testable: a
//! [`WavSource`] plays a recording through exactly the code a microphone would
//! drive, so the pipeline can be exercised in CI with no audio hardware.

pub mod pipeline;
pub mod resample;
pub mod source;
pub mod vad;

pub use pipeline::{Preprocessor, ProcessedFrame};
pub use source::{AudioError, AudioSource, NullSource, WavSource};
pub use vad::{EnergyVad, Vad, VadDecision};

#[cfg(feature = "mic")]
pub mod mic;
#[cfg(feature = "mic")]
pub use mic::{MicSource, list_devices};

/// The one sample rate the rest of voxide works in.
pub const SAMPLE_RATE: u32 = 16_000;

/// Default samples per frame: 32 ms at 16 kHz.
///
/// Short enough that voice-activity decisions feel immediate, long enough that
/// per-frame overhead stays negligible.
pub const FRAME_LEN: usize = 512;

/// Frames per second at the default frame length.
pub fn frames_per_second() -> f32 {
    SAMPLE_RATE as f32 / FRAME_LEN as f32
}

/// Number of frames spanning `seconds`.
pub fn frames_for(seconds: f32) -> usize {
    (frames_per_second() * seconds).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_timing_matches_the_documented_rate() {
        assert!((frames_per_second() - 31.25).abs() < 1e-6);
        // Rounded up, so a duration is never under-covered.
        assert_eq!(frames_for(1.0), 32);
        assert_eq!(frames_for(0.3), 10);
    }
}
