//! Sample-rate and channel conversion to 16 kHz mono.
//!
//! Capture devices rarely offer 16 kHz directly; 44.1 and 48 kHz are typical.
//! Downsampling by simply picking every Nth sample folds everything above the
//! new Nyquist frequency back down as aliasing, which speech recognisers
//! handle badly, so a low-pass filter is applied first.

/// Averages interleaved channels down to one.
pub fn to_mono(interleaved: &[i16], channels: u16) -> Vec<i16> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let channels = channels as usize;
    interleaved
        .chunks_exact(channels)
        .map(|frame| {
            // Sum in i32: several i16 channels at full scale overflow i16.
            let sum: i32 = frame.iter().map(|s| i32::from(*s)).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

/// Converts `input` from `from_rate` to `to_rate`.
///
/// A low-pass runs before decimation when downsampling, then linear
/// interpolation resamples. This is not a high-quality resampler and is not
/// trying to be: the target is a speech model at 16 kHz, where the audible
/// difference against a windowed-sinc implementation is immaterial and the
/// aliasing this prevents is not.
pub fn resample(input: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }

    let filtered = if from_rate > to_rate {
        let width = (from_rate as f32 / to_rate as f32).round().max(1.0) as usize;
        lowpass(input, width)
    } else {
        input.to_vec()
    };

    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((filtered.len() as f64) * ratio).round() as usize;
    if out_len == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let left = pos.floor() as usize;
        let frac = pos - left as f64;

        let a = f64::from(filtered[left.min(filtered.len() - 1)]);
        let b = f64::from(filtered[(left + 1).min(filtered.len() - 1)]);
        let value = a + (b - a) * frac;

        out.push(value.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16);
    }
    out
}

/// Two cascaded moving averages, used as the anti-aliasing low-pass.
///
/// One pass is not enough. A 3-tap average decimating 48 kHz to 16 kHz
/// attenuates the 12 kHz that would alias into the speech band by only about
/// 9.5 dB, which is audible as a tone in the output. Cascading squares the
/// magnitude response, taking that to roughly 19 dB, while costing about
/// 1.2 dB at 3.4 kHz where speech actually lives.
fn lowpass(input: &[i16], width: usize) -> Vec<i16> {
    if width <= 1 {
        return input.to_vec();
    }
    moving_average(&moving_average(input, width), width)
}

/// Centred moving average.
fn moving_average(input: &[i16], width: usize) -> Vec<i16> {
    if width <= 1 {
        return input.to_vec();
    }

    let half = width / 2;
    let mut out = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(input.len());
        let sum: i32 = input[start..end].iter().map(|s| i32::from(*s)).sum();
        out.push((sum / (end - start) as i32) as i16);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_passthrough() {
        let s = [1i16, 2, 3];
        assert_eq!(to_mono(&s, 1), s);
    }

    #[test]
    fn stereo_averages_channel_pairs() {
        assert_eq!(to_mono(&[10, 20, 30, 40], 2), vec![15, 35]);
    }

    #[test]
    fn mono_conversion_does_not_overflow_at_full_scale() {
        assert_eq!(to_mono(&[i16::MAX, i16::MAX], 2), vec![i16::MAX]);
        assert_eq!(to_mono(&[i16::MIN, i16::MIN], 2), vec![i16::MIN]);
    }

    #[test]
    fn same_rate_is_a_passthrough() {
        let s = [1i16, 2, 3];
        assert_eq!(resample(&s, 16_000, 16_000), s);
    }

    #[test]
    fn downsampling_produces_the_expected_length() {
        let input = vec![0i16; 48_000];
        let out = resample(&input, 48_000, 16_000);
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn upsampling_produces_the_expected_length() {
        let input = vec![0i16; 8_000];
        let out = resample(&input, 8_000, 16_000);
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn preserves_a_constant_signal() {
        let input = vec![1000i16; 4_800];
        let out = resample(&input, 48_000, 16_000);
        // Edges soften slightly from the averaging window; the interior must
        // hold the original level.
        let mid = out[out.len() / 2];
        assert!((mid - 1000).abs() < 5, "midpoint drifted to {mid}");
    }

    #[test]
    fn attenuates_a_frequency_above_the_new_nyquist() {
        // 12 kHz sampled at 48 kHz is above the 8 kHz Nyquist limit of a
        // 16 kHz target. Without the low-pass it would alias down into the
        // speech band instead of being suppressed.
        let sample_rate = 48_000.0f32;
        let input: Vec<i16> = (0..48_000)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (10_000.0 * (2.0 * std::f32::consts::PI * 12_000.0 * t).sin()) as i16
            })
            .collect();

        let out = resample(&input, 48_000, 16_000);
        let rms =
            (out.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / out.len() as f64).sqrt();
        assert!(rms < 2_000.0, "aliased energy survived: rms {rms}");
    }

    /// The other half of the filter trade: suppressing 12 kHz is only useful
    /// if speech survives. 3.4 kHz is the top of the telephone band.
    #[test]
    fn preserves_the_speech_band() {
        let sample_rate = 48_000.0f32;
        let input: Vec<i16> = (0..48_000)
            .map(|i| {
                let t = i as f32 / sample_rate;
                (10_000.0 * (2.0 * std::f32::consts::PI * 3_400.0 * t).sin()) as i16
            })
            .collect();

        let input_rms =
            (input.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / input.len() as f64).sqrt();
        let out = resample(&input, 48_000, 16_000);
        let out_rms =
            (out.iter().map(|s| f64::from(*s).powi(2)).sum::<f64>() / out.len() as f64).sqrt();

        let retained = out_rms / input_rms;
        assert!(
            retained > 0.8,
            "speech band lost too much energy: retained {retained:.3}"
        );
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert!(resample(&[], 48_000, 16_000).is_empty());
    }

    #[test]
    fn moving_average_smooths_without_changing_length() {
        let input = vec![0i16, 100, 0, 100, 0, 100];
        let out = moving_average(&input, 3);
        assert_eq!(out.len(), input.len());
        assert!(out[2] < 100 && out[2] > 0);
    }
}
