use crate::{FRAME_LEN, SAMPLE_RATE, resample};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("audio io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to read wav file {path}: {source}")]
    Wav {
        path: String,
        #[source]
        source: Box<hound::Error>,
    },

    #[error("unsupported sample format: {0}")]
    Format(String),

    #[error("no input device available")]
    NoDevice,

    #[error("audio backend error: {0}")]
    Backend(String),
}

/// A source of 16 kHz mono PCM frames.
///
/// `next_frame` fills `out` and returns how many samples it wrote. A return of
/// zero means the source is exhausted, which for a file means end-of-input and
/// for a microphone never happens.
pub trait AudioSource: Send {
    fn next_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError>;

    fn frame_len(&self) -> usize {
        FRAME_LEN
    }

    /// Human-readable description, used in logs and `--help` output.
    fn describe(&self) -> String;
}

/// Silence. Useful for exercising timing paths without any signal.
#[derive(Debug)]
pub struct NullSource {
    frame_len: usize,
    remaining: Option<usize>,
}

impl NullSource {
    /// Produces silence forever.
    pub fn new() -> Self {
        Self {
            frame_len: FRAME_LEN,
            remaining: None,
        }
    }

    /// Produces `frames` of silence, then reports exhaustion.
    pub fn with_frames(frames: usize) -> Self {
        Self {
            frame_len: FRAME_LEN,
            remaining: Some(frames),
        }
    }
}

impl Default for NullSource {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioSource for NullSource {
    fn next_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError> {
        if let Some(remaining) = &mut self.remaining {
            if *remaining == 0 {
                return Ok(0);
            }
            *remaining -= 1;
        }
        let n = out.len().min(self.frame_len);
        out[..n].fill(0);
        Ok(n)
    }

    fn frame_len(&self) -> usize {
        self.frame_len
    }

    fn describe(&self) -> String {
        "null (silence)".to_owned()
    }
}

/// Plays a WAV file as if it were a microphone.
///
/// This is what makes the pipeline testable. A recorded utterance drives
/// exactly the same code path as live capture, so wake-word thresholds, VAD
/// timing, and end-to-end behaviour can all be asserted in CI with no audio
/// hardware and no flakiness.
#[derive(Debug)]
pub struct WavSource {
    samples: Vec<i16>,
    position: usize,
    frame_len: usize,
    label: String,
}

impl WavSource {
    /// Loads a WAV file, converting to 16 kHz mono.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AudioError> {
        let path = path.as_ref();
        let reader = hound::WavReader::open(path).map_err(|source| AudioError::Wav {
            path: path.display().to_string(),
            source: Box::new(source),
        })?;

        let spec = reader.spec();
        let raw = read_samples(reader, &spec)?;
        let mono = resample::to_mono(&raw, spec.channels);
        let samples = resample::resample(&mono, spec.sample_rate, SAMPLE_RATE);

        tracing::debug!(
            path = %path.display(),
            source_rate = spec.sample_rate,
            channels = spec.channels,
            samples = samples.len(),
            "loaded wav"
        );

        Ok(Self {
            samples,
            position: 0,
            frame_len: FRAME_LEN,
            label: path.display().to_string(),
        })
    }

    /// Builds a source directly from 16 kHz mono samples. Intended for tests.
    pub fn from_samples(samples: Vec<i16>) -> Self {
        Self {
            samples,
            position: 0,
            frame_len: FRAME_LEN,
            label: "in-memory".to_owned(),
        }
    }

    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / SAMPLE_RATE as f32
    }
}

impl AudioSource for WavSource {
    fn next_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError> {
        if self.position >= self.samples.len() {
            return Ok(0);
        }

        let want = out.len().min(self.frame_len);
        let end = (self.position + want).min(self.samples.len());
        let n = end - self.position;

        out[..n].copy_from_slice(&self.samples[self.position..end]);
        // Zero-pad a short final frame so consumers always see a full buffer.
        if n < want {
            out[n..want].fill(0);
        }
        self.position = end;

        Ok(n)
    }

    fn frame_len(&self) -> usize {
        self.frame_len
    }

    fn describe(&self) -> String {
        format!("wav {} ({:.2}s)", self.label, self.duration_secs())
    }
}

fn read_samples(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
    spec: &hound::WavSpec,
) -> Result<Vec<i16>, AudioError> {
    let mut reader = reader;
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => Ok(reader
            .samples::<i16>()
            .filter_map(std::result::Result::ok)
            .collect()),
        (hound::SampleFormat::Int, 8) => Ok(reader
            .samples::<i32>()
            .filter_map(std::result::Result::ok)
            // 8-bit WAV is unsigned with a midpoint of 128.
            .map(|s| ((s - 128) * 256) as i16)
            .collect()),
        (hound::SampleFormat::Int, 24 | 32) => {
            let shift = spec.bits_per_sample - 16;
            Ok(reader
                .samples::<i32>()
                .filter_map(std::result::Result::ok)
                .map(|s| (s >> shift) as i16)
                .collect())
        }
        (hound::SampleFormat::Float, _) => Ok(reader
            .samples::<f32>()
            .filter_map(std::result::Result::ok)
            .map(|s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
            .collect()),
        (format, bits) => Err(AudioError::Format(format!("{format:?}/{bits}-bit"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_wav(path: &Path, spec: hound::WavSpec, samples: &[i16]) {
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for s in samples {
            w.write_sample(*s).unwrap();
        }
        w.finalize().unwrap();
    }

    fn spec_16k_mono() -> hound::WavSpec {
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        }
    }

    #[test]
    fn null_source_yields_silence_forever() {
        let mut s = NullSource::new();
        let mut buf = [7i16; FRAME_LEN];
        assert_eq!(s.next_frame(&mut buf).unwrap(), FRAME_LEN);
        assert!(buf.iter().all(|x| *x == 0));
    }

    #[test]
    fn bounded_null_source_reports_exhaustion() {
        let mut s = NullSource::with_frames(2);
        let mut buf = [0i16; FRAME_LEN];
        assert_eq!(s.next_frame(&mut buf).unwrap(), FRAME_LEN);
        assert_eq!(s.next_frame(&mut buf).unwrap(), FRAME_LEN);
        assert_eq!(s.next_frame(&mut buf).unwrap(), 0);
    }

    #[test]
    fn wav_source_reads_frames_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("a.wav");
        let samples: Vec<i16> = (0..FRAME_LEN as i16 * 2).collect();
        write_wav(&path, spec_16k_mono(), &samples);

        let mut src = WavSource::open(&path).unwrap();
        let mut buf = [0i16; FRAME_LEN];

        assert_eq!(src.next_frame(&mut buf).unwrap(), FRAME_LEN);
        assert_eq!(buf[0], 0);
        assert_eq!(buf[1], 1);

        assert_eq!(src.next_frame(&mut buf).unwrap(), FRAME_LEN);
        assert_eq!(buf[0], FRAME_LEN as i16);

        assert_eq!(src.next_frame(&mut buf).unwrap(), 0);
    }

    #[test]
    fn short_final_frame_is_zero_padded() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("short.wav");
        write_wav(&path, spec_16k_mono(), &[100i16; 10]);

        let mut src = WavSource::open(&path).unwrap();
        let mut buf = [-1i16; FRAME_LEN];
        assert_eq!(src.next_frame(&mut buf).unwrap(), 10);
        assert_eq!(buf[9], 100);
        assert_eq!(buf[10], 0, "tail should be zero-padded, not stale");
    }

    #[test]
    fn resamples_a_non_16k_file_on_load() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("48k.wav");
        let spec = hound::WavSpec {
            sample_rate: 48_000,
            ..spec_16k_mono()
        };
        write_wav(&path, spec, &vec![0i16; 48_000]);

        let src = WavSource::open(&path).unwrap();
        assert_eq!(src.total_samples(), 16_000);
        assert!((src.duration_secs() - 1.0).abs() < 0.01);
    }

    #[test]
    fn downmixes_stereo_on_load() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            ..spec_16k_mono()
        };
        // Interleaved L/R pairs; each pair averages to 15.
        write_wav(&path, spec, &[10, 20, 10, 20, 10, 20, 10, 20]);

        let src = WavSource::open(&path).unwrap();
        assert_eq!(src.total_samples(), 4);
    }

    #[test]
    fn missing_file_is_a_named_error() {
        let err = WavSource::open("/nonexistent/voxide/x.wav").unwrap_err();
        assert!(err.to_string().contains("x.wav"), "got: {err}");
    }

    #[test]
    fn describe_mentions_the_duration() {
        let src = WavSource::from_samples(vec![0; SAMPLE_RATE as usize]);
        assert!(src.describe().contains("1.00s"), "got {}", src.describe());
    }
}
