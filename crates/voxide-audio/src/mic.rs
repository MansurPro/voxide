//! Live microphone capture via cpal.
//!
//! Compiled only with the `mic` feature. On Linux it needs ALSA development
//! headers, which the primary development environment for this project does
//! not have, so this module is verified exclusively by CI. Keep it simple and
//! keep the logic that can be tested without hardware in [`crate::resample`]
//! and [`crate::pipeline`], which are covered by the default test suite.

use crate::source::{AudioError, AudioSource};
use crate::{FRAME_LEN, SAMPLE_RATE, resample};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::time::Duration;

/// How long `next_frame` waits for audio before reporting a backend error.
/// Generous: a busy machine can stall a callback well past a frame period.
const RECV_TIMEOUT: Duration = Duration::from_secs(5);

/// Bounded so a consumer that stalls drops old audio instead of growing the
/// queue without limit. Roughly two seconds at the default frame length.
const QUEUE_FRAMES: usize = 64;

/// Names of available input devices, for `--device` and diagnostics.
pub fn list_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices.filter_map(|d| d.name().ok()).collect(),
        Err(e) => {
            tracing::warn!(error = %e, "could not enumerate input devices");
            Vec::new()
        }
    }
}

/// A microphone, delivering 16 kHz mono frames.
pub struct MicSource {
    frames: Receiver<Vec<i16>>,
    /// Dropping this tells the capture thread to tear the stream down.
    _shutdown: ShutdownGuard,
    device_name: String,
    device_rate: u32,
}

struct ShutdownGuard(Option<std::sync::mpsc::Sender<()>>);

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}

impl MicSource {
    /// Opens the default input device, or the first whose name contains
    /// `preferred`.
    ///
    /// The cpal stream is built and owned by a dedicated thread because
    /// `cpal::Stream` is not `Send` on most platforms, and [`AudioSource`]
    /// must be. Samples cross the boundary over a bounded channel.
    pub fn open(preferred: Option<&str>) -> Result<Self, AudioError> {
        let (frame_tx, frame_rx) = sync_channel::<Vec<i16>>(QUEUE_FRAMES);
        let (shutdown_tx, shutdown_rx) = std::sync::mpsc::channel::<()>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(String, u32), String>>();

        let preferred = preferred.map(str::to_owned);

        std::thread::Builder::new()
            .name("voxide-mic".into())
            .spawn(move || {
                match build_stream(preferred.as_deref(), frame_tx) {
                    Ok((stream, name, rate)) => {
                        if let Err(e) = stream.play() {
                            let _ = ready_tx.send(Err(format!("failed to start stream: {e}")));
                            return;
                        }
                        let _ = ready_tx.send(Ok((name, rate)));
                        // Park until dropped. `stream` must stay alive in this
                        // thread for capture to continue.
                        let _ = shutdown_rx.recv();
                        drop(stream);
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.to_string()));
                    }
                }
            })
            .map_err(AudioError::Io)?;

        let (device_name, device_rate) = ready_rx
            .recv()
            .map_err(|_| AudioError::Backend("capture thread died during startup".into()))?
            .map_err(AudioError::Backend)?;

        tracing::info!(device = %device_name, rate = device_rate, "microphone open");

        Ok(Self {
            frames: frame_rx,
            _shutdown: ShutdownGuard(Some(shutdown_tx)),
            device_name,
            device_rate,
        })
    }
}

fn build_stream(
    preferred: Option<&str>,
    frame_tx: SyncSender<Vec<i16>>,
) -> Result<(cpal::Stream, String, u32), AudioError> {
    let host = cpal::default_host();

    let device = match preferred {
        Some(want) => host
            .input_devices()
            .map_err(|e| AudioError::Backend(e.to_string()))?
            .find(|d| {
                d.name()
                    .map(|n| n.to_lowercase().contains(&want.to_lowercase()))
                    .unwrap_or(false)
            })
            .ok_or(AudioError::NoDevice)?,
        None => host.default_input_device().ok_or(AudioError::NoDevice)?,
    };

    let name = device.name().unwrap_or_else(|_| "unknown".into());
    let config = device
        .default_input_config()
        .map_err(|e| AudioError::Backend(e.to_string()))?;

    let device_rate = config.sample_rate().0;
    let channels = config.channels();
    let sample_format = config.sample_format();
    let stream_config: cpal::StreamConfig = config.into();

    // Samples the device must supply to yield one 16 kHz output frame.
    let input_per_frame = (FRAME_LEN as u64 * device_rate as u64).div_ceil(SAMPLE_RATE as u64);

    let mut pending: Vec<i16> = Vec::with_capacity(input_per_frame as usize * 2);
    let mut emit = move |mono: Vec<i16>| {
        pending.extend_from_slice(&mono);

        while pending.len() >= input_per_frame as usize {
            let chunk: Vec<i16> = pending.drain(..input_per_frame as usize).collect();
            let mut resampled = resample::resample(&chunk, device_rate, SAMPLE_RATE);
            resampled.resize(FRAME_LEN, 0);

            // Bounded channel: if the consumer has fallen behind, drop this
            // frame rather than block the audio callback. Blocking here would
            // glitch capture for every other client of the device too.
            if frame_tx.try_send(resampled).is_err() {
                tracing::trace!("dropped a frame; consumer is behind");
            }
        }
    };

    let on_error = |e| tracing::error!(error = %e, "audio stream error");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let ints: Vec<i16> = data
                    .iter()
                    .map(|s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16)
                    .collect();
                emit(resample::to_mono(&ints, channels));
            },
            on_error,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                emit(resample::to_mono(data, channels));
            },
            on_error,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let ints: Vec<i16> = data
                    .iter()
                    .map(|s| (i32::from(*s) - 32_768) as i16)
                    .collect();
                emit(resample::to_mono(&ints, channels));
            },
            on_error,
            None,
        ),
        other => {
            return Err(AudioError::Format(format!("{other:?}")));
        }
    }
    .map_err(|e| AudioError::Backend(e.to_string()))?;

    Ok((stream, name, device_rate))
}

impl AudioSource for MicSource {
    fn next_frame(&mut self, out: &mut [i16]) -> Result<usize, AudioError> {
        match self.frames.recv_timeout(RECV_TIMEOUT) {
            Ok(frame) => {
                let n = out.len().min(frame.len());
                out[..n].copy_from_slice(&frame[..n]);
                Ok(n)
            }
            Err(RecvTimeoutError::Timeout) => Err(AudioError::Backend(format!(
                "no audio from {} for {RECV_TIMEOUT:?}",
                self.device_name
            ))),
            // The capture thread is gone, so the source is genuinely exhausted.
            Err(RecvTimeoutError::Disconnected) => Ok(0),
        }
    }

    fn describe(&self) -> String {
        format!("microphone {} ({} Hz)", self.device_name, self.device_rate)
    }
}
