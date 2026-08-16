//! The `voxide run` listening loop.

use anyhow::{Result, bail};
use std::path::Path;
use voxide_actions::{DefaultExecutor, Executor};
use voxide_asr::Transcriber;
use voxide_audio::{AudioSource, WavSource};
use voxide_core::CommandSet;
use voxide_intent::Matcher;
use voxide_pipeline::{Config, Pipeline, PipelineEvent};
use voxide_wake::{AlwaysOn, TranscriptSpotter, WakeDetector};

/// Where audio comes from.
pub enum Input<'a> {
    Wav(&'a Path),
    /// Device name substring; `None` selects the system default. The field is
    /// only read by the `mic` backend, so it is dead code without that feature.
    Mic(#[cfg_attr(not(feature = "mic"), allow(dead_code))] Option<&'a str>),
}

pub struct Options<'a> {
    pub input: Input<'a>,
    pub wake_word: Option<&'a str>,
    pub model: Option<&'a Path>,
    pub dry_run: bool,
    pub config: Config,
}

/// Runs the pipeline until the audio source is exhausted.
pub fn run(set: &CommandSet, matcher: &dyn Matcher, opts: Options<'_>) -> Result<()> {
    let source = build_source(&opts.input)?;
    let transcriber = build_transcriber(opts.model)?;
    let wake = build_wake(opts.wake_word);

    println!("source:      {}", source.describe());
    println!("recogniser:  {}", transcriber.backend());
    println!("wake:        {}", wake.name());
    println!("matcher:     {}", matcher.backend());
    println!("threshold:   {:.2}\n", opts.config.threshold);

    let executor = if opts.dry_run {
        DefaultExecutor::dry_run()
    } else {
        DefaultExecutor::new()
    };

    let mut pipeline = Pipeline::new(
        source,
        transcriber,
        wake,
        matcher,
        Box::new(voxide_audio::EnergyVad::with_defaults()),
        opts.config,
    );

    let mut events = Vec::new();
    loop {
        events.clear();
        let more = pipeline.step(&mut events)?;

        for event in &events {
            report(event, set, &executor, opts.dry_run);
        }

        if !more {
            break;
        }
    }

    Ok(())
}

fn report(event: &PipelineEvent, set: &CommandSet, executor: &DefaultExecutor, dry_run: bool) {
    match event {
        PipelineEvent::SpeechStarted => println!("  [listening]"),
        PipelineEvent::SpeechEnded { .. } => println!("  [done]"),
        PipelineEvent::WokeUp => println!("  [awake]"),
        PipelineEvent::Transcribed { text, confidence } => {
            println!("  heard: {text:?} ({confidence:.2})");
        }
        PipelineEvent::Ignored { .. } => {}
        PipelineEvent::NoMatch { text, best } => match best {
            Some((id, score)) => {
                println!("  no match for {text:?} (closest {id} at {score:.3})");
            }
            None => println!("  no match for {text:?}"),
        },
        PipelineEvent::Matched {
            id, score, slots, ..
        } => {
            println!("  -> {id} ({score:.3})");

            let Some(command) = set.get(id) else {
                eprintln!("  matcher returned unknown command id {id:?}");
                return;
            };

            match executor.execute(command, slots) {
                Ok(outcome) if dry_run => println!("     would run: {}", outcome.stdout.trim()),
                Ok(outcome) => {
                    if !outcome.stdout.is_empty() {
                        print!("{}", outcome.stdout);
                    }
                    if !outcome.stderr.is_empty() {
                        eprint!("{}", outcome.stderr);
                    }
                    if !outcome.success() {
                        eprintln!("     exited with {:?}", outcome.code);
                    }
                }
                Err(e) => eprintln!("     {e}"),
            }
        }
        PipelineEvent::SourceExhausted => {}
    }
}

fn build_source<'a>(input: &Input<'a>) -> Result<Box<dyn AudioSource + 'a>> {
    match input {
        Input::Wav(path) => Ok(Box::new(WavSource::open(path)?)),

        #[cfg(feature = "mic")]
        Input::Mic(device) => Ok(Box::new(voxide_audio::MicSource::open(*device)?)),

        #[cfg(not(feature = "mic"))]
        Input::Mic(_) => bail!(
            "this build has no microphone support.\n\
             Rebuild with `--features mic` (needs ALSA development headers on Linux), \
             or feed a recording with `--from <file.wav>`."
        ),
    }
}

fn build_transcriber<'a>(model: Option<&Path>) -> Result<Box<dyn Transcriber + 'a>> {
    #[cfg(feature = "vosk")]
    {
        let dir = match model {
            Some(p) => p.to_path_buf(),
            None => voxide_asr::vosk_backend::find_model("models").ok_or_else(|| {
                anyhow::anyhow!(
                    "no speech model found under ./models.\n\
                     Pass one with `--model <dir>`, or fetch one from \
                     https://alphacephei.com/vosk/models"
                )
            })?,
        };
        return Ok(Box::new(voxide_asr::VoskTranscriber::open(dir)?));
    }

    #[cfg(not(feature = "vosk"))]
    {
        let _ = model;
        bail!(
            "this build has no speech recognition.\n\
             Rebuild with `--features vosk` (needs libvosk installed), or drive \
             voxide from text with `voxide say \"...\"`."
        )
    }
}

fn build_wake<'a>(wake_word: Option<&str>) -> Box<dyn WakeDetector + 'a> {
    match wake_word {
        Some(word) => {
            // Recognisers mangle names, so accept the word with and without an
            // internal space as a cheap tolerance for mishearings.
            let mut variants = vec![word.to_owned()];
            if let Some(split) = word.char_indices().nth(word.chars().count() / 2) {
                variants.push(format!("{} {}", &word[..split.0], &word[split.0..]));
            }
            Box::new(TranscriptSpotter::new(variants))
        }
        None => Box::new(AlwaysOn),
    }
}
