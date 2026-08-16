# voxide

**Offline, semantic voice control for your development workflow.**

Say what you mean, not what the manual says.

```console
$ voxide say "make sure everything still passes"
   -> cargo.test  (0.91)
running 80 tests ...

$ voxide say "checkout the login branch"
   -> git.checkout  (0.94)   branch = "login"
Switched to branch 'login'
```

Neither of those phrasings appears anywhere in a config file.

---

## Why another voice-control tool?

Every established option — Talon, Cursorless, Numen, Dragonfly/Caster — matches
speech against a **grammar**. You define patterns; you then have to say one of
them. The near-universal complaint is the ramp: reviewers describe a fortnight
of frustration while a custom phonetic alphabet and command grammar become
second nature.

voxide replaces the grammar with a **sentence-embedding model**. Training
phrases and your utterance are both embedded into a vector space and compared
by cosine similarity, so a phrasing nobody wrote down still lands on the right
command.

This is a measurable claim, so voxide measures it. `voxide eval` scores both
backends against a golden corpus of 40 utterances written the way people
actually talk:

| backend | overall top-1 | on the 5 paraphrase cases |
| --- | --- | --- |
| `lexical` (token + trigram overlap — the conventional approach) | 87.5% | **0 / 5** |
| `semantic` (MiniLM embeddings) | 92.5% | 5 / 5 |
| `hybrid` (both, blended — the default with `--features embed`) | **100%** | **5 / 5** |

Each backend is scored at its own threshold, because the scales are not
comparable: lexical overlap is near-binary, cosine similarity is continuous.

The blend wins because the two backends fail in almost disjoint ways. Semantic
matching reads a `{slot}` marker as a literal word, so "switch to develop" is
scored on the branch name and drifts toward `cargo.build`; lexical matching
drops the marker and gets it right. Lexical cannot recognise a rewording it was
never shown; semantic can. Neither is a subset of the other.

The lexical baseline is not a strawman: it gets 35/35 on utterances close to a
training phrase. It fails **every one** of the five that drift — "make sure
everything still passes" reaches 0.054 against `cargo.test`, "clean up the
indentation" loses outright to `cargo.test.filter`. That cliff is the entire
problem, and it is what voxide is built to remove.

Run it yourself:

```console
$ voxide eval --compare --sweep
```

## Everything is local

No network calls, no telemetry, no account. Speech recognition, embeddings, and
matching all run on your machine. The only thing voxide ever downloads is the
model itself, once, via `voxide models pull` — pinned to an exact upstream
artifact and verified against a recorded SHA-256, so a corrupted or swapped
download is a named error rather than a model that quietly misbehaves.

That downloader is part of the binary. No Python, no `pip install`, no
virtualenv in the setup path.

---

## Status

Early, and honest about it. What works today:

| | |
| --- | --- |
| ✅ | Command packs, slot extraction, shell execution |
| ✅ | Lexical matcher, ranked results, `voxide why` |
| ✅ | Eval harness with threshold sweep and backend comparison |
| ✅ | Audio capture, resampling, VAD, pre-roll — all trait-based |
| ✅ | Wake word detection, pipeline state machine, `voxide run` |
| ✅ | Cross-platform CI, `cargo-deny` license gating, 145 tests |
| 🚧 | Semantic matcher and microphone backend — implemented, CI-verified (see *Development*) |
| ✅ | `voxide models pull` — pinned, checksummed, atomic, zip-slip safe |
| 📋 | Lua actions, keystroke injection, dictation mode |
| 📋 | Acoustic wake word, so the recogniser can idle until spoken to |

Text mode is not scaffolding that goes away: it is how the eval harness runs,
how packs get debugged, and a usable entry point in its own right.

---

## Install

```console
$ git clone https://github.com/MansurPro/voxide
$ cd voxide
$ cargo install --path crates/voxide --features embed
```

Omit `--features embed` for a dependency-free build that uses lexical matching.

## Usage

```console
$ voxide packs list                    # what commands exist
$ voxide say "run the linter"          # match and execute
$ voxide say --dry-run "..."           # show the command line, run nothing
$ voxide why "..."                     # ranked candidates and why they scored
$ voxide eval --compare --sweep        # score the matcher against the corpus
$ voxide run --wake voxide             # listen on the microphone
$ voxide run --from recording.wav      # replay a recording through the pipeline
$ voxide models list                   # what models are available / installed
$ voxide models pull vosk-en-small     # fetch one, verified against a pinned hash
```

`voxide run` needs a build with `--features mic,vosk` and a speech model. Without
them it says so and tells you what to do, rather than failing obscurely.

`voxide why` is the debugging tool. It shows the ranking, which training phrase
was responsible, and the margin over the runner-up:

```console
$ voxide why "what changed"
phrase:    "what changed"
backend:   lexical
threshold: 0.62

-> 1.000  git.status     via "what changed"
   0.410  git.diff       via "what did I change"
   0.233  git.log        via "what's the history"

margin over runner-up: 0.590
```

A narrow margin is the signal to add a distinguishing phrase to the pack.

## Writing a pack

A pack is one TOML file. Phrases are training examples, not a grammar — three
or four naturally different wordings is plenty.

```toml
[pack]
name = "docker"

[[command]]
id = "docker.ps"
description = "List running containers"
phrases = ["what containers are running", "show me the containers"]
action = { type = "shell", run = "docker ps" }

[[command]]
id = "docker.logs"
phrases = ["show logs for {service}", "tail the {service} logs"]
slots = [{ name = "service", entity = "container or service name" }]
action = { type = "shell", run = "docker compose logs -f --tail=100 {{service}}" }
```

Drop it in `./packs/`, `$VOXIDE_PACKS`, or `~/.config/voxide/packs/`.

A malformed pack is a **hard error naming the file and the problem**. Packs are
never silently skipped — that failure mode wastes hours.

---

## Architecture

```
microphone ──► VAD ──► speech-to-text ──► wake word ──► matcher ──► slots ──► action
  (cpal)      (energy)      (vosk)        (transcript)  (embedding)         (shell)
```

Every stage is a trait, and that is load-bearing rather than decorative:

```rust
pub trait AudioSource  { fn next_frame(&mut self, out: &mut [i16]) -> Result<usize>; }
pub trait Transcriber  { fn accept(&mut self, samples: &[i16]) -> Option<Utterance>; }
pub trait Matcher      { fn rank(&self, text: &str, limit: usize) -> Vec<Match>; }
pub trait Executor     { fn execute(&self, cmd: &LoadedCommand, slots: &Slots) -> ...; }
```

Because `AudioSource` has a `WavSource` implementation, the entire pipeline is
testable from recorded fixtures — no microphone, no CI hardware, no flakes.
Backends that need native libraries (`mic`, `vosk`, `embed`) are Cargo features
that are **off by default**, so `cargo test` works anywhere.

### Crates

| crate | role |
| --- | --- |
| `voxide-core` | Pack format, slots, action specs, template rendering |
| `voxide-audio` | `AudioSource` trait, WAV and microphone sources, resampling, VAD |
| `voxide-asr` | `Transcriber` trait, Vosk backend, scriptable mock |
| `voxide-wake` | `WakeDetector` trait, always-on and transcript spotting |
| `voxide-intent` | `Matcher` trait, lexical baseline, semantic and hybrid backends, slot extraction |
| `voxide-models` | Pinned, checksum-verified model downloads and zip extraction |
| `voxide-actions` | `Executor` trait, shell backend with real deadlines |
| `voxide-pipeline` | The listening state machine. Emits events, performs no I/O |
| `voxide` | CLI |

## Development

```console
$ cargo test --workspace      # no native dependencies needed
$ cargo clippy --workspace --all-targets -- -D warnings
$ cargo run -p voxide -- eval --sweep
```

Three backends **cannot be compiled in the primary development environment**
and are therefore verified only by CI. This is deliberate, and those CI jobs
gate the merge:

- **`embed`** links ONNX Runtime, whose prebuilt binaries come from a CDN that
  the development sandbox's network policy blocks.
- **`mic`** needs ALSA headers, which that environment does not provide.
- **`vosk`** links `libvosk`, a native library installed separately.

If one of those jobs goes red, the corresponding feature is broken — a green
`test` job does not cover them.

## Acknowledgements

The approach to embedding-based intent matching with a fingerprint-keyed vector
cache, and the shape of a capability-tiered scripting sandbox, were informed by
the author's earlier work on a fork of [Priler/jarvis](https://github.com/Priler/jarvis)
by Abraham Tugalov. voxide shares no source code with that project and is
licensed independently. See [`NOTICE`](NOTICE).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.
