//! Sentence-embedding intent matching.
//!
//! This is the reason voxide exists. Conventional voice-control tools compile
//! their phrases into a grammar, so an utterance either fits a pattern the
//! author wrote or it does not match at all. Users therefore have to memorise
//! phrasings, and the usual complaint about such tools is the fortnight it
//! takes before that becomes second nature.
//!
//! Here, phrases are embedded into a vector space and compared by cosine
//! similarity, so "check if it builds" lands on `cargo.check` without anyone
//! having written that wording down.

use crate::matcher::{Match, Matcher, normalize};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use voxide_core::{CommandSet, template};

/// Model used for phrase and query embeddings.
///
/// A 384-dimension MiniLM is the right trade here: large enough that
/// paraphrase similarity is meaningful, small enough (~90 MB) that a first run
/// is not a hostile download, and quantised ONNX so inference stays in the
/// low single-digit milliseconds on CPU.
const MODEL: fastembed::EmbeddingModel = fastembed::EmbeddingModel::AllMiniLML6V2;

#[derive(Debug, thiserror::Error)]
pub enum SemanticError {
    #[error("failed to initialise the embedding model: {0}")]
    Model(String),

    #[error("failed to embed text: {0}")]
    Embed(String),

    #[error("no commands to index")]
    Empty,

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// One indexed training phrase and its unit-length embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    id: String,
    phrase: String,
    vector: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Cache {
    /// Fingerprint of the command set these vectors were built from.
    fingerprint: String,
    model: String,
    entries: Vec<Entry>,
}

pub struct SemanticMatcher {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    entries: Vec<Entry>,
}

impl SemanticMatcher {
    /// Builds or loads the index for `commands`.
    ///
    /// Vectors are cached on disk under the user's cache directory and keyed
    /// by a fingerprint of the command ids and phrases, so editing a shell
    /// flag or a description costs nothing while editing a phrase re-embeds.
    pub fn load(commands: &CommandSet, lang: &str) -> Result<Self, SemanticError> {
        if commands.is_empty() {
            return Err(SemanticError::Empty);
        }

        let fingerprint = commands.fingerprint(lang);
        let cache_path = cache_path();

        // fastembed defaults its model cache to `./.fastembed_cache`, relative
        // to the *current directory*. For an installed binary that means
        // re-downloading ~90 MB into every project a user speaks a command in,
        // and silently degrading to lexical matching wherever the cwd is not
        // writable. Pin it next to the phrase vectors instead, so one download
        // serves the whole machine.
        let mut model = fastembed::TextEmbedding::try_new(
            fastembed::TextInitOptions::new(MODEL)
                .with_cache_dir(model_cache_dir())
                .with_show_download_progress(true),
        )
        .map_err(|e| SemanticError::Model(e.to_string()))?;

        if let Some(cached) = read_cache(&cache_path, &fingerprint) {
            tracing::debug!(entries = cached.len(), "loaded cached phrase vectors");
            return Ok(Self {
                model: std::sync::Mutex::new(model),
                entries: cached,
            });
        }

        let entries = build_index(&mut model, commands, lang)?;
        write_cache(&cache_path, &fingerprint, &entries);

        Ok(Self {
            model: std::sync::Mutex::new(model),
            entries,
        })
    }
}

/// Score below which a semantic match is treated as "no match".
///
/// Measured, not guessed. On the golden corpus, off-domain speech ("play some
/// jazz music", "how tall is the eiffel tower") peaks at 0.297 against the
/// bundled packs, while the weakest true paraphrase scores 0.359. This sits in
/// that gap. `voxide eval --sweep` re-derives it for a different pack set.
pub const SEMANTIC_THRESHOLD: f32 = 0.35;

// Guards the regression this constant exists to prevent: reusing the lexical
// floor here rejected 6 of 40 correct matches and put the semantic backend 10
// points *below* the baseline it is meant to beat. The window is measured, so
// re-derive it with `voxide eval --sweep` before moving either bound.
const _: () = {
    assert!(SEMANTIC_THRESHOLD > 0.297, "admits off-domain speech");
    assert!(SEMANTIC_THRESHOLD < 0.359, "rejects known-good paraphrases");
    assert!(SEMANTIC_THRESHOLD < crate::DEFAULT_THRESHOLD);
};

impl Matcher for SemanticMatcher {
    fn backend(&self) -> &'static str {
        "semantic"
    }

    fn default_threshold(&self) -> f32 {
        SEMANTIC_THRESHOLD
    }

    fn rank(&self, text: &str, limit: usize) -> Vec<Match> {
        let norm = normalize(text);
        if norm.is_empty() {
            return Vec::new();
        }

        let query = {
            let mut model = match self.model.lock() {
                Ok(m) => m,
                Err(poisoned) => poisoned.into_inner(),
            };
            match model.embed(vec![norm.as_ref()], None) {
                Ok(mut v) if !v.is_empty() => unit(v.swap_remove(0)),
                Ok(_) => return Vec::new(),
                Err(e) => {
                    tracing::error!(error = %e, "failed to embed query");
                    return Vec::new();
                }
            }
        };

        // Best-scoring phrase per command, rather than one averaged vector per
        // command. Averaging blurs a command whose phrasings are deliberately
        // varied, and it discards which phrase was responsible -- which is
        // exactly what `voxide why` needs to show.
        let mut best: Vec<Match> = Vec::new();
        for entry in &self.entries {
            let score = cosine(&query, &entry.vector);
            match best.iter_mut().find(|m| m.id == entry.id) {
                Some(existing) if existing.score < score => {
                    existing.score = score;
                    existing.via = Some(entry.phrase.clone());
                }
                Some(_) => {}
                None => best.push(Match {
                    id: entry.id.clone(),
                    score,
                    via: Some(entry.phrase.clone()),
                }),
            }
        }

        best.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        best.truncate(limit);
        best
    }
}

fn build_index(
    model: &mut fastembed::TextEmbedding,
    commands: &CommandSet,
    lang: &str,
) -> Result<Vec<Entry>, SemanticError> {
    let mut ids = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut originals = Vec::new();

    for lc in commands.commands() {
        for phrase in lc.command.phrases.for_lang(lang) {
            // Keep the slot name as a word here, unlike lexical matching: a
            // sentence encoder benefits from "run the filter tests" reading as
            // a grammatical sentence rather than "run the tests".
            let cleaned = template::strip_slot_markers(phrase);
            let norm = normalize(&cleaned).into_owned();
            if norm.is_empty() {
                continue;
            }
            ids.push(lc.command.id.clone());
            originals.push(phrase.to_owned());
            texts.push(norm);
        }
    }

    if texts.is_empty() {
        return Err(SemanticError::Empty);
    }

    tracing::info!(phrases = texts.len(), "building phrase index");
    let vectors = model
        .embed(texts, None)
        .map_err(|e| SemanticError::Embed(e.to_string()))?;

    Ok(vectors
        .into_iter()
        .zip(ids)
        .zip(originals)
        .map(|((vector, id), phrase)| Entry {
            id,
            phrase,
            vector: unit(vector),
        })
        .collect())
}

/// Scales a vector to unit length so cosine similarity is a plain dot product.
fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Dot product of two unit vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Per-user cache root: `~/.cache/voxide`, or the platform equivalent.
fn cache_root() -> PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);

    #[cfg(not(windows))]
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")));

    base.unwrap_or_else(std::env::temp_dir).join("voxide")
}

/// Where the ONNX weights and tokenizer are kept.
fn model_cache_dir() -> PathBuf {
    cache_root().join("models")
}

fn cache_path() -> PathBuf {
    cache_root().join("phrase-vectors.json")
}

fn read_cache(path: &Path, fingerprint: &str) -> Option<Vec<Entry>> {
    let text = std::fs::read_to_string(path).ok()?;
    let cache: Cache = serde_json::from_str(&text).ok()?;
    // A stale cache is not an error worth reporting; it just gets rebuilt.
    if cache.fingerprint != fingerprint || cache.model != format!("{MODEL:?}") {
        return None;
    }
    Some(cache.entries)
}

fn write_cache(path: &Path, fingerprint: &str, entries: &[Entry]) {
    let cache = Cache {
        fingerprint: fingerprint.to_owned(),
        model: format!("{MODEL:?}"),
        entries: entries.to_vec(),
    };

    // Cache failures must never break matching; the index is already in memory.
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        tracing::debug!(error = %e, "could not create cache directory");
        return;
    }
    match serde_json::to_vec(&cache) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(path, bytes) {
                tracing::debug!(error = %e, "could not write phrase vector cache");
            }
        }
        Err(e) => tracing::debug!(error = %e, "could not serialise phrase vector cache"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_normalises_to_length_one() {
        let v = unit(vec![3.0, 4.0]);
        let len = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6, "length was {len}");
    }

    #[test]
    fn unit_leaves_a_zero_vector_alone() {
        assert_eq!(unit(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_of_identical_unit_vectors_is_one() {
        let a = unit(vec![1.0, 2.0, 3.0]);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn stale_cache_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("c.json");
        write_cache(&path, "fingerprint-a", &[]);

        assert!(read_cache(&path, "fingerprint-a").is_some());
        assert!(read_cache(&path, "fingerprint-b").is_none());
    }

    #[test]
    fn missing_cache_reads_as_none() {
        assert!(read_cache(Path::new("/nonexistent/voxide/c.json"), "x").is_none());
    }
}
