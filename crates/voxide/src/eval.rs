//! Scoring a matcher against a golden phrase corpus.
//!
//! The claim voxide makes is that semantic matching removes the need to
//! memorise phrasings. That is testable, so it is tested: the corpus contains
//! utterances written the way people speak rather than copied from the packs,
//! and `--compare` scores both backends on it side by side.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use voxide_intent::Matcher;

#[derive(Debug, Deserialize)]
struct CorpusFile {
    #[serde(default, rename = "case")]
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub say: String,
    pub expect: String,
    #[serde(default)]
    pub note: Option<String>,
}

/// Outcome of scoring one backend over the corpus.
#[derive(Debug, Clone)]
pub struct Report {
    pub backend: String,
    pub total: usize,
    pub top1: usize,
    pub top3: usize,
    /// Cases whose top-1 was wrong or below threshold.
    pub failures: Vec<Failure>,
    /// Mean score of the correct answer when it ranked first.
    pub mean_correct_score: f32,
}

#[derive(Debug, Clone)]
pub struct Failure {
    pub say: String,
    pub expected: String,
    pub got: Option<String>,
    pub got_score: f32,
    /// Score the expected command received, wherever it ranked.
    pub expected_score: f32,
    pub note: Option<String>,
}

impl Report {
    pub fn top1_accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.top1 as f32 / self.total as f32
        }
    }

    pub fn top3_accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.top3 as f32 / self.total as f32
        }
    }
}

/// Loads every `*.toml` corpus file under `dir`.
pub fn load_corpus(dir: &Path) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading corpus directory {}", dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "toml"))
        .collect();
    files.sort();

    for path in files {
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let parsed: CorpusFile =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cases.extend(parsed.cases);
    }

    Ok(cases)
}

/// Scores `matcher` against `cases`.
pub fn run(matcher: &dyn Matcher, cases: &[Case], threshold: f32) -> Report {
    let mut top1 = 0;
    let mut top3 = 0;
    let mut failures = Vec::new();
    let mut correct_scores = Vec::new();

    for case in cases {
        let ranked = matcher.rank(&case.say, 3);
        let expected_score = ranked
            .iter()
            .find(|m| m.id == case.expect)
            .map(|m| m.score)
            .unwrap_or(0.0);

        let hit_top1 = ranked
            .first()
            .is_some_and(|m| m.id == case.expect && m.score >= threshold);

        if hit_top1 {
            top1 += 1;
            correct_scores.push(expected_score);
        } else {
            failures.push(Failure {
                say: case.say.clone(),
                expected: case.expect.clone(),
                got: ranked.first().map(|m| m.id.clone()),
                got_score: ranked.first().map(|m| m.score).unwrap_or(0.0),
                expected_score,
                note: case.note.clone(),
            });
        }

        if ranked
            .iter()
            .any(|m| m.id == case.expect && m.score >= threshold)
        {
            top3 += 1;
        }
    }

    let mean_correct_score = if correct_scores.is_empty() {
        0.0
    } else {
        correct_scores.iter().sum::<f32>() / correct_scores.len() as f32
    };

    Report {
        backend: matcher.backend().to_owned(),
        total: cases.len(),
        top1,
        top3,
        failures,
        mean_correct_score,
    }
}

/// Prints a human-readable report.
pub fn print_report(report: &Report, show_failures: bool) {
    println!(
        "{:<10} top-1 {:>5.1}%  ({}/{})   top-3 {:>5.1}%   mean score {:.3}",
        report.backend,
        report.top1_accuracy() * 100.0,
        report.top1,
        report.total,
        report.top3_accuracy() * 100.0,
        report.mean_correct_score,
    );

    if show_failures && !report.failures.is_empty() {
        println!("\n  {} failing case(s):", report.failures.len());
        for f in &report.failures {
            let got = f.got.as_deref().unwrap_or("<nothing>");
            println!(
                "    {:?}\n      expected {} (scored {:.3}), got {} ({:.3}){}",
                f.say,
                f.expected,
                f.expected_score,
                got,
                f.got_score,
                f.note
                    .as_deref()
                    .map(|n| format!("   [{n}]"))
                    .unwrap_or_default(),
            );
        }
    }
}

/// Reports top-1 accuracy across a range of thresholds.
///
/// Useful for picking a default: too low and unrelated speech triggers
/// commands, too high and legitimate phrasings are rejected.
pub fn sweep(matcher: &dyn Matcher, cases: &[Case]) -> BTreeMap<String, f32> {
    let mut out = BTreeMap::new();
    for step in 0..=10 {
        let threshold = step as f32 / 10.0;
        let report = run(matcher, cases, threshold);
        out.insert(format!("{threshold:.1}"), report.top1_accuracy());
    }
    out
}

pub fn print_sweep(sweep: &BTreeMap<String, f32>) {
    println!("\nthreshold sweep (top-1 accuracy):");
    for (threshold, accuracy) in sweep {
        let bar = "#".repeat((accuracy * 40.0).round() as usize);
        println!("  {threshold}  {:>5.1}%  {bar}", accuracy * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxide_intent::Match;

    /// Matcher that returns a fixed ranking, so the scoring logic can be
    /// tested without a model or a pack directory.
    struct Fake(Vec<(&'static str, f32)>);

    impl Matcher for Fake {
        fn backend(&self) -> &'static str {
            "fake"
        }
        fn rank(&self, _text: &str, limit: usize) -> Vec<Match> {
            self.0
                .iter()
                .take(limit)
                .map(|(id, score)| Match {
                    id: (*id).to_owned(),
                    score: *score,
                    via: None,
                })
                .collect()
        }
    }

    fn case(say: &str, expect: &str) -> Case {
        Case {
            say: say.into(),
            expect: expect.into(),
            note: None,
        }
    }

    #[test]
    fn counts_a_correct_top1() {
        let m = Fake(vec![("a", 0.9), ("b", 0.4)]);
        let r = run(&m, &[case("x", "a")], 0.6);
        assert_eq!(r.top1, 1);
        assert_eq!(r.top3, 1);
        assert!(r.failures.is_empty());
        assert!((r.top1_accuracy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_correct_answer_below_threshold_is_a_failure() {
        let m = Fake(vec![("a", 0.3)]);
        let r = run(&m, &[case("x", "a")], 0.6);
        assert_eq!(r.top1, 0);
        assert_eq!(r.failures.len(), 1);
        assert_eq!(r.failures[0].expected_score, 0.3);
    }

    #[test]
    fn counts_top3_when_top1_is_wrong() {
        let m = Fake(vec![("b", 0.9), ("c", 0.8), ("a", 0.7)]);
        let r = run(&m, &[case("x", "a")], 0.6);
        assert_eq!(r.top1, 0);
        assert_eq!(r.top3, 1);
        assert_eq!(r.failures[0].got.as_deref(), Some("b"));
    }

    #[test]
    fn records_the_expected_score_even_when_it_loses() {
        let m = Fake(vec![("b", 0.9), ("a", 0.7)]);
        let r = run(&m, &[case("x", "a")], 0.6);
        assert_eq!(r.failures[0].expected_score, 0.7);
        assert_eq!(r.failures[0].got_score, 0.9);
    }

    #[test]
    fn empty_ranking_is_a_failure_with_no_winner() {
        let m = Fake(vec![]);
        let r = run(&m, &[case("x", "a")], 0.6);
        assert_eq!(r.failures.len(), 1);
        assert!(r.failures[0].got.is_none());
    }

    #[test]
    fn empty_corpus_reports_zero_not_nan() {
        let m = Fake(vec![("a", 1.0)]);
        let r = run(&m, &[], 0.6);
        assert_eq!(r.top1_accuracy(), 0.0);
        assert_eq!(r.mean_correct_score, 0.0);
    }

    #[test]
    fn sweep_covers_the_whole_range_and_decreases() {
        let m = Fake(vec![("a", 0.55)]);
        let s = sweep(&m, &[case("x", "a")]);
        assert_eq!(s.len(), 11);
        assert_eq!(s["0.5"], 1.0);
        assert_eq!(s["0.6"], 0.0);
    }
}
