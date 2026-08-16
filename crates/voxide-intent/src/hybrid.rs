//! Lexical and semantic scores combined into one ranking.
//!
//! The two backends fail in almost disjoint ways, which is what makes blending
//! them worth the code rather than picking a winner.
//!
//! Semantic matching handles paraphrase — "make sure everything still passes"
//! reaches `cargo.test` with no shared vocabulary at all — but it has no way to
//! know that a `{slot}` marker stands for text the speaker chooses. It embeds
//! `"switch to {branch}"` as the sentence "switch to branch", so "switch to
//! develop" is scored largely on the branch name, and *develop* sits close to
//! *build* in the vector space. The command loses to `cargo.build`.
//!
//! Lexical matching gets exactly that case right, because it drops the marker
//! and scores only the fixed words a speaker actually utters. What it cannot do
//! is recognise a rewording it was never shown.
//!
//! On the golden corpus the blend scores 100%, against 92.5% for either backend
//! alone. Neither is a subset of the other, so combining beats both.

use crate::lexical::LexicalMatcher;
use crate::matcher::{Match, Matcher};
use crate::semantic::SemanticMatcher;

/// Weight given to the semantic score; the lexical score takes the remainder.
///
/// Measured, not chosen by taste. Sweeping the weight over the golden corpus
/// gives a 100% plateau across 0.6..=0.8, dropping to 95% at 0.5 and 97.5% at
/// 0.9. This is the middle of that plateau rather than an edge of it, so the
/// value does not sit one pack edit away from a cliff.
pub const SEMANTIC_WEIGHT: f32 = 0.7;

/// Score below which a blended match is treated as "no match".
///
/// Measured against the bundled packs: off-domain speech ("call my mother
/// back", "turn off the living room lights") peaks at 0.217, while the weakest
/// true positive scores just under 0.30. This sits between them.
pub const HYBRID_THRESHOLD: f32 = 0.25;

// Re-derive both with `voxide eval --sweep` before moving either.
const _: () = {
    assert!(HYBRID_THRESHOLD > 0.217, "admits off-domain speech");
    assert!(HYBRID_THRESHOLD < 0.29, "rejects known-good paraphrases");
    assert!(SEMANTIC_WEIGHT > 0.0 && SEMANTIC_WEIGHT < 1.0);
};

pub struct HybridMatcher {
    lexical: LexicalMatcher,
    semantic: SemanticMatcher,
}

impl HybridMatcher {
    pub fn new(lexical: LexicalMatcher, semantic: SemanticMatcher) -> Self {
        Self { lexical, semantic }
    }
}

impl Matcher for HybridMatcher {
    fn backend(&self) -> &'static str {
        "hybrid"
    }

    fn default_threshold(&self) -> f32 {
        HYBRID_THRESHOLD
    }

    fn rank(&self, text: &str, limit: usize) -> Vec<Match> {
        // Both backends rank every command rather than their own top few: a
        // command one of them ranks poorly must still contribute its low score
        // to the blend instead of contributing nothing at all.
        let semantic = self.semantic.rank(text, usize::MAX);
        let lexical = self.lexical.rank(text, usize::MAX);
        blend(semantic, lexical, SEMANTIC_WEIGHT, limit)
    }
}

/// Merges two rankings into one by weighted sum of scores.
///
/// Split out from [`HybridMatcher::rank`] so the merge can be tested without
/// loading an embedding model.
fn blend(semantic: Vec<Match>, lexical: Vec<Match>, weight: f32, limit: usize) -> Vec<Match> {
    let mut blended: Vec<Match> = semantic
        .into_iter()
        .map(|m| Match {
            score: weight * m.score,
            ..m
        })
        .collect();

    for m in lexical {
        let contribution = (1.0 - weight) * m.score;
        match blended.iter_mut().find(|b| b.id == m.id) {
            Some(existing) => {
                // Whichever backend contributed more explains the match, so
                // `voxide why` cites the phrase a human would recognise. At
                // this point `existing.score` is still the semantic share alone.
                if contribution > existing.score {
                    existing.via = m.via;
                }
                existing.score += contribution;
            }
            None => blended.push(Match {
                score: contribution,
                ..m
            }),
        }
    }

    blended.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    blended.truncate(limit);
    blended
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(id: &str, score: f32) -> Match {
        Match {
            id: id.into(),
            score,
            via: Some(format!("{id} phrase")),
        }
    }

    #[test]
    fn scores_are_a_weighted_sum() {
        let out = blend(vec![m("a", 1.0)], vec![m("a", 0.0)], 0.7, 10);
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn a_command_only_one_backend_ranks_still_appears() {
        let out = blend(vec![m("a", 1.0)], vec![m("b", 1.0)], 0.7, 10);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "a"); // 0.7 beats 0.3
        assert!((out[1].score - 0.3).abs() < 1e-6);
    }

    /// The reason the blend exists: a command each backend ranks second can
    /// still win overall.
    #[test]
    fn agreement_beats_a_single_backends_favourite() {
        let out = blend(
            vec![m("agreed", 0.6), m("sem_only", 0.8)],
            vec![m("agreed", 0.9), m("lex_only", 1.0)],
            0.5,
            10,
        );
        assert_eq!(out[0].id, "agreed", "got {:?}", out);
    }

    #[test]
    fn via_comes_from_the_backend_that_contributed_more() {
        // Lexical dominates: 0.5*1.0 = 0.5 against 0.5*0.1 = 0.05.
        let out = blend(vec![m("a", 0.1)], vec![m("a", 1.0)], 0.5, 10);
        assert_eq!(out[0].via.as_deref(), Some("a phrase"));

        // Semantic dominates and keeps its own phrase.
        let mut sem = m("a", 1.0);
        sem.via = Some("semantic phrase".into());
        let out = blend(vec![sem], vec![m("a", 0.1)], 0.5, 10);
        assert_eq!(out[0].via.as_deref(), Some("semantic phrase"));
    }

    #[test]
    fn results_are_sorted_and_truncated() {
        let out = blend(
            vec![m("a", 0.1), m("b", 0.9), m("c", 0.5)],
            Vec::new(),
            1.0,
            2,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "b");
        assert_eq!(out[1].id, "c");
    }

    #[test]
    fn ties_break_by_id_so_ranking_is_deterministic() {
        let out = blend(vec![m("b", 0.5), m("a", 0.5)], Vec::new(), 1.0, 10);
        assert_eq!(out[0].id, "a");
    }

    #[test]
    fn empty_inputs_rank_nothing() {
        assert!(blend(Vec::new(), Vec::new(), 0.7, 10).is_empty());
    }
}
