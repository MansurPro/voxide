use crate::matcher::{Match, Matcher, normalize, tokenize};
use std::collections::HashSet;
use voxide_core::{CommandSet, template};

/// Token- and character-overlap matching. No model required.
///
/// This is the baseline. It is roughly as capable as the grammar-and-fuzzy
/// approach conventional voice tools use, and it exists so the semantic
/// backend has something honest to be compared against in `voxide eval`.
///
/// Its ceiling is structural: it can only match words the author already
/// wrote down. "check if it builds" scores near zero against the phrase
/// "compile the project" because they share no tokens and few trigrams, even
/// though they mean the same thing.
pub struct LexicalMatcher {
    entries: Vec<Entry>,
}

struct Entry {
    id: String,
    phrase: String,
    tokens: HashSet<String>,
    trigrams: HashSet<[char; 3]>,
    /// Phrases with `{slot}` markers are scored differently: the words the
    /// speaker supplies for the slot are unknown at index time, so overlap
    /// metrics that punish unmatched query tokens are the wrong tool.
    has_slots: bool,
}

/// Ceiling for a slot-bearing phrase.
///
/// Held just below 1.0 so a command whose full phrase was spoken verbatim
/// always outranks a template that merely had its fixed words covered.
const SLOT_PHRASE_CEILING: f32 = 0.98;

impl LexicalMatcher {
    /// Indexes every phrase of every command for `lang`.
    pub fn new(commands: &CommandSet, lang: &str) -> Self {
        let mut entries = Vec::new();

        for lc in commands.commands() {
            for phrase in lc.command.phrases.for_lang(lang) {
                let has_slots = template::has_slot_markers(phrase);

                // Index only the words the speaker will actually say. For a
                // slot phrase that is the fixed text around the marker: the
                // marker's *name* is documentation, not something anyone utters.
                let indexable = template::literal_text(phrase);
                let norm = normalize(&indexable).into_owned();
                if norm.is_empty() {
                    continue;
                }

                entries.push(Entry {
                    id: lc.command.id.clone(),
                    phrase: phrase.to_owned(),
                    tokens: tokenize(&norm).into_iter().map(str::to_owned).collect(),
                    trigrams: trigrams(&norm),
                    has_slots,
                });
            }
        }

        Self { entries }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Matcher for LexicalMatcher {
    fn backend(&self) -> &'static str {
        "lexical"
    }

    fn rank(&self, text: &str, limit: usize) -> Vec<Match> {
        let norm = normalize(text);
        if norm.is_empty() {
            return Vec::new();
        }

        let query_tokens: HashSet<String> =
            tokenize(&norm).into_iter().map(str::to_owned).collect();
        let query_trigrams = trigrams(&norm);

        // Score every phrase, then keep the best phrase per command id.
        let mut best: Vec<Match> = Vec::new();
        for entry in &self.entries {
            let score = if entry.has_slots {
                // Only the fixed words are known in advance, so measure how
                // many of them the utterance covers. Jaccard would be wrong
                // here: it penalises the extra tokens that *are* the argument.
                SLOT_PHRASE_CEILING * recall(&query_tokens, &entry.tokens)
            } else {
                let token_score = jaccard(&query_tokens, &entry.tokens);
                let trigram_score = dice(&query_trigrams, &entry.trigrams);
                0.5 * token_score + 0.5 * trigram_score
            };

            if score <= 0.0 {
                continue;
            }

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
                // Ties broken by id so output is deterministic across runs.
                .then_with(|| a.id.cmp(&b.id))
        });
        best.truncate(limit);
        best
    }
}

/// Fraction of the phrase's fixed words that the query contains.
fn recall(query: &HashSet<String>, required: &HashSet<String>) -> f32 {
    if required.is_empty() {
        return 0.0;
    }
    required.intersection(query).count() as f32 / required.len() as f32
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn dice(a: &HashSet<[char; 3]>, b: &HashSet<[char; 3]>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    2.0 * inter / (a.len() + b.len()) as f32
}

/// Character trigrams over the normalised string, padded at both ends so short
/// words still produce features.
fn trigrams(s: &str) -> HashSet<[char; 3]> {
    let chars: Vec<char> = std::iter::once(' ')
        .chain(s.chars())
        .chain(std::iter::once(' '))
        .collect();
    chars.windows(3).map(|w| [w[0], w[1], w[2]]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxide_core::pack::{Action, Command, LoadedCommand, Phrases};

    fn cmd(id: &str, phrases: &[&str]) -> LoadedCommand {
        LoadedCommand {
            command: Command {
                id: id.to_owned(),
                description: String::new(),
                phrases: Phrases::Flat(phrases.iter().map(|s| (*s).to_owned()).collect()),
                slots: Vec::new(),
                action: Action::Shell {
                    run: "true".to_owned(),
                    cwd: None,
                    timeout_ms: 1000,
                },
                chain: false,
            },
            pack_name: "test".to_owned(),
            pack_dir: std::path::PathBuf::from("."),
        }
    }

    fn matcher() -> LexicalMatcher {
        let set = CommandSet::from_commands(vec![
            cmd("cargo.test", &["run the tests", "run tests"]),
            cmd("cargo.build", &["build the project", "compile the project"]),
            cmd("git.status", &["what changed", "git status"]),
        ]);
        LexicalMatcher::new(&set, "en")
    }

    #[test]
    fn exact_phrase_scores_near_one() {
        let m = matcher();
        let top = &m.rank("run the tests", 5)[0];
        assert_eq!(top.id, "cargo.test");
        assert!(top.score > 0.95, "score was {}", top.score);
    }

    #[test]
    fn close_wording_still_matches() {
        let m = matcher();
        let top = &m.rank("run tests", 5)[0];
        assert_eq!(top.id, "cargo.test");
    }

    #[test]
    fn results_are_ranked_descending() {
        let m = matcher();
        let ranked = m.rank("build the project", 5);
        for pair in ranked.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn reports_the_phrase_responsible() {
        let m = matcher();
        assert_eq!(
            m.rank("git status", 1)[0].via.as_deref(),
            Some("git status")
        );
    }

    #[test]
    fn limit_is_respected() {
        assert_eq!(matcher().rank("the project", 2).len(), 2);
    }

    #[test]
    fn empty_query_matches_nothing() {
        assert!(matcher().rank("", 5).is_empty());
        assert!(matcher().rank("   ", 5).is_empty());
    }

    #[test]
    fn best_applies_the_threshold() {
        let m = matcher();

        // An exact phrase scores 1.0, so it clears any threshold up to 1.0.
        assert!(m.best("run the tests", 1.0).is_some());

        // A phrase that is not verbatim in the pack scores below 1.0 and is
        // gated accordingly.
        let query = "run the test suite";
        let approx = m.rank(query, 1)[0].score;
        assert!(approx < 1.0, "expected an imperfect score, got {approx}");
        assert!(m.best(query, approx - 0.01).is_some());
        assert!(m.best(query, approx + 0.01).is_none());
    }

    /// The documented ceiling of this backend, pinned as a test so the eval
    /// comparison against the semantic matcher stays meaningful. Paraphrase
    /// with no shared vocabulary is exactly where lexical matching fails.
    #[test]
    fn paraphrase_defeats_lexical_matching() {
        let m = matcher();
        let ranked = m.rank("check if it compiles", 3);
        let build = ranked.iter().find(|r| r.id == "cargo.build");
        assert!(
            build.is_none_or(|b| b.score < 0.5),
            "lexical unexpectedly handled a paraphrase: {ranked:?}"
        );
    }

    /// A slot phrase must match on the words the speaker actually says. The
    /// marker name is documentation: nobody saying "checkout main" utters the
    /// word "branch", so indexing it as a required token makes the command
    /// unreachable.
    #[test]
    fn slot_phrase_matches_an_arbitrary_argument() {
        let set = CommandSet::from_commands(vec![
            cmd("git.checkout", &["checkout {branch}", "switch to {branch}"]),
            cmd("cargo.check", &["check the code"]),
        ]);
        let m = LexicalMatcher::new(&set, "en");

        for utterance in ["checkout main", "checkout feature/login", "switch to dev"] {
            let ranked = m.rank(utterance, 3);
            assert_eq!(
                ranked[0].id, "git.checkout",
                "for {utterance:?}: {ranked:?}"
            );
            assert!(
                ranked[0].score > 0.9,
                "for {utterance:?} score was {}",
                ranked[0].score
            );
        }
    }

    #[test]
    fn slot_phrase_does_not_match_unrelated_words() {
        let set = CommandSet::from_commands(vec![cmd("git.checkout", &["checkout {branch}"])]);
        let m = LexicalMatcher::new(&set, "en");
        assert!(m.best("format the code", 0.62).is_none());
    }

    #[test]
    fn verbatim_phrase_outranks_a_slot_template() {
        // "check the code" is spoken exactly; the template only had its fixed
        // words covered, so it must not win.
        let set = CommandSet::from_commands(vec![
            cmd("cargo.check", &["check the code"]),
            cmd("git.checkout", &["check {branch}"]),
        ]);
        let m = LexicalMatcher::new(&set, "en");
        assert_eq!(m.rank("check the code", 2)[0].id, "cargo.check");
    }

    #[test]
    fn empty_command_set_ranks_nothing() {
        let m = LexicalMatcher::new(&CommandSet::default(), "en");
        assert!(m.is_empty());
        assert!(m.rank("anything", 5).is_empty());
    }
}
