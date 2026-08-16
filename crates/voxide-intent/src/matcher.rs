use std::borrow::Cow;

/// A candidate command with the confidence that it is what the user meant.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub id: String,
    /// Backend-specific similarity, normalised to roughly `0.0..=1.0`.
    pub score: f32,
    /// The training phrase responsible for the score, when the backend tracks
    /// one. Shown by `voxide why` so a user can see *why* a phrase matched.
    pub via: Option<String>,
}

/// Ranks known commands against an utterance.
///
/// Implementations return matches sorted by descending score. Returning a
/// ranked list rather than a single winner is what makes `voxide why` and the
/// threshold sweep in `voxide eval` possible.
pub trait Matcher: Send + Sync {
    /// Ranked candidates, best first. May be empty.
    fn rank(&self, text: &str, limit: usize) -> Vec<Match>;

    /// Human-readable backend name, used in eval reports.
    fn backend(&self) -> &'static str;

    /// Score below which this backend's matches are treated as "no match".
    ///
    /// Per-backend because the scales are not comparable: lexical overlap is
    /// near-binary (a correct match scores ~0.99), while cosine similarity is
    /// continuous and a genuine paraphrase lands in the middle of the range.
    /// One shared number silently rejects most of what the semantic backend
    /// gets right, so each implementation carries its own.
    fn default_threshold(&self) -> f32 {
        crate::DEFAULT_THRESHOLD
    }

    /// Best candidate at or above `threshold`.
    fn best(&self, text: &str, threshold: f32) -> Option<Match> {
        self.rank(text, 1)
            .into_iter()
            .find(|m| m.score >= threshold)
    }
}

/// Lowercases, strips punctuation, and collapses whitespace.
///
/// Speech recognisers emit inconsistent casing and stray punctuation, and no
/// matching backend should have to care.
pub fn normalize(text: &str) -> Cow<'_, str> {
    let needs_work = text
        .chars()
        .any(|c| c.is_uppercase() || (!c.is_alphanumeric() && !c.is_whitespace()))
        || text.split_whitespace().count() != text.split(' ').count()
        || text.starts_with(' ')
        || text.ends_with(' ');

    if !needs_work {
        return Cow::Borrowed(text);
    }

    let mut out = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            pending_space = false;
            out.extend(ch.to_lowercase());
        } else {
            // Punctuation and whitespace alike act as a single separator, so
            // "commit, please" and "commit please" normalise identically.
            pending_space = true;
        }
    }
    Cow::Owned(out)
}

/// Splits normalised text into word tokens.
pub fn tokenize(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_lowercases_and_strips_punctuation() {
        assert_eq!(normalize("Run the Tests!"), "run the tests");
        assert_eq!(normalize("commit, please."), "commit please");
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize("run   the\ttests"), "run the tests");
        assert_eq!(normalize("  padded  "), "padded");
    }

    #[test]
    fn normalize_borrows_when_already_clean() {
        assert!(matches!(
            normalize("run the tests"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn normalize_keeps_non_ascii_letters() {
        assert_eq!(normalize("Привет, мир"), "привет мир");
    }

    #[test]
    fn tokenize_splits_on_spaces() {
        assert_eq!(tokenize("run the tests"), vec!["run", "the", "tests"]);
        assert!(tokenize("").is_empty());
    }
}
