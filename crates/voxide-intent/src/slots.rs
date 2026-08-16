//! Slot extraction by aligning an utterance against the phrase that matched.
//!
//! A phrase is authored as `"checkout {branch}"`. Once that phrase wins, the
//! literal segments around the marker pin down where the argument sits, and
//! whatever falls between them is the value.
//!
//! This is model-free and handles the common cases exactly. It is the floor,
//! not the ceiling: a zero-shot NER backend will later cover phrasings whose
//! wording drifts from the template, and both feed the same [`Slots`] map.

use voxide_core::{SlotValue, Slots};

/// One piece of a parsed phrase template.
#[derive(Debug, PartialEq, Eq)]
enum Segment<'a> {
    Literal(&'a str),
    Slot(&'a str),
}

/// Splits `"checkout {branch} now"` into literal and slot segments.
fn segments(phrase: &str) -> Vec<Segment<'_>> {
    let mut out = Vec::new();
    let mut rest = phrase;

    while let Some(open) = rest.find('{') {
        if open > 0 {
            out.push(Segment::Literal(&rest[..open]));
        }
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            out.push(Segment::Literal("{"));
            rest = after;
            break;
        };
        out.push(Segment::Slot(after[..close].trim()));
        rest = &after[close + 1..];
    }

    if !rest.is_empty() {
        out.push(Segment::Literal(rest));
    }
    out
}

/// Compares two char slices ignoring case and treating any run of
/// non-alphanumeric characters as a single separator.
///
/// This lets the literal `"checkout "` line up with `"Checkout, "` as spoken
/// text arrives with inconsistent punctuation from a recogniser.
fn literal_matches_at(hay: &[char], pos: usize, needle: &str) -> Option<usize> {
    let mut i = pos;
    let mut chars = needle.chars().peekable();

    while let Some(nc) = chars.next() {
        if !nc.is_alphanumeric() {
            // Separator run in the needle: consume a separator run in the hay.
            while chars.peek().is_some_and(|c| !c.is_alphanumeric()) {
                chars.next();
            }
            let start = i;
            while i < hay.len() && !hay[i].is_alphanumeric() {
                i += 1;
            }
            // A separator is required unless we are at either boundary.
            if i == start && i != 0 && i < hay.len() {
                return None;
            }
            continue;
        }

        if i >= hay.len() {
            return None;
        }
        if !hay[i].eq_ignore_ascii_case(&nc) && hay[i].to_lowercase().ne(nc.to_lowercase()) {
            return None;
        }
        i += 1;
    }

    Some(i)
}

/// Finds the next position at or after `from` where `needle` matches.
fn find_literal(hay: &[char], from: usize, needle: &str) -> Option<(usize, usize)> {
    (from..=hay.len()).find_map(|start| literal_matches_at(hay, start, needle).map(|e| (start, e)))
}

/// Extracts slot values from `utterance` using the template `phrase`.
///
/// Returns an empty map when the phrase declares no slots, or when the
/// literals around a slot cannot be found in order.
pub fn extract(phrase: &str, utterance: &str) -> Slots {
    let segs = segments(phrase);
    if !segs.iter().any(|s| matches!(s, Segment::Slot(_))) {
        return Slots::new();
    }

    let hay: Vec<char> = utterance.chars().collect();
    let mut slots = Slots::new();
    let mut pos = 0usize;
    let mut pending: Option<&str> = None;

    for seg in &segs {
        match seg {
            Segment::Literal(lit) if lit.trim().is_empty() => {
                // Whitespace-only separator between markers carries no anchor.
            }
            Segment::Literal(lit) => {
                let Some((start, end)) = find_literal(&hay, pos, lit) else {
                    return Slots::new();
                };
                if let Some(name) = pending.take() {
                    let value: String = hay[pos..start].iter().collect();
                    insert_if_present(&mut slots, name, &value);
                }
                pos = end;
            }
            Segment::Slot(name) => {
                if let Some(previous) = pending.replace(name) {
                    // Two adjacent markers with nothing between them cannot be
                    // split; the earlier one gets nothing.
                    let _ = previous;
                }
            }
        }
    }

    if let Some(name) = pending {
        let value: String = hay[pos.min(hay.len())..].iter().collect();
        insert_if_present(&mut slots, name, &value);
    }

    slots
}

fn insert_if_present(slots: &mut Slots, name: &str, raw: &str) {
    let trimmed = raw.trim().trim_matches(|c: char| {
        !c.is_alphanumeric() && c != '/' && c != '.' && c != '-' && c != '_'
    });
    if !trimmed.is_empty() {
        slots.insert(name.to_owned(), SlotValue::parse(trimmed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(slots: &Slots, key: &str) -> Option<String> {
        slots.get(key).map(|v| v.as_str().into_owned())
    }

    #[test]
    fn captures_a_trailing_slot() {
        let s = extract("checkout {branch}", "checkout main");
        assert_eq!(text(&s, "branch").as_deref(), Some("main"));
    }

    #[test]
    fn preserves_original_casing_of_the_value() {
        let s = extract("checkout {branch}", "checkout Feature/Login");
        assert_eq!(text(&s, "branch").as_deref(), Some("Feature/Login"));
    }

    #[test]
    fn captures_a_slot_between_literals() {
        let s = extract("run the {filter} test", "run the parser test");
        assert_eq!(text(&s, "filter").as_deref(), Some("parser"));
    }

    #[test]
    fn captures_two_slots() {
        let s = extract("copy {src} to {dst}", "copy notes.txt to backup");
        assert_eq!(text(&s, "src").as_deref(), Some("notes.txt"));
        assert_eq!(text(&s, "dst").as_deref(), Some("backup"));
    }

    #[test]
    fn tolerates_punctuation_and_case_in_the_utterance() {
        let s = extract("checkout {branch}", "Checkout,  main");
        assert_eq!(text(&s, "branch").as_deref(), Some("main"));
    }

    #[test]
    fn multiword_values_are_captured_whole() {
        let s = extract(
            "commit with message {msg}",
            "commit with message fix the cache hash",
        );
        assert_eq!(text(&s, "msg").as_deref(), Some("fix the cache hash"));
    }

    #[test]
    fn numeric_values_parse_as_numbers() {
        let s = extract("go to line {n}", "go to line 42");
        assert_eq!(s.get("n"), Some(&SlotValue::Number(42.0)));
    }

    #[test]
    fn phrase_without_slots_yields_nothing() {
        assert!(extract("run the tests", "run the tests").is_empty());
    }

    #[test]
    fn missing_value_yields_no_entry() {
        // Nothing follows the anchor, so there is no branch to capture.
        assert!(extract("checkout {branch}", "checkout").is_empty());
    }

    #[test]
    fn unmatched_literal_yields_nothing() {
        assert!(extract("run the {filter} test", "completely different words").is_empty());
    }

    #[test]
    fn segments_split_correctly() {
        assert_eq!(
            segments("checkout {branch} now"),
            vec![
                Segment::Literal("checkout "),
                Segment::Slot("branch"),
                Segment::Literal(" now")
            ]
        );
    }

    #[test]
    fn unterminated_marker_does_not_panic() {
        let _ = extract("checkout {branch", "checkout main");
    }
}
