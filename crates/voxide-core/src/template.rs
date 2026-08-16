use crate::slot::Slots;

/// Extracts the `{{slot}}` placeholder names referenced by a template.
///
/// Used at load time to reject a pack whose action references a slot it never
/// declares, rather than silently substituting an empty string at run time.
pub fn referenced_slots(template: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = template;

    while let Some(open) = rest.find("{{") {
        let after = &rest[open + 2..];
        let Some(close) = after.find("}}") else { break };
        let name = after[..close].trim();
        if !name.is_empty() && !out.iter().any(|n| n == name) {
            out.push(name.to_owned());
        }
        rest = &after[close + 2..];
    }

    out
}

/// Substitutes `{{slot}}` placeholders with their extracted values.
///
/// Unfilled placeholders collapse to an empty string and the surrounding
/// whitespace is normalised, so an optional slot in `cargo test {{filter}}`
/// yields `cargo test` rather than `cargo test ` with a dangling argument.
pub fn render(template: &str, slots: &Slots) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;

    while let Some(open) = rest.find("{{") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 2..];

        let Some(close) = after.find("}}") else {
            // Unterminated placeholder: emit the literal braces and stop.
            out.push_str("{{");
            rest = after;
            break;
        };

        let name = after[..close].trim();
        if let Some(value) = slots.get(name) {
            out.push_str(&value.as_str());
        }
        rest = &after[close + 2..];
    }

    out.push_str(rest);
    collapse_whitespace(&out)
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    let mut wrote_any = false;

    for ch in s.chars() {
        if ch.is_whitespace() {
            pending_space = wrote_any;
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(ch);
            wrote_any = true;
        }
    }

    out
}

/// True when a phrase contains at least one `{slot}` marker.
pub fn has_slot_markers(phrase: &str) -> bool {
    phrase
        .find('{')
        .is_some_and(|open| phrase[open + 1..].contains('}'))
}

/// The literal text of a phrase, with `{slot}` markers removed entirely.
///
/// Distinct from [`strip_slot_markers`], which keeps the marker's name as a
/// word. For matching, the marker stands for text the speaker supplies, so the
/// name itself must not be treated as a word they are expected to say:
/// `"checkout {branch}"` yields `"checkout"`, because someone saying
/// "checkout main" never utters the word "branch".
pub fn literal_text(phrase: &str) -> String {
    let mut out = String::with_capacity(phrase.len());
    let mut rest = phrase;

    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find('}') else {
            out.push('{');
            rest = after;
            break;
        };
        out.push(' ');
        rest = &after[close + 1..];
    }

    out.push_str(rest);
    collapse_whitespace(&out)
}

/// Strips `{slot}` markers from a training phrase before embedding it.
///
/// Phrases are authored as `"run the {filter} test"` so a human can see where
/// the argument goes. The braces are noise to a sentence encoder, so they are
/// removed while the word itself is kept: `"run the filter test"`.
pub fn strip_slot_markers(phrase: &str) -> String {
    let mut out = String::with_capacity(phrase.len());
    let mut rest = phrase;

    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];

        let Some(close) = after.find('}') else {
            out.push('{');
            rest = after;
            break;
        };

        out.push_str(after[..close].trim());
        rest = &after[close + 1..];
    }

    out.push_str(rest);
    collapse_whitespace(&out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot::SlotValue;

    fn slots(pairs: &[(&str, &str)]) -> Slots {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), SlotValue::parse(v)))
            .collect()
    }

    #[test]
    fn substitutes_a_filled_slot() {
        let s = slots(&[("branch", "main")]);
        assert_eq!(render("git checkout {{branch}}", &s), "git checkout main");
    }

    #[test]
    fn unfilled_slot_leaves_no_trailing_space() {
        assert_eq!(render("cargo test {{filter}}", &Slots::new()), "cargo test");
    }

    #[test]
    fn handles_several_placeholders() {
        let s = slots(&[("a", "one"), ("b", "two")]);
        assert_eq!(render("{{a}} and {{b}}", &s), "one and two");
    }

    #[test]
    fn unterminated_placeholder_is_left_literal() {
        assert_eq!(render("echo {{oops", &Slots::new()), "echo {{oops");
    }

    #[test]
    fn finds_referenced_slot_names() {
        assert_eq!(
            referenced_slots("git commit -m {{msg}} on {{branch}}"),
            vec!["msg".to_owned(), "branch".to_owned()]
        );
    }

    #[test]
    fn referenced_slots_deduplicates() {
        assert_eq!(referenced_slots("{{x}} {{x}}"), vec!["x".to_owned()]);
    }

    #[test]
    fn strips_markers_for_embedding() {
        assert_eq!(
            strip_slot_markers("run the {filter} test"),
            "run the filter test"
        );
        assert_eq!(strip_slot_markers("checkout {branch}"), "checkout branch");
    }

    #[test]
    fn strip_markers_is_a_noop_without_braces() {
        assert_eq!(strip_slot_markers("run the tests"), "run the tests");
    }

    #[test]
    fn literal_text_drops_the_marker_entirely() {
        assert_eq!(literal_text("checkout {branch}"), "checkout");
        assert_eq!(literal_text("run the {filter} tests"), "run the tests");
        assert_eq!(literal_text("copy {a} to {b}"), "copy to");
    }

    #[test]
    fn literal_text_is_a_noop_without_markers() {
        assert_eq!(literal_text("run the tests"), "run the tests");
    }

    #[test]
    fn detects_slot_markers() {
        assert!(has_slot_markers("checkout {branch}"));
        assert!(!has_slot_markers("run the tests"));
        assert!(!has_slot_markers("unterminated {oops"));
    }
}
