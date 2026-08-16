use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Declaration of a named argument a command can extract from an utterance.
///
/// `entity` is a free-form natural-language label handed to the extractor
/// (for example `"branch name"`, `"file path"`, `"test name"`). Zero-shot NER
/// models take the label itself as the query, which is why it reads like prose
/// rather than an enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotDef {
    pub name: String,

    /// Natural-language description of what to extract.
    pub entity: String,

    /// When true, the command still matches if the slot cannot be filled.
    #[serde(default)]
    pub optional: bool,

    /// Value used when extraction finds nothing and the slot is optional.
    #[serde(default)]
    pub default: Option<String>,
}

/// A value extracted from an utterance and bound to a slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SlotValue {
    Number(f64),
    Text(String),
}

impl SlotValue {
    /// Parses a captured span, preferring a numeric interpretation.
    pub fn parse(raw: &str) -> Self {
        match raw.trim().parse::<f64>() {
            Ok(n) if n.is_finite() => SlotValue::Number(n),
            _ => SlotValue::Text(raw.trim().to_owned()),
        }
    }

    /// Renders the value for substitution into an action template.
    pub fn as_str(&self) -> std::borrow::Cow<'_, str> {
        match self {
            SlotValue::Text(s) => std::borrow::Cow::Borrowed(s),
            // Integral floats render without a trailing ".0"; `cargo test 3.0`
            // is not what anyone meant when they said "three".
            SlotValue::Number(n) if n.fract() == 0.0 && n.abs() < 1e15 => {
                std::borrow::Cow::Owned((*n as i64).to_string())
            }
            SlotValue::Number(n) => std::borrow::Cow::Owned(n.to_string()),
        }
    }
}

impl std::fmt::Display for SlotValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_str())
    }
}

/// Slot name to extracted value. Ordered so rendering and logs are stable.
pub type Slots = BTreeMap<String, SlotValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbers_but_keeps_words() {
        assert_eq!(SlotValue::parse("42"), SlotValue::Number(42.0));
        assert_eq!(SlotValue::parse("main"), SlotValue::Text("main".to_owned()));
    }

    #[test]
    fn integral_numbers_render_without_decimal_point() {
        assert_eq!(SlotValue::Number(3.0).as_str(), "3");
        assert_eq!(SlotValue::Number(2.5).as_str(), "2.5");
    }

    #[test]
    fn parse_trims_surrounding_whitespace() {
        assert_eq!(
            SlotValue::parse("  feature/login  "),
            SlotValue::Text("feature/login".to_owned())
        );
    }

    #[test]
    fn non_finite_input_stays_text() {
        // "inf" and "NaN" parse as f64 but are useless as command arguments.
        assert_eq!(SlotValue::parse("inf"), SlotValue::Text("inf".to_owned()));
        assert_eq!(SlotValue::parse("NaN"), SlotValue::Text("NaN".to_owned()));
    }
}
