//! Intent matching: turning a recognised phrase into a command id.
//!
//! Two implementations of [`Matcher`] ship here:
//!
//! - [`LexicalMatcher`] — token and character overlap. No model, no native
//!   dependencies, always available. This is the baseline that conventional
//!   grammar-driven voice tools are effectively limited to.
//! - [`SemanticMatcher`] — sentence-embedding cosine similarity, behind the
//!   `embed` feature. This is what lets a user say "let's see if it compiles"
//!   and hit `cargo.check` without ever having been told that phrase exists.
//!
//! Keeping both behind one trait is deliberate: `voxide eval` scores them on
//! the same corpus, so the benefit of the semantic path is a measured number
//! rather than a claim.

pub mod lexical;
pub mod matcher;
pub mod slots;

#[cfg(feature = "embed")]
pub mod semantic;

pub use lexical::LexicalMatcher;
pub use matcher::{Match, Matcher};
pub use slots::extract as extract_slots;

#[cfg(feature = "embed")]
pub use semantic::SemanticMatcher;

/// Default score below which a match is treated as "no match".
///
/// Tuned per backend by `voxide eval --sweep`; this is only the starting point
/// for a fresh install.
pub const DEFAULT_THRESHOLD: f32 = 0.62;
