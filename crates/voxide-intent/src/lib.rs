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

/// Default score below which a lexical match is treated as "no match".
///
/// Lexical overlap is near-binary: on the golden corpus a correct match scores
/// ~0.99, so anything in the middle of the range is noise.
///
/// This is *not* a usable floor for the semantic backend, whose scores live on
/// an entirely different scale -- see [`Matcher::default_threshold`].
pub const DEFAULT_THRESHOLD: f32 = 0.62;
