//! Core types shared across voxide: command packs, slots, and action specs.
//!
//! This crate is deliberately dependency-light and contains no I/O beyond
//! reading pack manifests, so every other crate can depend on it without
//! pulling in an audio or ML backend.

pub mod error;
pub mod pack;
pub mod slot;
pub mod template;

pub use error::{PackError, Result};
pub use pack::{Action, Command, CommandSet, LoadedCommand, PackMeta, Phrases, Sandbox};
pub use slot::{SlotDef, SlotValue, Slots};

/// Language used when a caller does not specify one.
pub const DEFAULT_LANG: &str = "en";
