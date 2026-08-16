use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use voxide_core::CommandSet;

/// Resolves the directory holding command packs.
///
/// Search order, first hit wins:
///   1. `--packs <dir>`
///   2. `$VOXIDE_PACKS`
///   3. `./packs` relative to the current directory (project-local packs)
///   4. the per-user config directory
pub fn resolve_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(dir) = explicit {
        return dir.to_path_buf();
    }
    if let Some(env) = std::env::var_os("VOXIDE_PACKS") {
        return PathBuf::from(env);
    }
    let local = PathBuf::from("packs");
    if local.is_dir() {
        return local;
    }
    user_dir()
}

/// Per-user pack directory, following platform conventions.
pub fn user_dir() -> PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);

    #[cfg(not(windows))]
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")));

    base.unwrap_or_else(|| PathBuf::from("."))
        .join("voxide")
        .join("packs")
}

/// Loads every pack under `dir`, with a diagnostic naming the directory.
pub fn load(dir: &Path) -> Result<CommandSet> {
    let set = CommandSet::load_dir(dir)
        .with_context(|| format!("loading command packs from {}", dir.display()))?;

    if set.is_empty() {
        tracing::warn!(
            dir = %dir.display(),
            "no command packs found; `voxide say` will match nothing"
        );
    }
    Ok(set)
}
