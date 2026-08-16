use std::path::PathBuf;

/// Errors produced while loading or validating command packs.
///
/// Every variant carries the offending path. Pack loading is the single most
/// common place a user gets something wrong, and a diagnostic that does not
/// name the file is close to useless.
#[derive(Debug, thiserror::Error)]
pub enum PackError {
    #[error("failed to read pack at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse pack at {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: Box<toml::de::Error>,
    },

    #[error("pack at {path} declares no commands")]
    Empty { path: PathBuf },

    #[error("command '{id}' in {path} declares no phrases; it could never be matched")]
    NoPhrases { id: String, path: PathBuf },

    #[error(
        "command '{id}' in {path} references slot '{slot}' in its action template, \
         but no such slot is declared"
    )]
    UnknownSlot {
        id: String,
        slot: String,
        path: PathBuf,
    },

    #[error("duplicate command id '{id}': defined in both {first} and {second}")]
    DuplicateId {
        id: String,
        first: PathBuf,
        second: PathBuf,
    },
}

pub type Result<T> = std::result::Result<T, PackError>;
