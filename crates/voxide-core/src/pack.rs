use crate::error::{PackError, Result};
use crate::slot::SlotDef;
use crate::template;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Training phrases for a command.
///
/// Authors may write either a flat list (single language) or a map keyed by
/// language code. Accepting both is deliberate: requiring the map form for
/// every pack is the kind of papercut that makes a file parse as valid TOML,
/// fail deserialization, and get skipped with nothing but a log line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Phrases {
    Flat(Vec<String>),
    ByLang(BTreeMap<String, Vec<String>>),
}

impl Phrases {
    /// Phrases for a language, falling back to English, then to any language.
    pub fn for_lang(&self, lang: &str) -> Vec<&str> {
        match self {
            Phrases::Flat(v) => v.iter().map(String::as_str).collect(),
            Phrases::ByLang(m) => m
                .get(lang)
                .or_else(|| m.get("en"))
                .or_else(|| m.values().next())
                .map(|v| v.iter().map(String::as_str).collect())
                .unwrap_or_default(),
        }
    }

    /// Every phrase across every language, used for fingerprinting.
    pub fn all(&self) -> Vec<&str> {
        match self {
            Phrases::Flat(v) => v.iter().map(String::as_str).collect(),
            Phrases::ByLang(m) => m.values().flatten().map(String::as_str).collect(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Phrases::Flat(v) => v.is_empty(),
            Phrases::ByLang(m) => m.values().all(Vec::is_empty),
        }
    }
}

/// Capability tier granted to a Lua action.
///
/// The tier is declared by the pack author, so this is a capability model, not
/// a containment boundary: it limits what an honest pack can reach by accident,
/// not what a hostile pack can do on purpose. The real trust decision is
/// installing the pack at all, which is why `voxide packs add` reports the
/// highest tier a pack requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sandbox {
    /// Pure computation. No filesystem, network, environment, or process access.
    Minimal,
    /// Adds scoped filesystem access and per-command persistent state.
    #[default]
    Standard,
    /// Adds network, environment, clipboard, and subprocess execution.
    Full,
}

/// Wall-clock budget for any action, in milliseconds.
///
/// This is enforced by a real deadline on a worker thread, not by a bytecode
/// hook, so it also bounds blocking calls such as sleeps and HTTP requests.
fn default_timeout_ms() -> u64 {
    10_000
}

/// What a command does once it has been matched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Run a shell command. `run` may contain `{{slot}}` placeholders.
    Shell {
        run: String,
        #[serde(default)]
        cwd: Option<String>,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
    },
    /// Execute a Lua script stored alongside the pack manifest.
    Lua {
        script: String,
        #[serde(default)]
        sandbox: Sandbox,
        #[serde(default = "default_timeout_ms")]
        timeout_ms: u64,
    },
    /// Type a literal string, with `{{slot}}` substitution.
    Text { text: String },
    /// Send a key chord, for example `"ctrl+shift+p"`.
    Keys { keys: String },
}

impl Action {
    /// The template text this action will interpolate, if any.
    pub fn template(&self) -> Option<&str> {
        match self {
            Action::Shell { run, .. } => Some(run),
            Action::Text { text } => Some(text),
            Action::Lua { .. } | Action::Keys { .. } => None,
        }
    }

    pub fn timeout_ms(&self) -> u64 {
        match self {
            Action::Shell { timeout_ms, .. } | Action::Lua { timeout_ms, .. } => *timeout_ms,
            Action::Text { .. } | Action::Keys { .. } => default_timeout_ms(),
        }
    }
}

/// A single voice-addressable command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Command {
    pub id: String,

    #[serde(default)]
    pub description: String,

    pub phrases: Phrases,

    #[serde(default)]
    pub slots: Vec<SlotDef>,

    pub action: Action,

    /// Keep the session listening after this command runs, so a follow-up
    /// needs no wake word.
    #[serde(default)]
    pub chain: bool,
}

/// Pack-level metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackMeta {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

/// The on-disk shape of a `*.toml` pack manifest.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct PackFile {
    pack: PackMeta,
    #[serde(default, rename = "command")]
    command: Vec<Command>,
}

/// A command paired with the pack it came from.
#[derive(Debug, Clone)]
pub struct LoadedCommand {
    pub command: Command,
    pub pack_name: String,
    /// Directory holding the manifest; Lua script paths resolve against it.
    pub pack_dir: PathBuf,
}

/// Every command loaded from a pack directory.
#[derive(Debug, Clone, Default)]
pub struct CommandSet {
    commands: Vec<LoadedCommand>,
}

impl CommandSet {
    pub fn commands(&self) -> &[LoadedCommand] {
        &self.commands
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&LoadedCommand> {
        self.commands.iter().find(|c| c.command.id == id)
    }

    /// Stable content hash over ids and phrases.
    ///
    /// The embedding cache is keyed on this, so editing a phrase or adding a
    /// command invalidates the cached vectors while an unrelated edit (a
    /// description, a shell flag) does not force a re-embed.
    pub fn fingerprint(&self, lang: &str) -> String {
        let mut entries: Vec<String> = self
            .commands
            .iter()
            .map(|lc| {
                let mut phrases: Vec<&str> = lc.command.phrases.for_lang(lang);
                phrases.sort_unstable();
                format!("{}\u{1}{}", lc.command.id, phrases.join("\u{2}"))
            })
            .collect();
        entries.sort_unstable();

        let mut hasher = Sha256::new();
        hasher.update(lang.as_bytes());
        for e in &entries {
            hasher.update(b"\x00");
            hasher.update(e.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Loads every `*.toml` manifest under `dir`, recursively.
    ///
    /// A malformed pack is a hard error. Skipping it with a warning is how a
    /// user ends up debugging why a command they can plainly see in a file
    /// never fires.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let mut commands: Vec<LoadedCommand> = Vec::new();
        let mut seen: BTreeMap<String, PathBuf> = BTreeMap::new();

        let mut manifests: Vec<PathBuf> = walkdir::WalkDir::new(dir)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(walkdir::DirEntry::into_path)
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .collect();
        // Deterministic order so duplicate-id errors name the same pair every run.
        manifests.sort();

        for path in manifests {
            let loaded = load_pack_file(&path)?;
            for lc in loaded {
                if let Some(first) = seen.get(&lc.command.id) {
                    return Err(PackError::DuplicateId {
                        id: lc.command.id.clone(),
                        first: first.clone(),
                        second: path.clone(),
                    });
                }
                seen.insert(lc.command.id.clone(), path.clone());
                commands.push(lc);
            }
        }

        Ok(Self { commands })
    }

    /// Builds a set directly from in-memory commands. Intended for tests.
    pub fn from_commands(commands: Vec<LoadedCommand>) -> Self {
        Self { commands }
    }
}

fn load_pack_file(path: &Path) -> Result<Vec<LoadedCommand>> {
    let text = std::fs::read_to_string(path).map_err(|source| PackError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let parsed: PackFile = toml::from_str(&text).map_err(|source| PackError::Parse {
        path: path.to_path_buf(),
        source: Box::new(source),
    })?;

    if parsed.command.is_empty() {
        return Err(PackError::Empty {
            path: path.to_path_buf(),
        });
    }

    let pack_dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    let mut out = Vec::with_capacity(parsed.command.len());
    for command in parsed.command {
        validate(&command, path)?;
        out.push(LoadedCommand {
            command,
            pack_name: parsed.pack.name.clone(),
            pack_dir: pack_dir.clone(),
        });
    }

    Ok(out)
}

fn validate(command: &Command, path: &Path) -> Result<()> {
    if command.phrases.is_empty() {
        return Err(PackError::NoPhrases {
            id: command.id.clone(),
            path: path.to_path_buf(),
        });
    }

    if let Some(tmpl) = command.action.template() {
        for referenced in template::referenced_slots(tmpl) {
            if !command.slots.iter().any(|s| s.name == referenced) {
                return Err(PackError::UnknownSlot {
                    id: command.id.clone(),
                    slot: referenced,
                    path: path.to_path_buf(),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_pack(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    const GOOD: &str = r#"
[pack]
name = "cargo"
description = "Rust build commands"

[[command]]
id = "cargo.test"
description = "Run the test suite"
phrases = ["run the tests", "run tests"]
action = { type = "shell", run = "cargo test" }
"#;

    #[test]
    fn loads_a_flat_phrase_pack() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(tmp.path(), "cargo.toml", GOOD);

        let set = CommandSet::load_dir(tmp.path()).unwrap();
        assert_eq!(set.len(), 1);
        let c = set.get("cargo.test").unwrap();
        assert_eq!(c.pack_name, "cargo");
        assert_eq!(
            c.command.phrases.for_lang("en"),
            vec!["run the tests", "run tests"]
        );
    }

    /// The bug this format exists to prevent: in the project that inspired
    /// voxide, a flat `phrases` array where a per-language map was required
    /// made serde fail, and the loader logged a warning and dropped every
    /// command in the file. Both shapes must load.
    #[test]
    fn loads_both_flat_and_per_language_phrases() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(
            tmp.path(),
            "mixed.toml",
            r#"
[pack]
name = "mixed"

[[command]]
id = "flat"
phrases = ["hello"]
action = { type = "shell", run = "echo hi" }

[[command]]
id = "bylang"
phrases.en = ["goodbye"]
phrases.ru = ["пока"]
action = { type = "shell", run = "echo bye" }
"#,
        );

        let set = CommandSet::load_dir(tmp.path()).unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(
            set.get("flat").unwrap().command.phrases.for_lang("en"),
            vec!["hello"]
        );
        assert_eq!(
            set.get("bylang").unwrap().command.phrases.for_lang("ru"),
            vec!["пока"]
        );
        // Unknown language falls back to English rather than matching nothing.
        assert_eq!(
            set.get("bylang").unwrap().command.phrases.for_lang("de"),
            vec!["goodbye"]
        );
    }

    #[test]
    fn malformed_pack_is_a_hard_error_naming_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(
            tmp.path(),
            "broken.toml",
            "[pack]\nname = \"x\"\n[[command]]\nid = 3\n",
        );

        let err = CommandSet::load_dir(tmp.path()).unwrap_err();
        assert!(matches!(err, PackError::Parse { .. }));
        assert!(err.to_string().contains("broken.toml"), "got: {err}");
    }

    #[test]
    fn command_without_phrases_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(
            tmp.path(),
            "p.toml",
            r#"
[pack]
name = "p"
[[command]]
id = "nope"
phrases = []
action = { type = "shell", run = "true" }
"#,
        );
        assert!(matches!(
            CommandSet::load_dir(tmp.path()).unwrap_err(),
            PackError::NoPhrases { .. }
        ));
    }

    #[test]
    fn action_referencing_an_undeclared_slot_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(
            tmp.path(),
            "p.toml",
            r#"
[pack]
name = "p"
[[command]]
id = "oops"
phrases = ["checkout something"]
action = { type = "shell", run = "git checkout {{branch}}" }
"#,
        );
        let err = CommandSet::load_dir(tmp.path()).unwrap_err();
        assert!(matches!(err, PackError::UnknownSlot { .. }));
        assert!(err.to_string().contains("branch"), "got: {err}");
    }

    #[test]
    fn declared_slot_satisfies_the_template() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(
            tmp.path(),
            "p.toml",
            r#"
[pack]
name = "p"
[[command]]
id = "git.checkout"
phrases = ["checkout {branch}"]
slots = [{ name = "branch", entity = "branch name" }]
action = { type = "shell", run = "git checkout {{branch}}" }
"#,
        );
        assert!(CommandSet::load_dir(tmp.path()).is_ok());
    }

    #[test]
    fn duplicate_ids_across_packs_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(tmp.path(), "a.toml", GOOD);
        write_pack(tmp.path(), "b.toml", GOOD);

        let err = CommandSet::load_dir(tmp.path()).unwrap_err();
        assert!(matches!(err, PackError::DuplicateId { .. }));
        assert!(err.to_string().contains("cargo.test"), "got: {err}");
    }

    #[test]
    fn timeout_default_is_actually_applied() {
        // Regression guard: a declared-but-unread default constant is how the
        // inspiring project ended up giving every script a 0 ms budget.
        let tmp = tempfile::tempdir().unwrap();
        write_pack(tmp.path(), "p.toml", GOOD);
        let set = CommandSet::load_dir(tmp.path()).unwrap();
        assert_eq!(
            set.get("cargo.test").unwrap().command.action.timeout_ms(),
            10_000
        );
    }

    #[test]
    fn sandbox_defaults_to_standard() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(
            tmp.path(),
            "p.toml",
            r#"
[pack]
name = "p"
[[command]]
id = "s"
phrases = ["go"]
action = { type = "lua", script = "s.lua" }
"#,
        );
        let set = CommandSet::load_dir(tmp.path()).unwrap();
        match &set.get("s").unwrap().command.action {
            Action::Lua {
                sandbox,
                timeout_ms,
                ..
            } => {
                assert_eq!(*sandbox, Sandbox::Standard);
                assert_eq!(*timeout_ms, 10_000);
            }
            other => panic!("expected lua action, got {other:?}"),
        }
    }

    #[test]
    fn fingerprint_tracks_phrases_not_descriptions() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(tmp.path(), "p.toml", GOOD);
        let before = CommandSet::load_dir(tmp.path()).unwrap().fingerprint("en");

        // Editing a description must not invalidate cached vectors.
        write_pack(
            tmp.path(),
            "p.toml",
            &GOOD.replace("Run the test suite", "Runs tests"),
        );
        let after_desc = CommandSet::load_dir(tmp.path()).unwrap().fingerprint("en");
        assert_eq!(before, after_desc);

        // Editing a phrase must.
        write_pack(
            tmp.path(),
            "p.toml",
            &GOOD.replace("run tests", "execute tests"),
        );
        let after_phrase = CommandSet::load_dir(tmp.path()).unwrap().fingerprint("en");
        assert_ne!(before, after_phrase);
    }

    #[test]
    fn fingerprint_is_language_scoped() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(
            tmp.path(),
            "p.toml",
            r#"
[pack]
name = "p"
[[command]]
id = "c"
phrases.en = ["one"]
phrases.ru = ["один"]
action = { type = "shell", run = "true" }
"#,
        );
        let set = CommandSet::load_dir(tmp.path()).unwrap();
        assert_ne!(set.fingerprint("en"), set.fingerprint("ru"));
    }

    #[test]
    fn nested_directories_are_discovered() {
        let tmp = tempfile::tempdir().unwrap();
        write_pack(tmp.path(), "nested/deep/cargo.toml", GOOD);
        assert_eq!(CommandSet::load_dir(tmp.path()).unwrap().len(), 1);
    }

    #[test]
    fn missing_directory_yields_an_empty_set() {
        // walkdir on a nonexistent path yields no entries; an empty pack dir is
        // a legitimate first-run state, not a failure.
        let set = CommandSet::load_dir("/nonexistent/voxide/packs").unwrap();
        assert!(set.is_empty());
    }
}
