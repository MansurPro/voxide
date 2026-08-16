use std::time::Duration;
use voxide_core::{Action, LoadedCommand, Slots};

/// Result of running one command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// Command exit status, where the backend has one.
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// Whether to keep listening without a wake word.
    pub chain: bool,
}

impl Outcome {
    pub fn success(&self) -> bool {
        self.code.is_none_or(|c| c == 0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("action timed out after {0:?}")]
    Timeout(Duration),

    #[error("failed to spawn `{cmd}`: {source}")]
    Spawn {
        cmd: String,
        #[source]
        source: std::io::Error,
    },

    #[error("io error during execution: {0}")]
    Io(#[from] std::io::Error),

    #[error("action type `{0}` is not supported by this build")]
    Unsupported(&'static str),

    #[error("required slot `{slot}` was not filled for command `{id}`")]
    MissingSlot { id: String, slot: String },

    #[error("{0}")]
    Backend(String),
}

/// Runs a matched command with its extracted slots.
pub trait Executor: Send + Sync {
    fn execute(&self, command: &LoadedCommand, slots: &Slots) -> Result<Outcome, ExecError>;
}

/// Fails a command whose non-optional slots were not filled.
///
/// Running `git checkout` with an empty branch is worse than not running it:
/// the user asked for something specific and would get a silent no-op.
pub fn check_required_slots(command: &LoadedCommand, slots: &Slots) -> Result<(), ExecError> {
    for def in &command.command.slots {
        if def.optional || def.default.is_some() {
            continue;
        }
        if !slots.contains_key(&def.name) {
            return Err(ExecError::MissingSlot {
                id: command.command.id.clone(),
                slot: def.name.clone(),
            });
        }
    }
    Ok(())
}

/// Applies declared defaults to any slot the extractor did not fill.
pub fn apply_defaults(command: &LoadedCommand, slots: &mut Slots) {
    for def in &command.command.slots {
        if let Some(default) = &def.default
            && !slots.contains_key(&def.name)
        {
            slots.insert(def.name.clone(), voxide_core::SlotValue::parse(default));
        }
    }
}

/// Dispatches to whichever backend the action names.
///
/// Backends that are not compiled in report [`ExecError::Unsupported`] rather
/// than being silently skipped.
pub struct DefaultExecutor {
    shell: crate::shell::ShellExecutor,
}

impl DefaultExecutor {
    pub fn new() -> Self {
        Self {
            shell: crate::shell::ShellExecutor::default(),
        }
    }

    /// Prints what would run instead of running it.
    pub fn dry_run() -> Self {
        Self {
            shell: crate::shell::ShellExecutor::dry_run(),
        }
    }
}

impl Default for DefaultExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor for DefaultExecutor {
    fn execute(&self, command: &LoadedCommand, slots: &Slots) -> Result<Outcome, ExecError> {
        let mut slots = slots.clone();
        apply_defaults(command, &mut slots);
        check_required_slots(command, &slots)?;

        match &command.command.action {
            Action::Shell { .. } => self.shell.execute(command, &slots),
            Action::Lua { .. } => Err(ExecError::Unsupported("lua")),
            Action::Text { .. } => Err(ExecError::Unsupported("text")),
            Action::Keys { .. } => Err(ExecError::Unsupported("keys")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxide_core::pack::{Action, Command, LoadedCommand, Phrases};
    use voxide_core::{SlotDef, SlotValue};

    fn command_with_slots(slots: Vec<SlotDef>) -> LoadedCommand {
        LoadedCommand {
            command: Command {
                id: "t".into(),
                description: String::new(),
                phrases: Phrases::Flat(vec!["go".into()]),
                slots,
                action: Action::Shell {
                    run: "true".into(),
                    cwd: None,
                    timeout_ms: 1000,
                },
                chain: false,
            },
            pack_name: "p".into(),
            pack_dir: ".".into(),
        }
    }

    fn slot_def(name: &str, optional: bool, default: Option<&str>) -> SlotDef {
        SlotDef {
            name: name.into(),
            entity: "thing".into(),
            optional,
            default: default.map(str::to_owned),
        }
    }

    #[test]
    fn missing_required_slot_is_an_error() {
        let c = command_with_slots(vec![slot_def("branch", false, None)]);
        let err = check_required_slots(&c, &Slots::new()).unwrap_err();
        assert!(matches!(err, ExecError::MissingSlot { .. }));
    }

    #[test]
    fn optional_slot_may_be_absent() {
        let c = command_with_slots(vec![slot_def("filter", true, None)]);
        assert!(check_required_slots(&c, &Slots::new()).is_ok());
    }

    #[test]
    fn defaults_fill_unset_slots() {
        let c = command_with_slots(vec![slot_def("branch", false, Some("main"))]);
        let mut slots = Slots::new();
        apply_defaults(&c, &mut slots);
        assert_eq!(slots.get("branch"), Some(&SlotValue::Text("main".into())));
        assert!(check_required_slots(&c, &slots).is_ok());
    }

    #[test]
    fn defaults_do_not_overwrite_extracted_values() {
        let c = command_with_slots(vec![slot_def("branch", false, Some("main"))]);
        let mut slots = Slots::new();
        slots.insert("branch".into(), SlotValue::Text("dev".into()));
        apply_defaults(&c, &mut slots);
        assert_eq!(slots.get("branch"), Some(&SlotValue::Text("dev".into())));
    }

    #[test]
    fn outcome_success_tracks_exit_code() {
        let mk = |code| Outcome {
            code,
            stdout: String::new(),
            stderr: String::new(),
            chain: false,
        };
        assert!(mk(Some(0)).success());
        assert!(!mk(Some(1)).success());
        assert!(mk(None).success());
    }

    #[test]
    fn uncompiled_backends_report_unsupported() {
        let mut c = command_with_slots(vec![]);
        c.command.action = Action::Keys {
            keys: "ctrl+p".into(),
        };
        let err = DefaultExecutor::new()
            .execute(&c, &Slots::new())
            .unwrap_err();
        assert!(matches!(err, ExecError::Unsupported("keys")));
    }
}
