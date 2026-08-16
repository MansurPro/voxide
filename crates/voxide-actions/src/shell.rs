use crate::executor::{ExecError, Executor, Outcome};
use std::io::Read;
use std::process::{Child, Command as SysCommand, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use voxide_core::{Action, LoadedCommand, Slots, template};

/// How often the deadline is checked while a child process runs.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// How long to wait for output readers to finish after a timeout kill.
///
/// The process group is already dead by this point, so this only covers the
/// scheduling delay before the reader threads observe EOF.
const DRAIN_GRACE: Duration = Duration::from_millis(250);

/// Runs an action's rendered command line through the platform shell.
#[derive(Debug, Clone, Default)]
pub struct ShellExecutor {
    dry_run: bool,
}

impl ShellExecutor {
    pub fn new() -> Self {
        Self { dry_run: false }
    }

    /// Reports the command line it would have run, without running it.
    pub fn dry_run() -> Self {
        Self { dry_run: true }
    }
}

impl Executor for ShellExecutor {
    fn execute(&self, command: &LoadedCommand, slots: &Slots) -> Result<Outcome, ExecError> {
        let Action::Shell {
            run,
            cwd,
            timeout_ms,
        } = &command.command.action
        else {
            return Err(ExecError::Unsupported(
                "shell executor given a non-shell action",
            ));
        };

        let rendered = template::render(run, slots);

        if self.dry_run {
            return Ok(Outcome {
                code: Some(0),
                stdout: rendered,
                stderr: String::new(),
                chain: command.command.chain,
            });
        }

        let (shell, flag) = platform_shell();
        let mut sys = SysCommand::new(shell);
        sys.arg(flag)
            .arg(&rendered)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // A relative `cwd` resolves against the pack directory, so a pack can
        // ship scripts and data next to its manifest and still work regardless
        // of where the user launched voxide from.
        if let Some(dir) = cwd {
            let path = std::path::Path::new(dir);
            if path.is_absolute() {
                sys.current_dir(path);
            } else {
                sys.current_dir(command.pack_dir.join(path));
            }
        }

        // Put the child in its own process group so a timeout can signal the
        // whole tree. Without this, `sh -c "sleep 30"` may fork rather than
        // exec, and killing the shell leaves `sleep` alive holding the pipe.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            sys.process_group(0);
        }

        let child = sys.spawn().map_err(|source| ExecError::Spawn {
            cmd: rendered.clone(),
            source,
        })?;

        let timeout = Duration::from_millis(*timeout_ms);
        wait_with_deadline(child, timeout, command.command.chain)
    }
}

/// Waits for `child`, killing it and its process group if `timeout` elapses.
///
/// stdout and stderr are drained on separate threads. Reading them inline
/// would deadlock as soon as a child writes more than one pipe buffer, which
/// is precisely when a command is worth timing out.
///
/// Results come back over channels rather than `JoinHandle::join`, because
/// join has no timeout: if any surviving descendant still holds the pipe open,
/// joining would block for exactly as long as the deadline was meant to
/// prevent.
fn wait_with_deadline(
    mut child: Child,
    timeout: Duration,
    chain: bool,
) -> Result<Outcome, ExecError> {
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let (out_tx, out_rx) = mpsc::channel();
    let (err_tx, err_rx) = mpsc::channel();
    std::thread::spawn(move || out_tx.send(drain(stdout)));
    std::thread::spawn(move || err_tx.send(drain(stderr)));

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait()? {
            Some(status) => break Some(status),
            None if Instant::now() >= deadline => {
                kill_group(&mut child);
                break None;
            }
            None => std::thread::sleep(POLL_INTERVAL),
        }
    };

    // On the happy path the writers are already closed, so this returns at
    // once. On the timeout path the grace bounds the wait.
    let grace = if status.is_some() {
        Duration::from_secs(30)
    } else {
        DRAIN_GRACE
    };
    let stdout = out_rx.recv_timeout(grace).unwrap_or_default();
    let stderr = err_rx.recv_timeout(DRAIN_GRACE).unwrap_or_default();

    match status {
        Some(status) => Ok(Outcome {
            code: status.code(),
            stdout,
            stderr,
            chain,
        }),
        None => Err(ExecError::Timeout(timeout)),
    }
}

/// Terminates the child and every process in its group.
fn kill_group(child: &mut Child) {
    #[cfg(unix)]
    {
        // Negative pid targets the whole group, which `process_group(0)` made
        // the child the leader of. Failures are uninteresting: the usual cause
        // is that everything already exited.
        let pid = child.id() as i32;
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }

    // Also signal the child directly. On Windows this is the only mechanism
    // available without a Job object, so a shell that spawned grandchildren
    // can still leave them running; the drain grace keeps that from stalling
    // the caller.
    let _ = child.kill();
    let _ = child.wait();
}

fn drain<R: Read>(reader: Option<R>) -> String {
    let Some(mut reader) = reader else {
        return String::new();
    };
    let mut buf = Vec::new();
    let _ = reader.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

fn platform_shell() -> (&'static str, &'static str) {
    if cfg!(windows) {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxide_core::SlotValue;
    use voxide_core::pack::{Command, LoadedCommand, Phrases};

    fn shell_command(run: &str, timeout_ms: u64) -> LoadedCommand {
        LoadedCommand {
            command: Command {
                id: "t".into(),
                description: String::new(),
                phrases: Phrases::Flat(vec!["go".into()]),
                slots: Vec::new(),
                action: Action::Shell {
                    run: run.into(),
                    cwd: None,
                    timeout_ms,
                },
                chain: false,
            },
            pack_name: "p".into(),
            pack_dir: ".".into(),
        }
    }

    #[test]
    fn runs_a_command_and_captures_stdout() {
        let c = shell_command("echo hello", 5_000);
        let out = ShellExecutor::new().execute(&c, &Slots::new()).unwrap();
        assert!(out.success());
        assert_eq!(out.stdout.trim(), "hello");
    }

    #[test]
    fn reports_a_nonzero_exit_code() {
        let c = shell_command("exit 3", 5_000);
        let out = ShellExecutor::new().execute(&c, &Slots::new()).unwrap();
        assert_eq!(out.code, Some(3));
        assert!(!out.success());
    }

    #[test]
    fn captures_stderr() {
        let c = shell_command("echo oops 1>&2", 5_000);
        let out = ShellExecutor::new().execute(&c, &Slots::new()).unwrap();
        assert_eq!(out.stderr.trim(), "oops");
    }

    #[test]
    fn interpolates_slots() {
        let c = shell_command("echo {{name}}", 5_000);
        let mut slots = Slots::new();
        slots.insert("name".into(), SlotValue::Text("world".into()));
        let out = ShellExecutor::new().execute(&c, &slots).unwrap();
        assert_eq!(out.stdout.trim(), "world");
    }

    /// The correction this executor exists for. A blocking child must be
    /// stopped by the deadline; an instruction-count hook of the kind embedded
    /// interpreters usually offer would let this run to completion.
    #[test]
    #[cfg(unix)]
    fn a_blocking_command_hits_the_deadline() {
        let c = shell_command("sleep 30", 200);
        let start = Instant::now();
        let err = ShellExecutor::new().execute(&c, &Slots::new()).unwrap_err();

        assert!(matches!(err, ExecError::Timeout(_)), "got {err:?}");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "deadline did not fire promptly: {:?}",
            start.elapsed()
        );
    }

    /// Output larger than a pipe buffer must not deadlock the waiter.
    #[test]
    #[cfg(unix)]
    fn large_output_does_not_deadlock() {
        let c = shell_command("seq 1 200000", 30_000);
        let out = ShellExecutor::new().execute(&c, &Slots::new()).unwrap();
        assert!(out.success());
        assert!(
            out.stdout.len() > 128 * 1024,
            "only got {} bytes",
            out.stdout.len()
        );
    }

    #[test]
    fn dry_run_reports_without_executing() {
        let c = shell_command("echo {{name}}", 5_000);
        let mut slots = Slots::new();
        slots.insert("name".into(), SlotValue::Text("world".into()));
        let out = ShellExecutor::dry_run().execute(&c, &slots).unwrap();
        assert_eq!(out.stdout, "echo world");
    }

    #[test]
    fn unfilled_optional_slot_leaves_no_dangling_argument() {
        let c = shell_command("echo done {{filter}}", 5_000);
        let out = ShellExecutor::dry_run().execute(&c, &Slots::new()).unwrap();
        assert_eq!(out.stdout, "echo done");
    }

    #[test]
    fn relative_cwd_resolves_against_the_pack_directory() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("sub")).unwrap();
        std::fs::write(tmp.path().join("sub/marker.txt"), "found").unwrap();

        let mut c = shell_command("cat marker.txt", 5_000);
        c.pack_dir = tmp.path().to_path_buf();
        c.command.action = Action::Shell {
            run: "cat marker.txt".into(),
            cwd: Some("sub".into()),
            timeout_ms: 5_000,
        };

        let out = ShellExecutor::new().execute(&c, &Slots::new()).unwrap();
        assert_eq!(out.stdout.trim(), "found");
    }

    #[test]
    fn chain_flag_propagates_to_the_outcome() {
        let mut c = shell_command("true", 5_000);
        c.command.chain = true;
        let out = ShellExecutor::new().execute(&c, &Slots::new()).unwrap();
        assert!(out.chain);
    }
}
