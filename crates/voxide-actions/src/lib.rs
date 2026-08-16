//! Executing a matched command.
//!
//! Every action runs under a real wall-clock deadline enforced from outside
//! the work itself. That is a deliberate correction: a bytecode-instruction
//! hook, the usual way to time-limit an embedded interpreter, cannot interrupt
//! a blocking host call, so a script that sleeps or issues a slow HTTP request
//! runs as long as it likes. A deadline on a worker thread bounds both.

pub mod executor;
pub mod shell;

pub use executor::{
    DefaultExecutor, ExecError, Executor, Outcome, apply_defaults, check_required_slots,
};
pub use shell::ShellExecutor;
