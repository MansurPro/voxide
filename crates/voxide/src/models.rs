//! The `voxide models` subcommand.

use anyhow::{Result, bail};
use std::io::Write;
use std::path::{Path, PathBuf};
use voxide_models::{CATALOG, HttpFetcher, Progress};

/// Default install root, overridable with `--models`.
pub fn default_root() -> PathBuf {
    std::env::var_os("VOXIDE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Draws a single updating progress line on stderr.
///
/// stderr rather than stdout so `voxide models pull` stays pipe-friendly.
struct Bar {
    total: Option<u64>,
    done: u64,
    last_percent: u64,
}

impl Bar {
    fn new() -> Self {
        Self {
            total: None,
            done: 0,
            last_percent: u64::MAX,
        }
    }
}

impl Progress for Bar {
    fn start(&mut self, name: &str, total: Option<u64>) {
        self.total = total;
        self.done = 0;
        self.last_percent = u64::MAX;
        match total {
            Some(t) => eprintln!("  fetching {name} ({:.1} MB)", t as f64 / 1_048_576.0),
            None => eprintln!("  fetching {name}"),
        }
    }

    fn advance(&mut self, bytes: u64) {
        self.done += bytes;
        let Some(total) = self.total.filter(|t| *t > 0) else {
            return;
        };

        // Repaint only when the whole-number percentage changes, so a fast
        // download does not spend its time writing to the terminal.
        let percent = self.done * 100 / total;
        if percent != self.last_percent {
            self.last_percent = percent;
            eprint!(
                "\r    {percent:3}%  {:.1}/{:.1} MB",
                self.done as f64 / 1_048_576.0,
                total as f64 / 1_048_576.0
            );
            let _ = std::io::stderr().flush();
        }
    }

    fn finish(&mut self, _name: &str) {
        if self.total.is_some() {
            eprintln!();
        }
    }
}

pub fn list(root: &Path) -> Result<()> {
    println!("install root: {}\n", root.display());

    for spec in CATALOG {
        let installed = voxide_models::is_installed(root, spec);
        let mark = if installed { "installed" } else { "-" };
        println!("{:<16} {:>10}   {}", spec.id, mark, spec.description);
        println!(
            "{:<16} {:>10}   {} · {} · {}",
            "", "", spec.task, spec.approx_size, spec.license
        );
        if installed {
            println!(
                "{:<16} {:>10}   {}",
                "",
                "",
                voxide_models::model_dir(root, spec.id).display()
            );
        }
        println!();
    }

    Ok(())
}

pub fn pull(root: &Path, id: Option<&str>, force: bool) -> Result<()> {
    let targets: Vec<_> = match id {
        Some(id) => match voxide_models::find(id) {
            Some(spec) => vec![spec],
            None => {
                bail!("unknown model {id:?}. Run `voxide models list` to see what is available.")
            }
        },
        None => CATALOG.iter().collect(),
    };

    let fetcher = HttpFetcher::new();
    let mut bar = Bar::new();
    let mut changed = 0;

    for spec in targets {
        println!("{} — {}", spec.id, spec.description);
        match voxide_models::install(&fetcher, root, spec, force, &mut bar) {
            Ok(true) => {
                changed += 1;
                println!(
                    "  installed to {}",
                    voxide_models::model_dir(root, spec.id).display()
                );
            }
            Ok(false) => println!("  already present (use --force to re-download)"),
            // Returned rather than printed here; `main` renders it once.
            Err(e) => return Err(e.into()),
        }
    }

    if changed > 0 {
        println!("\nDone. Point voxide at it with `voxide run --model <dir>`.");
    }
    Ok(())
}
