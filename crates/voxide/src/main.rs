//! voxide — offline, semantic voice control for your development workflow.

mod eval;
mod listen;
mod models;
mod packs;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use voxide_actions::{DefaultExecutor, ExecError, Executor};
use voxide_core::CommandSet;
use voxide_intent::{LexicalMatcher, Matcher};

#[derive(Parser)]
#[command(
    name = "voxide",
    version,
    about = "Offline, semantic voice control for your development workflow",
    long_about = None
)]
struct Cli {
    /// Directory containing command packs.
    #[arg(long, global = true, value_name = "DIR")]
    packs: Option<PathBuf>,

    /// Language code used to select training phrases.
    #[arg(long, global = true, default_value = voxide_core::DEFAULT_LANG)]
    lang: String,

    /// Minimum score for a match to be acted on. Defaults to the active
    /// backend's own floor, since lexical and semantic scores are not on a
    /// comparable scale.
    #[arg(long, global = true, value_name = "SCORE")]
    threshold: Option<f32>,

    /// Directory holding downloaded models.
    #[arg(long, global = true, value_name = "DIR")]
    models: Option<PathBuf>,

    /// Increase log verbosity. Repeat for more detail.
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Match a phrase and run the command it resolves to.
    Say {
        /// The phrase, as you would speak it.
        #[arg(required = true, num_args = 1..)]
        words: Vec<String>,

        /// Show what would run without running it.
        #[arg(long)]
        dry_run: bool,
    },

    /// Explain how a phrase is matched, without running anything.
    Why {
        #[arg(required = true, num_args = 1..)]
        words: Vec<String>,

        /// Number of candidates to show.
        #[arg(long, default_value_t = 5)]
        top: usize,
    },

    /// Inspect installed command packs.
    Packs {
        #[command(subcommand)]
        action: PacksCmd,
    },

    /// Listen continuously and run commands as they are spoken.
    Run {
        /// Read audio from a WAV file instead of a microphone.
        #[arg(long, value_name = "FILE")]
        from: Option<PathBuf>,

        /// Input device name substring. Defaults to the system default.
        #[arg(long, value_name = "NAME")]
        device: Option<String>,

        /// Require this wake word before acting. Omit for always-on.
        #[arg(long, value_name = "WORD")]
        wake: Option<String>,

        /// Speech model directory.
        #[arg(long, value_name = "DIR")]
        model: Option<PathBuf>,

        /// Report what would run without running it.
        #[arg(long)]
        dry_run: bool,
    },

    /// Download and inspect speech and embedding models.
    Models {
        #[command(subcommand)]
        action: ModelsCmd,
    },

    /// Score the matcher against a golden phrase corpus.
    Eval {
        /// Directory of corpus `*.toml` files.
        #[arg(long, default_value = "tests/corpus", value_name = "DIR")]
        corpus: PathBuf,

        /// Also score the lexical baseline, to quantify what semantic
        /// matching buys over conventional phrase matching.
        #[arg(long)]
        compare: bool,

        /// Print a top-1 accuracy curve across thresholds.
        #[arg(long)]
        sweep: bool,

        /// Fail if top-1 accuracy falls below this. Used as a CI gate.
        #[arg(long, value_name = "RATIO")]
        min_accuracy: Option<f32>,

        /// Hide the per-case failure listing.
        #[arg(long)]
        quiet: bool,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Show every model voxide can fetch, and whether it is installed.
    List,
    /// Download a model, verifying it against a pinned checksum.
    Pull {
        /// Model id. Omit to fetch everything in the catalog.
        id: Option<String>,

        /// Re-download even if the model is already installed.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Subcommand)]
enum PacksCmd {
    /// List every loaded command.
    List,
    /// Print the directory packs are loaded from.
    Dir,
}

/// Exit code for "the phrase did not resolve to a command".
///
/// Distinct from 1 so a caller wiring voxide into a script can tell "nothing
/// matched" apart from "the command ran and failed".
const EXIT_NO_MATCH: u8 = 2;

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let result = match &cli.command {
        Cmd::Say { words, dry_run } => cmd_say(&cli, &words.join(" "), *dry_run),
        Cmd::Why { words, top } => cmd_why(&cli, &words.join(" "), *top),
        Cmd::Packs { action } => cmd_packs(&cli, action),
        Cmd::Run {
            from,
            device,
            wake,
            model,
            dry_run,
        } => cmd_run(
            &cli,
            from.as_deref(),
            device.as_deref(),
            wake.as_deref(),
            model.as_deref(),
            *dry_run,
        ),
        Cmd::Models { action } => match action {
            ModelsCmd::List => models::list(&models_root(&cli)),
            ModelsCmd::Pull { id, force } => {
                models::pull(&models_root(&cli), id.as_deref(), *force)
            }
        },
        Cmd::Eval {
            corpus,
            compare,
            sweep,
            min_accuracy,
            quiet,
        } => cmd_eval(&cli, corpus, *compare, *sweep, *min_accuracy, *quiet),
    };

    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        // A phrase that matched nothing is an ordinary outcome, not a crash;
        // `cmd_say` has already explained it with the near misses.
        Err(e) if e.is::<NoMatch>() => std::process::ExitCode::from(EXIT_NO_MATCH),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Marker error for a phrase that resolved to nothing.
#[derive(Debug)]
struct NoMatch;

impl std::fmt::Display for NoMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no command matched")
    }
}

impl std::error::Error for NoMatch {}

fn models_root(cli: &Cli) -> PathBuf {
    cli.models.clone().unwrap_or_else(models::default_root)
}

/// The score floor to apply: an explicit `--threshold`, else the backend's own.
fn threshold(cli: &Cli, matcher: &dyn Matcher) -> f32 {
    cli.threshold.unwrap_or_else(|| matcher.default_threshold())
}

fn init_tracing(verbose: u8) {
    let default = match verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_env("VOXIDE_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

/// Loads packs and builds the matcher the current build supports.
fn load(cli: &Cli) -> Result<(CommandSet, Box<dyn Matcher>)> {
    let dir = packs::resolve_dir(cli.packs.as_deref());
    let set = packs::load(&dir)?;
    let matcher = build_matcher(&set, &cli.lang);
    Ok((set, matcher))
}

fn build_matcher(set: &CommandSet, lang: &str) -> Box<dyn Matcher> {
    // The semantic backend is the point of the project, so prefer it whenever
    // it is compiled in and its model is available. The lexical matcher is the
    // always-present fallback and the baseline `voxide eval` compares against.
    #[cfg(feature = "embed")]
    {
        match voxide_intent::SemanticMatcher::load(set, lang) {
            Ok(m) => {
                return Box::new(voxide_intent::HybridMatcher::new(
                    LexicalMatcher::new(set, lang),
                    m,
                ));
            }
            Err(e) => tracing::warn!(
                error = %e,
                "semantic matcher unavailable, falling back to lexical matching; \
                 run `voxide models pull` to enable it"
            ),
        }
    }

    Box::new(LexicalMatcher::new(set, lang))
}

fn cmd_say(cli: &Cli, phrase: &str, dry_run: bool) -> Result<()> {
    let (set, matcher) = load(cli)?;

    let threshold = threshold(cli, matcher.as_ref());

    let Some(top) = matcher.best(phrase, threshold) else {
        let near = matcher.rank(phrase, 3);
        eprintln!("no command matched {phrase:?} at threshold {threshold:.2}");
        if !near.is_empty() {
            eprintln!("\nclosest candidates:");
            for m in &near {
                eprintln!("  {:.3}  {}", m.score, m.id);
            }
            eprintln!("\nrun `voxide why {phrase:?}` for detail");
        }
        return Err(NoMatch.into());
    };

    let Some(command) = set.get(&top.id) else {
        bail!("matcher returned unknown command id {:?}", top.id);
    };

    // Slots come from the phrase that actually won, so the markers line up
    // with the wording the user used.
    let slots = top
        .via
        .as_deref()
        .map(|p| voxide_intent::extract_slots(p, phrase))
        .unwrap_or_default();

    tracing::info!(id = %top.id, score = top.score, ?slots, "matched");

    let executor = if dry_run {
        DefaultExecutor::dry_run()
    } else {
        DefaultExecutor::new()
    };

    match executor.execute(command, &slots) {
        Ok(outcome) => {
            if dry_run {
                println!("{}  ({:.3})", outcome.stdout, top.score);
                return Ok(());
            }
            print!("{}", outcome.stdout);
            eprint!("{}", outcome.stderr);
            if !outcome.success() {
                bail!("command `{}` exited with {:?}", top.id, outcome.code);
            }
            Ok(())
        }
        Err(ExecError::MissingSlot { slot, .. }) => {
            bail!("matched `{}` but could not work out the {slot}", top.id)
        }
        Err(e) => Err(e.into()),
    }
}

fn cmd_why(cli: &Cli, phrase: &str, top: usize) -> Result<()> {
    let (_set, matcher) = load(cli)?;
    let ranked = matcher.rank(phrase, top.max(1));
    let threshold = threshold(cli, matcher.as_ref());

    println!("phrase:    {phrase:?}");
    println!("backend:   {}", matcher.backend());
    println!("threshold: {threshold:.2}\n");

    if ranked.is_empty() {
        println!("no candidates scored above zero");
        return Ok(());
    }

    let width = ranked.iter().map(|m| m.id.len()).max().unwrap_or(0);
    for (i, m) in ranked.iter().enumerate() {
        let marker = if i == 0 && m.score >= threshold {
            "->"
        } else {
            "  "
        };
        print!("{marker} {:.3}  {:width$}", m.score, m.id);
        if let Some(via) = &m.via {
            print!("   via {via:?}");
        }
        println!();
    }

    // The gap between first and second is what makes a match trustworthy; a
    // narrow gap is the signal to add a distinguishing phrase to the pack.
    if ranked.len() > 1 {
        let gap = ranked[0].score - ranked[1].score;
        println!("\nmargin over runner-up: {gap:.3}");
        if gap < 0.05 {
            println!("(narrow: consider adding a distinguishing phrase to the pack)");
        }
    }

    Ok(())
}

fn cmd_run(
    cli: &Cli,
    from: Option<&std::path::Path>,
    device: Option<&str>,
    wake: Option<&str>,
    model: Option<&std::path::Path>,
    dry_run: bool,
) -> Result<()> {
    let (set, matcher) = load(cli)?;
    let threshold = threshold(cli, matcher.as_ref());

    let input = match from {
        Some(path) => listen::Input::Wav(path),
        None => listen::Input::Mic(device),
    };

    listen::run(
        &set,
        matcher.as_ref(),
        listen::Options {
            input,
            wake_word: wake,
            model,
            dry_run,
            config: voxide_pipeline::Config {
                threshold,
                ..Default::default()
            },
        },
    )
}

fn cmd_eval(
    cli: &Cli,
    corpus_dir: &std::path::Path,
    compare: bool,
    sweep: bool,
    min_accuracy: Option<f32>,
    quiet: bool,
) -> Result<()> {
    let dir = packs::resolve_dir(cli.packs.as_deref());
    let set = packs::load(&dir)?;
    let cases = eval::load_corpus(corpus_dir)?;

    if cases.is_empty() {
        bail!("no corpus cases found in {}", corpus_dir.display());
    }

    println!(
        "corpus: {} case(s) from {}\npacks:  {} command(s) from {}\n",
        cases.len(),
        corpus_dir.display(),
        set.len(),
        dir.display()
    );

    // The baseline is always available, so `--compare` can always report a
    // delta rather than an unanchored number. Each backend is scored at its own
    // floor: holding one number across both would not be a fairer comparison,
    // it would just measure the wrong backend's calibration.
    let baseline = LexicalMatcher::new(&set, &cli.lang);
    let baseline_report = eval::run(&baseline, &cases, threshold(cli, &baseline));

    let primary: Box<dyn Matcher> = build_matcher(&set, &cli.lang);
    let primary_is_baseline = primary.backend() == baseline.backend();
    let report = eval::run(primary.as_ref(), &cases, threshold(cli, primary.as_ref()));

    if compare && !primary_is_baseline {
        eval::print_report(&baseline_report, false);
        eval::print_report(&report, !quiet);
        let delta = (report.top1_accuracy() - baseline_report.top1_accuracy()) * 100.0;
        println!(
            "\n{} matching is {delta:+.1} points over the lexical baseline",
            report.backend
        );
    } else {
        if compare {
            println!(
                "note: only the lexical backend is compiled in, so there is nothing \
                 to compare against.\n      Rebuild with `--features embed` for the \
                 semantic backend.\n"
            );
        }
        eval::print_report(&report, !quiet);
    }

    if sweep {
        eval::print_sweep(&eval::sweep(primary.as_ref(), &cases));
    }

    if let Some(floor) = min_accuracy
        && report.top1_accuracy() < floor
    {
        bail!(
            "top-1 accuracy {:.1}% is below the required {:.1}%",
            report.top1_accuracy() * 100.0,
            floor * 100.0
        );
    }

    Ok(())
}

fn cmd_packs(cli: &Cli, action: &PacksCmd) -> Result<()> {
    let dir = packs::resolve_dir(cli.packs.as_deref());

    match action {
        PacksCmd::Dir => {
            println!("{}", dir.display());
            Ok(())
        }
        PacksCmd::List => {
            let set = packs::load(&dir)?;
            if set.is_empty() {
                println!("no packs found in {}", dir.display());
                return Ok(());
            }

            println!("{} command(s) from {}\n", set.len(), dir.display());
            let mut current = String::new();
            for lc in set.commands() {
                if lc.pack_name != current {
                    current = lc.pack_name.clone();
                    println!("{current}");
                }
                let phrases = lc.command.phrases.for_lang(&cli.lang);
                println!(
                    "  {:<24} {} phrase(s)   {}",
                    lc.command.id,
                    phrases.len(),
                    lc.command.description
                );
            }
            Ok(())
        }
    }
}
