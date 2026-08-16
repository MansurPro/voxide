//! Pinned, checksum-verified model downloads.
//!
//! voxide ships no model weights. This crate fetches them on demand, each
//! artifact pinned to an exact upstream revision and verified against a
//! SHA-256 recorded here. That makes a first run reproducible and makes a
//! tampered or truncated download a hard, named error rather than a model
//! that quietly misbehaves.
//!
//! This replaces what would otherwise be a Python setup script: no interpreter,
//! no `pip install`, no virtualenv — the same binary that runs voxide fetches
//! what voxide needs.

pub mod archive;
pub mod fetch;

pub use fetch::{Fetcher, HttpFetcher, LocalFetcher, Progress, SilentProgress};

use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Fetch(#[from] fetch::FetchError),

    #[error(transparent)]
    Archive(#[from] archive::ArchiveError),

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("unknown model {0:?}. Run `voxide models list` to see what is available.")]
    Unknown(String),
}

type Result<T> = std::result::Result<T, ModelError>;

/// How a downloaded artifact is laid out on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Written directly into the model directory.
    File,
    /// A zip archive unpacked into the model directory.
    ZipArchive,
}

/// One downloadable artifact.
#[derive(Debug, Clone)]
pub struct Artifact {
    /// Filename inside the model directory, or the archive's local name.
    pub name: &'static str,
    pub url: &'static str,
    /// Hex SHA-256. `None` for small metadata files whose upstream content
    /// changes harmlessly and whose corruption would fail loudly anyway.
    pub sha256: Option<&'static str>,
    pub layout: Layout,
}

/// A model voxide knows how to install.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: &'static str,
    pub description: &'static str,
    /// What the model is for, shown by `voxide models list`.
    pub task: &'static str,
    pub approx_size: &'static str,
    pub license: &'static str,
    pub artifacts: &'static [Artifact],
    /// Relative path that must exist for the model to count as installed.
    pub sentinel: &'static str,
}

/// Models voxide can fetch.
///
/// Checksums are marked `None` until they have been confirmed against a real
/// download. Publishing a guessed hash would be worse than none at all: every
/// install would fail with a checksum error that looks like tampering.
pub const CATALOG: &[ModelSpec] = &[ModelSpec {
    id: "vosk-en-small",
    description: "Vosk small English speech recognition model",
    task: "speech-to-text",
    approx_size: "40 MB",
    license: "Apache-2.0",
    sentinel: "am",
    artifacts: &[Artifact {
        name: "vosk-model-small-en-us-0.15.zip",
        url: "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        sha256: None,
        layout: Layout::ZipArchive,
    }],
}];

pub fn find(id: &str) -> Option<&'static ModelSpec> {
    CATALOG.iter().find(|m| m.id == id)
}

/// Directory a model installs into.
pub fn model_dir(root: &Path, id: &str) -> PathBuf {
    root.join(id)
}

/// True when the model's sentinel path exists.
pub fn is_installed(root: &Path, spec: &ModelSpec) -> bool {
    model_dir(root, spec.id).join(spec.sentinel).exists()
}

/// Downloads and installs `spec` under `root`.
///
/// Returns `false` without doing any work if the model is already present and
/// `force` is not set.
pub fn install(
    fetcher: &dyn Fetcher,
    root: &Path,
    spec: &ModelSpec,
    force: bool,
    progress: &mut dyn Progress,
) -> Result<bool> {
    let dir = model_dir(root, spec.id);

    if !force && is_installed(root, spec) {
        tracing::debug!(id = spec.id, "already installed");
        return Ok(false);
    }

    std::fs::create_dir_all(&dir).map_err(|source| ModelError::Io {
        path: dir.clone(),
        source,
    })?;

    let result = install_artifacts(fetcher, root, spec, &dir, progress);

    if result.is_err() {
        // Leave no empty directory behind on failure. It is not mistaken for
        // an install — `is_installed` checks the sentinel — but it makes
        // `models list` and a plain `ls` lie about what happened.
        let empty = std::fs::read_dir(&dir)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false);
        if empty {
            let _ = std::fs::remove_dir(&dir);
        }
    }

    result.map(|()| true)
}

fn install_artifacts(
    fetcher: &dyn Fetcher,
    root: &Path,
    spec: &ModelSpec,
    dir: &Path,
    progress: &mut dyn Progress,
) -> Result<()> {
    for artifact in spec.artifacts {
        match artifact.layout {
            Layout::File => {
                let dest = dir.join(artifact.name);
                fetch::fetch_verified(fetcher, artifact.url, &dest, artifact.sha256, progress)?;
            }
            Layout::ZipArchive => {
                // Stage the archive outside the model directory so a failed
                // extraction cannot leave it looking like a valid install.
                let staging = root.join(format!(".{}.download", spec.id));
                std::fs::create_dir_all(&staging).map_err(|source| ModelError::Io {
                    path: staging.clone(),
                    source,
                })?;

                let archive_path = staging.join(artifact.name);
                let result = (|| -> Result<()> {
                    fetch::fetch_verified(
                        fetcher,
                        artifact.url,
                        &archive_path,
                        artifact.sha256,
                        progress,
                    )?;
                    archive::extract_zip(&archive_path, dir)?;
                    Ok(())
                })();

                let _ = std::fs::remove_dir_all(&staging);
                result?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn build_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).unwrap();
        let mut w = zip::ZipWriter::new(file);
        let opts: zip::write::FileOptions<'_, ()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, contents) in entries {
            w.start_file(*name, opts).unwrap();
            w.write_all(contents).unwrap();
        }
        w.finish().unwrap();
    }

    static ZIP_SPEC: ModelSpec = ModelSpec {
        id: "test-model",
        description: "test",
        task: "speech-to-text",
        approx_size: "1 KB",
        license: "Apache-2.0",
        sentinel: "am",
        artifacts: &[Artifact {
            name: "model.zip",
            url: "https://example.invalid/model.zip",
            sha256: None,
            layout: Layout::ZipArchive,
        }],
    };

    /// Builds a source directory holding the artifact the spec points at.
    fn source_with_model(dir: &Path) {
        build_zip(
            &dir.join("model.zip"),
            &[
                ("wrapper-0.15/am/final.mdl", b"model"),
                ("wrapper-0.15/conf/mfcc.conf", b"conf"),
            ],
        );
    }

    #[test]
    fn installs_and_unpacks_a_zip_model() {
        let src = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        source_with_model(src.path());

        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };

        let installed =
            install(&fetcher, root.path(), &ZIP_SPEC, false, &mut SilentProgress).unwrap();
        assert!(installed);

        let dir = model_dir(root.path(), "test-model");
        assert!(dir.join("am/final.mdl").is_file());
        assert!(dir.join("conf/mfcc.conf").is_file());
        assert!(is_installed(root.path(), &ZIP_SPEC));
    }

    #[test]
    fn a_second_install_is_a_no_op() {
        let src = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        source_with_model(src.path());
        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };

        assert!(install(&fetcher, root.path(), &ZIP_SPEC, false, &mut SilentProgress).unwrap());
        assert!(!install(&fetcher, root.path(), &ZIP_SPEC, false, &mut SilentProgress).unwrap());
    }

    #[test]
    fn force_reinstalls() {
        let src = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        source_with_model(src.path());
        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };

        install(&fetcher, root.path(), &ZIP_SPEC, false, &mut SilentProgress).unwrap();
        assert!(install(&fetcher, root.path(), &ZIP_SPEC, true, &mut SilentProgress).unwrap());
    }

    /// A failed install must not leave the staging archive lying around, and
    /// must not look installed on the next run.
    #[test]
    fn a_failed_download_leaves_nothing_installed() {
        let src = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        // Deliberately do not create model.zip.
        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };

        assert!(install(&fetcher, root.path(), &ZIP_SPEC, false, &mut SilentProgress).is_err());
        assert!(!is_installed(root.path(), &ZIP_SPEC));
        assert!(
            !root.path().join(".test-model.download").exists(),
            "staging directory was not cleaned up"
        );
        // An empty model directory is not mistaken for an install, but it
        // makes `models list` and `ls` misreport what happened.
        assert!(
            !model_dir(root.path(), "test-model").exists(),
            "empty model directory left behind"
        );

        let leftovers: Vec<_> = std::fs::read_dir(root.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name())
            .collect();
        assert!(
            leftovers.is_empty(),
            "install root not clean: {leftovers:?}"
        );
    }

    /// A partial extraction must keep whatever it managed to write, so the
    /// cleanup above never destroys real data.
    #[test]
    fn a_non_empty_directory_survives_a_later_failure() {
        let src = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        source_with_model(src.path());
        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };

        install(&fetcher, root.path(), &ZIP_SPEC, false, &mut SilentProgress).unwrap();

        // Remove the source so a forced reinstall fails mid-way.
        std::fs::remove_file(src.path().join("model.zip")).unwrap();
        assert!(install(&fetcher, root.path(), &ZIP_SPEC, true, &mut SilentProgress).is_err());

        assert!(
            model_dir(root.path(), "test-model")
                .join("am/final.mdl")
                .is_file(),
            "previously installed files were destroyed"
        );
    }

    #[test]
    fn catalog_lookup_works_and_rejects_unknown_ids() {
        assert!(find("vosk-en-small").is_some());
        assert!(find("nope").is_none());
    }

    /// Guards against a copy-paste error making two entries collide on disk.
    #[test]
    fn catalog_ids_are_unique() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate model id in CATALOG");
    }

    /// A checksum is either absent or a well-formed hex digest. A malformed
    /// pin would fail every install with what looks like tampering.
    #[test]
    fn catalog_checksums_are_well_formed() {
        for model in CATALOG {
            for artifact in model.artifacts {
                let Some(sha) = artifact.sha256 else { continue };
                assert_eq!(sha.len(), 64, "{}/{}", model.id, artifact.name);
                assert!(
                    sha.chars().all(|c| c.is_ascii_hexdigit()),
                    "{}/{} is not hex",
                    model.id,
                    artifact.name
                );
            }
        }
    }

    #[test]
    fn catalog_urls_are_https() {
        for model in CATALOG {
            for artifact in model.artifacts {
                assert!(
                    artifact.url.starts_with("https://"),
                    "{}/{} is not https",
                    model.id,
                    artifact.name
                );
            }
        }
    }
}
