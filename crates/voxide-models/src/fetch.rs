//! Fetching bytes, verifying them, and landing them on disk atomically.

use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("download failed for {url}: {message}")]
    Http { url: String, message: String },

    #[error(
        "checksum mismatch for {name}\n  expected sha256 {expected}\n  actual   sha256 {actual}\n\
         The file was discarded. This means the download was corrupted, or the upstream \
         artifact changed and the pin in voxide is stale."
    )]
    Checksum {
        name: String,
        expected: String,
        actual: String,
    },
}

type Result<T> = std::result::Result<T, FetchError>;

/// Reports download progress. Implemented by the CLI to draw a progress line.
pub trait Progress: Send {
    /// `total` is `None` when the server does not report a content length.
    fn start(&mut self, name: &str, total: Option<u64>);
    fn advance(&mut self, bytes: u64);
    fn finish(&mut self, name: &str);
}

/// Discards progress. Used by tests and non-interactive runs.
pub struct SilentProgress;

impl Progress for SilentProgress {
    fn start(&mut self, _name: &str, _total: Option<u64>) {}
    fn advance(&mut self, _bytes: u64) {}
    fn finish(&mut self, _name: &str) {}
}

/// Retrieves a URL into a local file.
///
/// A trait rather than a bare function so the install logic — checksum
/// verification, atomic replacement, archive extraction, skip-if-present —
/// can be tested end to end without network access. Only the HTTP
/// implementation is untestable offline, and it is deliberately thin.
pub trait Fetcher {
    fn fetch(&self, url: &str, dest: &Path, progress: &mut dyn Progress) -> Result<()>;
}

/// Downloads over HTTPS.
pub struct HttpFetcher {
    agent: ureq::Agent,
}

impl HttpFetcher {
    pub fn new() -> Self {
        Self {
            agent: ureq::Agent::new_with_defaults(),
        }
    }
}

impl Default for HttpFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Fetcher for HttpFetcher {
    fn fetch(&self, url: &str, dest: &Path, progress: &mut dyn Progress) -> Result<()> {
        let response = self.agent.get(url).call().map_err(|e| FetchError::Http {
            url: url.to_owned(),
            message: e.to_string(),
        })?;

        let total = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());

        let name = dest
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| url.to_owned());

        progress.start(&name, total);

        let mut reader = response.into_body().into_reader();
        let mut writer = create(dest)?;
        let mut buf = vec![0u8; 1 << 16];

        loop {
            let n = reader.read(&mut buf).map_err(|source| FetchError::Io {
                path: dest.to_path_buf(),
                source,
            })?;
            if n == 0 {
                break;
            }
            writer
                .write_all(&buf[..n])
                .map_err(|source| FetchError::Io {
                    path: dest.to_path_buf(),
                    source,
                })?;
            progress.advance(n as u64);
        }

        writer.flush().map_err(|source| FetchError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
        progress.finish(&name);
        Ok(())
    }
}

fn create(path: &Path) -> Result<std::fs::File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| FetchError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    std::fs::File::create(path).map_err(|source| FetchError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Copies a local file. Used by tests in place of [`HttpFetcher`].
///
/// Treats the URL's final path segment as a filename inside `root`.
pub struct LocalFetcher {
    pub root: PathBuf,
}

impl Fetcher for LocalFetcher {
    fn fetch(&self, url: &str, dest: &Path, progress: &mut dyn Progress) -> Result<()> {
        let name = url.rsplit('/').next().unwrap_or(url);
        let src = self.root.join(name);

        let bytes = std::fs::read(&src).map_err(|source| FetchError::Io {
            path: src.clone(),
            source,
        })?;

        progress.start(name, Some(bytes.len() as u64));
        let mut writer = create(dest)?;
        writer.write_all(&bytes).map_err(|source| FetchError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
        progress.advance(bytes.len() as u64);
        progress.finish(name);
        Ok(())
    }
}

/// Hex-encoded SHA-256 of a file, streamed rather than read whole.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|source| FetchError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf).map_err(|source| FetchError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Fails unless `path` hashes to `expected`.
pub fn verify(path: &Path, name: &str, expected: &str) -> Result<()> {
    let actual = sha256_file(path)?;
    if actual.eq_ignore_ascii_case(expected) {
        return Ok(());
    }
    Err(FetchError::Checksum {
        name: name.to_owned(),
        expected: expected.to_owned(),
        actual,
    })
}

/// Downloads to a sibling `.part` file, verifies, then renames into place.
///
/// The rename is the point. Writing straight to the destination leaves a
/// truncated file behind if the process dies mid-download, and the next run
/// sees a file that exists and assumes it is complete.
pub fn fetch_verified(
    fetcher: &dyn Fetcher,
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    progress: &mut dyn Progress,
) -> Result<()> {
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| dest.display().to_string());

    let part = dest.with_extension(format!(
        "{}part",
        dest.extension()
            .map(|e| format!("{}.", e.to_string_lossy()))
            .unwrap_or_default()
    ));

    fetcher.fetch(url, &part, progress)?;

    if let Some(expected) = expected_sha256
        && let Err(e) = verify(&part, &name, expected)
    {
        // Never leave an unverified artifact where a later run might trust it.
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }

    std::fs::rename(&part, dest).map_err(|source| FetchError::Io {
        path: dest.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(dir: &Path, name: &str, contents: &[u8]) -> String {
        std::fs::write(dir.join(name), contents).unwrap();
        format!("https://example.invalid/{name}")
    }

    /// sha256 of "hello"
    const HELLO_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn hashes_a_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("f");
        std::fs::write(&p, b"hello").unwrap();
        assert_eq!(sha256_file(&p).unwrap(), HELLO_SHA);
    }

    #[test]
    fn verify_accepts_a_matching_hash_ignoring_case() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("f");
        std::fs::write(&p, b"hello").unwrap();
        assert!(verify(&p, "f", HELLO_SHA).is_ok());
        assert!(verify(&p, "f", &HELLO_SHA.to_uppercase()).is_ok());
    }

    #[test]
    fn verify_rejects_a_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("f");
        std::fs::write(&p, b"goodbye").unwrap();
        let err = verify(&p, "f", HELLO_SHA).unwrap_err();
        assert!(matches!(err, FetchError::Checksum { .. }));
        // The message must name both hashes so a stale pin is diagnosable.
        assert!(err.to_string().contains(HELLO_SHA), "got: {err}");
    }

    #[test]
    fn fetch_verified_lands_the_file() {
        let src = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let url = fixture(src.path(), "model.bin", b"hello");

        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };
        let dest = out.path().join("model.bin");

        fetch_verified(&fetcher, &url, &dest, Some(HELLO_SHA), &mut SilentProgress).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"hello");
    }

    /// A corrupt download must not be left anywhere a later run could trust.
    #[test]
    fn a_bad_checksum_leaves_no_files_behind() {
        let src = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let url = fixture(src.path(), "model.bin", b"corrupted");

        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };
        let dest = out.path().join("model.bin");

        let err = fetch_verified(&fetcher, &url, &dest, Some(HELLO_SHA), &mut SilentProgress)
            .unwrap_err();
        assert!(matches!(err, FetchError::Checksum { .. }));

        assert!(!dest.exists(), "destination should not exist");
        let leftovers: Vec<_> = std::fs::read_dir(out.path())
            .unwrap()
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name())
            .collect();
        assert!(leftovers.is_empty(), "partial files left: {leftovers:?}");
    }

    #[test]
    fn an_unpinned_file_is_accepted_without_verification() {
        let src = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let url = fixture(src.path(), "config.json", b"{}");

        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };
        let dest = out.path().join("config.json");

        fetch_verified(&fetcher, &url, &dest, None, &mut SilentProgress).unwrap();
        assert!(dest.exists());
    }

    #[test]
    fn creates_missing_parent_directories() {
        let src = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let url = fixture(src.path(), "m.bin", b"hello");

        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };
        let dest = out.path().join("deep/nested/m.bin");

        fetch_verified(&fetcher, &url, &dest, Some(HELLO_SHA), &mut SilentProgress).unwrap();
        assert!(dest.exists());
    }

    #[test]
    fn a_missing_source_is_an_io_error_naming_the_path() {
        let src = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let fetcher = LocalFetcher {
            root: src.path().to_path_buf(),
        };
        let err = fetch_verified(
            &fetcher,
            "https://example.invalid/absent.bin",
            &out.path().join("absent.bin"),
            None,
            &mut SilentProgress,
        )
        .unwrap_err();
        assert!(matches!(err, FetchError::Io { .. }));
        assert!(err.to_string().contains("absent.bin"), "got: {err}");
    }
}
