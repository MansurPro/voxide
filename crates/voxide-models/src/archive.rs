//! Zip extraction.
//!
//! Speech models ship as zip archives, so voxide has to unpack one. An archive
//! is untrusted input even when it comes from a checksummed download: the
//! checksum proves the bytes are the ones that were pinned, not that their
//! contents are benign.

use std::path::{Component, Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not read archive {path}: {source}")]
    Zip {
        path: PathBuf,
        #[source]
        source: Box<zip::result::ZipError>,
    },

    #[error("archive entry {entry:?} would escape the extraction directory; refusing to unpack it")]
    UnsafePath { entry: String },
}

type Result<T> = std::result::Result<T, ArchiveError>;

/// Resolves an archive entry name to a path inside `root`.
///
/// Rejects absolute paths and any `..` component. Without this an archive can
/// name an entry `../../.ssh/authorized_keys` and write outside the directory
/// the user chose — the "zip slip" vulnerability. Prefix components and
/// Windows drive letters are stripped for the same reason.
pub fn safe_join(root: &Path, entry: &str) -> Result<PathBuf> {
    // Normalise separators: zip always uses `/`, even in archives built on
    // Windows, but a hostile archive can contain either.
    let normalised = entry.replace('\\', "/");
    let candidate = Path::new(&normalised);

    let mut out = root.to_path_buf();
    for component in candidate.components() {
        match component {
            Component::Normal(part) => out.push(part),
            // A leading "./" is harmless.
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ArchiveError::UnsafePath {
                    entry: entry.to_owned(),
                });
            }
        }
    }

    Ok(out)
}

/// Unpacks `archive` into `dest`.
///
/// When the archive has a single top-level directory — as Vosk models do —
/// that wrapper is stripped, so `dest` holds the model rather than
/// `dest/vosk-model-small-en-us-0.15/`.
pub fn extract_zip(archive: &Path, dest: &Path) -> Result<()> {
    let file = std::fs::File::open(archive).map_err(|source| ArchiveError::Io {
        path: archive.to_path_buf(),
        source,
    })?;

    let mut zip = zip::ZipArchive::new(file).map_err(|source| ArchiveError::Zip {
        path: archive.to_path_buf(),
        source: Box::new(source),
    })?;

    let strip = single_root_prefix(&mut zip);

    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|source| ArchiveError::Zip {
            path: archive.to_path_buf(),
            source: Box::new(source),
        })?;

        let raw_name = entry.name().to_owned();
        let relative = match &strip {
            Some(prefix) => match raw_name.strip_prefix(prefix.as_str()) {
                Some(rest) => rest.to_owned(),
                None => raw_name.clone(),
            },
            None => raw_name.clone(),
        };

        if relative.is_empty() {
            continue;
        }

        let target = safe_join(dest, &relative)?;

        if entry.is_dir() {
            std::fs::create_dir_all(&target).map_err(|source| ArchiveError::Io {
                path: target.clone(),
                source,
            })?;
            continue;
        }

        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|source| ArchiveError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }

        let mut out = std::fs::File::create(&target).map_err(|source| ArchiveError::Io {
            path: target.clone(),
            source,
        })?;
        std::io::copy(&mut entry, &mut out).map_err(|source| ArchiveError::Io {
            path: target.clone(),
            source,
        })?;
    }

    Ok(())
}

/// The common top-level directory of every entry, if there is exactly one.
fn single_root_prefix<R: std::io::Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
) -> Option<String> {
    let mut root: Option<String> = None;

    for i in 0..zip.len() {
        let name = zip.by_index(i).ok()?.name().to_owned();
        let first = name.split('/').next()?.to_owned();
        // An entry at the archive root means there is no single wrapper.
        if first.is_empty() || !name.contains('/') && !name.ends_with('/') {
            return None;
        }
        match &root {
            Some(existing) if *existing != first => return None,
            Some(_) => {}
            None => root = Some(first),
        }
    }

    root.map(|r| format!("{r}/"))
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

    #[test]
    fn safe_join_accepts_a_normal_path() {
        let root = Path::new("/models");
        assert_eq!(
            safe_join(root, "am/final.mdl").unwrap(),
            Path::new("/models/am/final.mdl")
        );
    }

    #[test]
    fn safe_join_ignores_a_leading_dot() {
        assert_eq!(
            safe_join(Path::new("/models"), "./conf/x").unwrap(),
            Path::new("/models/conf/x")
        );
    }

    /// Zip slip. An archive must not be able to write outside the directory
    /// the user pointed at.
    #[test]
    fn safe_join_rejects_parent_traversal() {
        for hostile in [
            "../escaped",
            "am/../../escaped",
            "../../.ssh/authorized_keys",
            "..\\..\\escaped",
        ] {
            assert!(
                matches!(
                    safe_join(Path::new("/models"), hostile),
                    Err(ArchiveError::UnsafePath { .. })
                ),
                "accepted hostile entry {hostile:?}"
            );
        }
    }

    #[test]
    fn safe_join_rejects_absolute_paths() {
        assert!(matches!(
            safe_join(Path::new("/models"), "/etc/passwd"),
            Err(ArchiveError::UnsafePath { .. })
        ));
    }

    #[test]
    fn extracts_a_flat_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("a.zip");
        build_zip(&zip_path, &[("one.txt", b"1"), ("two.txt", b"2")]);

        let dest = tmp.path().join("out");
        extract_zip(&zip_path, &dest).unwrap();

        assert_eq!(std::fs::read(dest.join("one.txt")).unwrap(), b"1");
        assert_eq!(std::fs::read(dest.join("two.txt")).unwrap(), b"2");
    }

    /// Vosk archives wrap everything in a versioned directory; the caller
    /// asked for a model directory, not a directory containing one.
    #[test]
    fn strips_a_single_wrapper_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("model.zip");
        build_zip(
            &zip_path,
            &[
                ("vosk-model-small-en-us-0.15/am/final.mdl", b"m"),
                ("vosk-model-small-en-us-0.15/conf/mfcc.conf", b"c"),
            ],
        );

        let dest = tmp.path().join("out");
        extract_zip(&zip_path, &dest).unwrap();

        assert!(
            dest.join("am/final.mdl").is_file(),
            "wrapper was not stripped"
        );
        assert!(dest.join("conf/mfcc.conf").is_file());
        assert!(!dest.join("vosk-model-small-en-us-0.15").exists());
    }

    #[test]
    fn keeps_structure_when_there_are_two_top_level_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("two.zip");
        build_zip(&zip_path, &[("a/one.txt", b"1"), ("b/two.txt", b"2")]);

        let dest = tmp.path().join("out");
        extract_zip(&zip_path, &dest).unwrap();

        assert!(dest.join("a/one.txt").is_file());
        assert!(dest.join("b/two.txt").is_file());
    }

    #[test]
    fn extraction_refuses_a_traversing_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("evil.zip");
        // Two top-level names, so the wrapper-stripping path does not apply.
        build_zip(
            &zip_path,
            &[("safe/ok.txt", b"1"), ("../escaped.txt", b"x")],
        );

        let dest = tmp.path().join("out");
        let err = extract_zip(&zip_path, &dest).unwrap_err();
        assert!(
            matches!(err, ArchiveError::UnsafePath { .. }),
            "got {err:?}"
        );
        assert!(!tmp.path().join("escaped.txt").exists(), "escaped the dest");
    }

    #[test]
    fn a_corrupt_archive_is_reported_with_its_path() {
        let tmp = tempfile::tempdir().unwrap();
        let zip_path = tmp.path().join("broken.zip");
        std::fs::write(&zip_path, b"not a zip at all").unwrap();

        let err = extract_zip(&zip_path, &tmp.path().join("out")).unwrap_err();
        assert!(err.to_string().contains("broken.zip"), "got: {err}");
    }
}
