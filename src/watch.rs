//! Incremental re-indexing driven by the filesystem.
//!
//! `notify` sits on inotify (Linux), FSEvents (macOS) and
//! ReadDirectoryChangesW (Windows). Three things make the difference between a
//! demo and something you can leave running:
//!
//! 1. **Ignore your own writes.** The index lives under the watched root, so
//!    every commit fires events that would re-trigger indexing forever. Paths
//!    inside `.ferret/` are dropped before anything else looks at them.
//! 2. **Debounce.** A single "save" in an editor is often a write, a rename, a
//!    chmod and a delete of a swap file. A build touches thousands of files in
//!    a burst. Events are collected into a set and only acted on after the
//!    filesystem goes quiet, which collapses a burst into one pass.
//! 3. **Coalesce by path.** The debounce buffer is a set, so a file saved
//!    forty times while a build runs is indexed once.
//!
//! Edits become visible to searchers when the batch flushes, because the
//! memtable is not queryable in this build — see README, "what I would build
//! next".

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::index::{Indexer, Options, SKIP_DIRS};
use crate::merge;
use crate::store::{Store, INDEX_DIR};
use crate::Result;

/// Quiet period before a batch is processed.
pub const DEBOUNCE: Duration = Duration::from_millis(400);
/// Process anyway once a batch reaches this size, so a huge burst does not
/// starve the indexer.
pub const MAX_BATCH: usize = 2000;

pub fn run(store: &Store, opts: Options) -> Result<()> {
    let (tx, rx) = channel::<notify::Result<Event>>();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })
    .map_err(|e| format!("cannot start watcher: {e}"))?;

    watcher
        .watch(&store.root, RecursiveMode::Recursive)
        .map_err(|e| format!("cannot watch {}: {e}", store.root.display()))?;

    eprintln!("watching {} — ctrl-c to stop", store.root.display());

    let mut indexer = Indexer::new(store, opts);
    let mut pending: HashSet<PathBuf> = HashSet::new();
    let mut last_event = Instant::now();

    loop {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(Ok(event)) => {
                if !matches!(
                    event.kind,
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                ) {
                    continue;
                }
                for path in event.paths {
                    if interesting(&store.root, &path) {
                        pending.insert(path);
                        last_event = Instant::now();
                    }
                }
            }
            Ok(Err(e)) => eprintln!("watch error: {e}"),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }

        let quiet = last_event.elapsed() >= DEBOUNCE;
        if !pending.is_empty() && (quiet || pending.len() >= MAX_BATCH) {
            let batch: Vec<PathBuf> = pending.drain().collect();
            match apply(&mut indexer, store, &batch) {
                Ok((added, removed)) => {
                    if added + removed > 0 {
                        eprintln!(
                            "  {added} indexed, {removed} removed  ({} live docs)",
                            store.corpus_stats()?.0
                        );
                    }
                }
                Err(e) => eprintln!("  indexing failed: {e}"),
            }
        }
    }
    Ok(())
}

fn apply(indexer: &mut Indexer<'_>, store: &Store, batch: &[PathBuf]) -> Result<(usize, usize)> {
    let mut added = 0usize;
    let mut removed = 0usize;

    store.conn.execute_batch("BEGIN")?;
    for path in batch {
        let result = if path.is_file() {
            indexer.index_one(path).map(|ok| (ok as usize, 0))
        } else if path.exists() {
            // A directory appeared: whatever is inside it arrives as its own
            // events, so nothing to do here.
            Ok((0, 0))
        } else {
            indexer.remove_one(path).map(|ok| (0, ok as usize))
        };
        match result {
            Ok((a, r)) => {
                added += a;
                removed += r;
            }
            Err(e) => eprintln!("  {}: {e}", path.display()),
        }
    }
    store.conn.execute_batch("COMMIT")?;

    // Flush so the change is searchable, then let the tiered policy clean up
    // the small segments this produces.
    if indexer.pending() {
        indexer.flush()?;
    }
    merge::maybe_merge(store, false)?;
    Ok((added, removed))
}

/// Filter events before they reach the debounce set.
fn interesting(root: &Path, path: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    for component in relative.components() {
        let name = component.as_os_str().to_string_lossy();
        // Without this the index's own commits would retrigger indexing.
        if name == INDEX_DIR {
            return false;
        }
        if SKIP_DIRS.contains(&name.as_ref()) {
            return false;
        }
    }
    // Editor scratch files: vim swap, emacs autosave, common temp suffixes.
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if name.ends_with('~') || name.ends_with(".swp") || name.ends_with(".swx") || name.ends_with(".tmp")
    {
        return false;
    }
    if name.starts_with('#') && name.ends_with('#') {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_its_own_index_writes() {
        let root = Path::new("/corpus");
        assert!(!interesting(root, Path::new("/corpus/.ferret/index.db-wal")));
        assert!(!interesting(root, Path::new("/corpus/node_modules/x/a.js")));
        assert!(!interesting(root, Path::new("/corpus/notes.md.swp")));
        assert!(interesting(root, Path::new("/corpus/src/main.rs")));
    }
}
