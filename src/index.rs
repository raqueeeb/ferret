//! Crawling and indexing.
//!
//! Concurrency shape: **many extractors, one writer.**
//!
//! Extraction is the expensive half (reading files, inflating PDF streams,
//! validating UTF-8) and it is embarrassingly parallel. Index construction is
//! the cheap half and is inherently serial — document ids must increase
//! monotonically for the gap encoding, and the memtable is a single mutable
//! structure. So worker threads produce `(path, text)` over a bounded channel
//! and the main thread consumes it.
//!
//! The channel is bounded deliberately. An unbounded queue in front of a
//! slower consumer is just a memory leak with extra steps: on a corpus of
//! large files the extractors would run ahead and hold hundreds of megabytes
//! of decoded text. The bound turns that into backpressure.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::time::Instant;

use crate::extract;
use crate::memtable::MemTable;
use crate::merge;
use crate::store::Store;
use crate::Result;

/// Directories never worth indexing: build output, dependency caches, VCS
/// internals. Each is skipped without being descended into, which on a real
/// checkout is the difference between 3,000 files and 300,000.
pub const SKIP_DIRS: &[&str] = &[
    ".ferret", ".git", ".hg", ".svn", ".bzr", "node_modules", "target", "build", "dist",
    "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".gradle", ".idea", ".vscode", ".next", ".nuxt", ".cache", ".terraform", "Pods",
    "DerivedData", ".stack-work", "_build", "coverage", ".sass-cache",
];

pub struct Options {
    /// Memtable budget in bytes before a flush.
    pub budget: usize,
    /// Extractor threads.
    pub threads: usize,
    /// Index dotfiles and dot-directories.
    pub hidden: bool,
    /// Merge segments after indexing.
    pub auto_merge: bool,
    /// Extra directory names to skip.
    pub skip: Vec<String>,
    pub verbose: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            budget: 64 * 1024 * 1024,
            threads: std::thread::available_parallelism()
                .map(|n| n.get().min(8))
                .unwrap_or(4),
            hidden: false,
            auto_merge: true,
            skip: Vec::new(),
            verbose: false,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub seen: u64,
    pub indexed: u64,
    pub unchanged: u64,
    pub skipped: u64,
    pub removed: u64,
    pub errors: u64,
    pub bytes_read: u64,
    pub tokens: u64,
    pub flushes: u64,
    pub elapsed_ms: u128,
}

pub struct Indexer<'a> {
    store: &'a Store,
    mem: MemTable,
    pub stats: Stats,
    opts: Options,
}

impl<'a> Indexer<'a> {
    pub fn new(store: &'a Store, opts: Options) -> Self {
        Indexer { store, mem: MemTable::new(opts.budget), stats: Stats::default(), opts }
    }

    /// Full crawl of the index root: adds new files, re-indexes changed ones,
    /// tombstones files that have disappeared.
    pub fn crawl(&mut self) -> Result<()> {
        let start = Instant::now();
        let root = self.store.root.clone();

        // Discard any catalogue rows left behind by a crash before looking at
        // what is already indexed, or those files would be treated as
        // unchanged and never re-read.
        let lost = self.store.recover()?;
        if lost > 0 {
            eprintln!("  recovered from an interrupted run: {lost} files queued for re-indexing");
        }

        let known = self.store.live_paths()?;

        // Phase 1: walk. Cheap, serial, and needed in full before deletions
        // can be detected.
        let mut candidates: Vec<PathBuf> = Vec::new();
        let mut present: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Copied out so the filter closure does not borrow `self`, which the
        // loop below mutates.
        let hidden = self.opts.hidden;
        let extra_skips = self.opts.skip.clone();
        let walker = walkdir::WalkDir::new(&root)
            .follow_links(false)
            .into_iter()
            .filter_entry(move |e| keep_entry(e, hidden, &extra_skips));

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => {
                    self.stats.errors += 1;
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            self.stats.seen += 1;
            let rel = match entry.path().strip_prefix(&root) {
                Ok(r) => r.to_string_lossy().into_owned(),
                Err(_) => continue,
            };
            present.insert(rel.clone());

            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => {
                    self.stats.errors += 1;
                    continue;
                }
            };
            let mtime = mtime_of(&meta);
            let size = meta.len() as i64;

            // Incremental skip: a file is unchanged if size and mtime match.
            // Content hashing would catch mtime-preserving edits, but it costs
            // a full read of every file on every crawl, which defeats the
            // point. `--rehash` exists for the paranoid case.
            if let Some(&(known_mtime, known_size)) = known.get(&rel) {
                if known_mtime == mtime && known_size == size {
                    self.stats.unchanged += 1;
                    continue;
                }
            }
            if extract::classify(entry.path()) == extract::Kind::Skip {
                self.stats.skipped += 1;
                continue;
            }
            candidates.push(entry.path().to_path_buf());
        }

        // Phase 2: extract in parallel, index serially.
        self.index_batch(candidates)?;

        // Phase 3: tombstone anything that vanished.
        for path in known.keys() {
            if !present.contains(path) {
                self.store.tombstone(path)?;
                self.stats.removed += 1;
            }
        }

        self.stats.elapsed_ms = start.elapsed().as_millis();
        Ok(())
    }

    /// Extract `paths` across a thread pool and feed the memtable.
    pub fn index_batch(&mut self, paths: Vec<PathBuf>) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        let threads = self.opts.threads.max(1).min(paths.len());
        let paths = Arc::new(paths);
        let cursor = Arc::new(AtomicUsize::new(0));
        // Bound: a few documents in flight per worker. Enough to keep the
        // writer fed, small enough that peak memory stays predictable.
        let (tx, rx) = sync_channel::<(PathBuf, String, i64, i64)>(threads * 4);

        let mut handles = Vec::new();
        for _ in 0..threads {
            let paths = Arc::clone(&paths);
            let cursor = Arc::clone(&cursor);
            let tx = tx.clone();
            handles.push(std::thread::spawn(move || -> u64 {
                let mut errors = 0u64;
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= paths.len() {
                        break;
                    }
                    let path = &paths[i];
                    let meta = match std::fs::metadata(path) {
                        Ok(m) => m,
                        Err(_) => {
                            errors += 1;
                            continue;
                        }
                    };
                    match extract::extract_file(path) {
                        Ok(Some(text)) => {
                            let mtime = mtime_of(&meta);
                            if tx.send((path.clone(), text, mtime, meta.len() as i64)).is_err() {
                                break;
                            }
                        }
                        Ok(None) => {}
                        Err(_) => errors += 1,
                    }
                }
                errors
            }));
        }
        drop(tx);

        let root = self.store.root.clone();
        self.store.conn.execute_batch("BEGIN")?;
        let mut in_txn = 0usize;
        for (path, text, mtime, size) in rx {
            let rel = match path.strip_prefix(&root) {
                Ok(r) => r.to_string_lossy().into_owned(),
                Err(_) => path.to_string_lossy().into_owned(),
            };
            // Re-indexing is tombstone-then-insert, never update in place.
            // The old postings stay in their segments until a merge drops
            // them, so an in-flight query keeps a coherent view of the corpus.
            if let Some((_, old_len)) = self.store.tombstone(&rel)? {
                let _ = old_len;
            }
            let doc_id = self.store.insert_doc(&rel, mtime, size, 0)?;
            let len = self.mem.add_document(doc_id, &text);
            self.store
                .conn
                .execute("UPDATE docs SET len = ?1 WHERE id = ?2", (len as i64, doc_id))?;
            self.store.bump_corpus(1, len as i64)?;

            self.stats.indexed += 1;
            self.stats.bytes_read += size as u64;
            self.stats.tokens += len as u64;
            in_txn += 1;

            if self.opts.verbose && self.stats.indexed % 2000 == 0 {
                eprintln!(
                    "  {} files, memtable {:.1} MB",
                    self.stats.indexed,
                    self.mem.bytes() as f64 / 1e6
                );
            }

            // Commit the catalogue periodically so a crash loses seconds of
            // work, not the whole crawl.
            if in_txn >= 4096 {
                self.store.conn.execute_batch("COMMIT; BEGIN")?;
                in_txn = 0;
            }
            if self.mem.should_flush() {
                self.store.conn.execute_batch("COMMIT")?;
                self.flush()?;
                self.store.conn.execute_batch("BEGIN")?;
                in_txn = 0;
            }
        }
        self.store.conn.execute_batch("COMMIT")?;

        for h in handles {
            if let Ok(errors) = h.join() {
                self.stats.errors += errors;
            }
        }
        Ok(())
    }

    /// Index one file, used by watch mode.
    pub fn index_one(&mut self, path: &Path) -> Result<bool> {
        if extract::classify(path) == extract::Kind::Skip {
            return Ok(false);
        }
        let meta = std::fs::metadata(path)?;
        if !meta.is_file() {
            return Ok(false);
        }
        let text = match extract::extract_file(path)? {
            Some(t) => t,
            None => return Ok(false),
        };
        let rel = path
            .strip_prefix(&self.store.root)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string_lossy().into_owned());

        self.store.tombstone(&rel)?;
        let doc_id = self
            .store
            .insert_doc(&rel, mtime_of(&meta), meta.len() as i64, 0)?;
        let len = self.mem.add_document(doc_id, &text);
        self.store
            .conn
            .execute("UPDATE docs SET len = ?1 WHERE id = ?2", (len as i64, doc_id))?;
        self.store.bump_corpus(1, len as i64)?;
        self.stats.indexed += 1;
        self.stats.tokens += len as u64;
        Ok(true)
    }

    pub fn remove_one(&mut self, path: &Path) -> Result<bool> {
        let rel = path
            .strip_prefix(&self.store.root)
            .map(|r| r.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string_lossy().into_owned());
        let removed = self.store.tombstone(&rel)?.is_some();
        if removed {
            self.stats.removed += 1;
        }
        Ok(removed)
    }

    pub fn memtable_bytes(&self) -> usize {
        self.mem.bytes()
    }

    pub fn memtable_docs(&self) -> i64 {
        self.mem.docs()
    }

    pub fn pending(&self) -> bool {
        !self.mem.is_empty()
    }

    pub fn flush(&mut self) -> Result<()> {
        if let Some(info) = self.mem.flush(self.store)? {
            self.stats.flushes += 1;
            if self.opts.verbose {
                eprintln!(
                    "  flushed segment {} — {} terms, {} postings, {:.1} MB",
                    info.id,
                    info.terms,
                    info.postings,
                    info.bytes as f64 / 1e6
                );
            }
        }
        Ok(())
    }

    /// Flush, then merge if the level-0 segment count warrants it.
    pub fn finish(&mut self) -> Result<()> {
        self.flush()?;
        if self.opts.auto_merge {
            merge::maybe_merge(self.store, self.opts.verbose)?;
        }
        Ok(())
    }
}

/// Directory filter applied during the walk. Pruning happens here rather than
/// after the fact so `walkdir` never descends into `node_modules` at all.
fn keep_entry(e: &walkdir::DirEntry, hidden: bool, extra: &[String]) -> bool {
    let name = e.file_name().to_string_lossy();
    if e.file_type().is_dir() {
        if SKIP_DIRS.contains(&name.as_ref()) || extra.iter().any(|s| s == name.as_ref()) {
            return false;
        }
        if !hidden && name.starts_with('.') && name != "." {
            return false;
        }
    }
    true
}

fn mtime_of(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
