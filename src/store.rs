//! Storage layer.
//!
//! SQLite is used as a *page store and B-tree*, not as a search engine. There
//! is no FTS5 anywhere in this project; the `terms` table holds opaque blobs
//! that this crate encodes and decodes itself.
//!
//! The one schema decision that matters is the primary key on `terms`:
//!
//! ```sql
//! PRIMARY KEY (term, seg) WITHOUT ROWID
//! ```
//!
//! `WITHOUT ROWID` makes the table a clustered B-tree keyed by the term
//! itself, so a postings lookup is one descent plus a short sequential run
//! over that term's segments — not one lookup per segment, and no secondary
//! index hop to fetch the blob. Ordering `term` before `seg` is what makes
//! query cost grow with *term count* rather than *segment count*, which is
//! what keeps latency flat as segments accumulate between merges.

use rusqlite::{Connection, OptionalExtension};
use std::path::{Path, PathBuf};

use crate::Result;

pub const SCHEMA_VERSION: i64 = 1;
/// Index lives in this directory under the indexed root.
pub const INDEX_DIR: &str = ".ferret";
pub const DB_FILE: &str = "index.db";

pub struct Store {
    pub conn: Connection,
    pub root: PathBuf,
    pub dir: PathBuf,
}

/// A document as recorded in the catalogue.
#[derive(Debug, Clone)]
pub struct DocRecord {
    pub id: i64,
    pub path: String,
    pub mtime: i64,
    pub size: i64,
    pub len: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct SegmentInfo {
    pub id: i64,
    pub level: i64,
    pub docs: i64,
    pub terms: i64,
    pub postings: i64,
    pub bytes: i64,
}

impl Store {
    /// Open (or create) the index rooted at `root`.
    pub fn open(root: &Path) -> Result<Store> {
        let root = root
            .canonicalize()
            .map_err(|e| format!("cannot resolve {}: {e}", root.display()))?;
        let dir = root.join(INDEX_DIR);
        std::fs::create_dir_all(&dir)?;
        let conn = Connection::open(dir.join(DB_FILE))?;
        Self::configure(&conn)?;
        let store = Store { conn, root, dir };
        store.migrate()?;
        Ok(store)
    }

    /// Open an existing index without creating one.
    pub fn open_existing(root: &Path) -> Result<Store> {
        let dir = root.join(INDEX_DIR);
        if !dir.join(DB_FILE).exists() {
            return Err(format!(
                "no index at {} — run `ferret index {}` first",
                root.display(),
                root.display()
            )
            .into());
        }
        Store::open(root)
    }

    /// A second connection to the same database, for streaming reads while the
    /// primary connection writes. Safe because of WAL: the reader sees a
    /// consistent snapshot taken when its transaction begins, and the writer
    /// never blocks it. This is what makes the merge memory-bounded — the
    /// merge reads through cursors instead of buffering segments.
    pub fn reader(&self) -> Result<Connection> {
        let conn = Connection::open(self.dir.join(DB_FILE))?;
        Self::configure(&conn)?;
        // A merge scans this connection once, front to back, and never looks
        // back. Caching those pages is pure waste: it doubled the process's
        // resident memory to hold data that will not be read again. 4MB is
        // enough for read-ahead on the B-tree spine.
        conn.pragma_update(None, "cache_size", -4096)?;
        conn.pragma_update(None, "mmap_size", 0i64)?;
        Ok(conn)
    }

    fn configure(conn: &Connection) -> Result<()> {
        // WAL: concurrent readers during merges and during watch-mode writes.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        // NORMAL: an fsync per checkpoint rather than per commit. A crash can
        // lose the last few commits, which for a *derived* index costs a
        // re-scan of a handful of files, not data.
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        // Postings blobs are read whole; large pages mean fewer B-tree hops.
        conn.pragma_update(None, "page_size", 8192)?;
        // 64MB page cache, negative value means KiB.
        conn.pragma_update(None, "cache_size", -65536)?;
        // Memory-map 1GB of the database: postings reads become memcpy from
        // the page cache instead of pread syscalls.
        conn.pragma_update(None, "mmap_size", 1_073_741_824i64)?;
        conn.pragma_update(None, "temp_store", "MEMORY")?;
        Ok(())
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Catalogue of indexed files. `len` is the token count, needed for
            -- BM25's average document length. A row is never updated in place
            -- on re-index: the old row is tombstoned and a new id is issued,
            -- so a query running concurrently keeps a coherent view.
            CREATE TABLE IF NOT EXISTS docs (
                id         INTEGER PRIMARY KEY,
                path       TEXT    NOT NULL,
                mtime      INTEGER NOT NULL,
                size       INTEGER NOT NULL,
                len        INTEGER NOT NULL,
                deleted    INTEGER NOT NULL DEFAULT 0,
                indexed_at INTEGER NOT NULL
            );

            -- Partial index: only live rows are unique by path, and only
            -- tombstones are listed in the deleted index. The deleted set is
            -- loaded on every query, so keeping it small and separately
            -- indexed matters.
            CREATE UNIQUE INDEX IF NOT EXISTS docs_live_path
                ON docs(path) WHERE deleted = 0;
            CREATE INDEX IF NOT EXISTS docs_tombstones
                ON docs(id) WHERE deleted = 1;

            CREATE TABLE IF NOT EXISTS segments (
                id         INTEGER PRIMARY KEY,
                level      INTEGER NOT NULL,
                docs       INTEGER NOT NULL,
                terms      INTEGER NOT NULL,
                postings   INTEGER NOT NULL,
                bytes      INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );

            -- The inverted index. `postings` is an opaque blob in this crate's
            -- own format: see codec.rs and segment.rs.
            CREATE TABLE IF NOT EXISTS terms (
                term     TEXT    NOT NULL,
                seg      INTEGER NOT NULL,
                df       INTEGER NOT NULL,
                ttf      INTEGER NOT NULL,
                postings BLOB    NOT NULL,
                PRIMARY KEY (term, seg)
            ) WITHOUT ROWID;
            "#,
        )?;

        let version: Option<i64> = self.meta_i64("schema_version")?;
        match version {
            None => {
                self.set_meta("schema_version", &SCHEMA_VERSION.to_string())?;
                self.set_meta("root", &self.root.to_string_lossy())?;
                self.set_meta("doc_count", "0")?;
                self.set_meta("total_tokens", "0")?;
            }
            Some(v) if v != SCHEMA_VERSION => {
                return Err(format!(
                    "index schema v{v}, this build speaks v{SCHEMA_VERSION}; delete {} and reindex",
                    self.dir.display()
                )
                .into());
            }
            _ => {}
        }
        Ok(())
    }

    // -- meta -------------------------------------------------------------

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn meta_i64(&self, key: &str) -> Result<Option<i64>> {
        Ok(self.meta(key)?.and_then(|v| v.parse().ok()))
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )?;
        Ok(())
    }

    // -- corpus statistics -------------------------------------------------

    /// (live document count, total tokens). Maintained incrementally so BM25
    /// never has to aggregate over the catalogue.
    pub fn corpus_stats(&self) -> Result<(u64, u64)> {
        let n = self.meta_i64("doc_count")?.unwrap_or(0).max(0) as u64;
        let t = self.meta_i64("total_tokens")?.unwrap_or(0).max(0) as u64;
        Ok((n, t))
    }

    pub fn bump_corpus(&self, docs_delta: i64, tokens_delta: i64) -> Result<()> {
        let (n, t) = self.corpus_stats()?;
        let n = (n as i64 + docs_delta).max(0);
        let t = (t as i64 + tokens_delta).max(0);
        self.set_meta("doc_count", &n.to_string())?;
        self.set_meta("total_tokens", &t.to_string())?;
        Ok(())
    }

    pub fn avg_doc_len(&self) -> Result<f32> {
        let (n, t) = self.corpus_stats()?;
        Ok(if n == 0 {
            1.0
        } else {
            (t as f32 / n as f32).max(1.0)
        })
    }

    // -- documents ---------------------------------------------------------

    /// Live document for `path`, if any.
    pub fn live_doc(&self, path: &str) -> Result<Option<DocRecord>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, path, mtime, size, len FROM docs WHERE path = ?1 AND deleted = 0",
                [path],
                |r| {
                    Ok(DocRecord {
                        id: r.get(0)?,
                        path: r.get(1)?,
                        mtime: r.get(2)?,
                        size: r.get(3)?,
                        len: r.get(4)?,
                    })
                },
            )
            .optional()?)
    }

    pub fn doc_path(&self, id: i64) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT path FROM docs WHERE id = ?1", [id], |r| r.get(0))
            .optional()?)
    }

    /// Insert a new live document, returning its id.
    pub fn insert_doc(&self, path: &str, mtime: i64, size: i64, len: i64) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO docs(path, mtime, size, len, deleted, indexed_at)
             VALUES (?1, ?2, ?3, ?4, 0, ?5)",
            (path, mtime, size, len, now_secs()),
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Tombstone the live document at `path`. Returns its token count so the
    /// caller can keep the corpus totals right.
    pub fn tombstone(&self, path: &str) -> Result<Option<(i64, i64)>> {
        let existing = self.live_doc(path)?;
        if let Some(doc) = &existing {
            self.conn
                .execute("UPDATE docs SET deleted = 1 WHERE id = ?1", [doc.id])?;
            self.bump_corpus(-1, -doc.len)?;
        }
        Ok(existing.map(|d| (d.id, d.len)))
    }

    /// Delete catalogue rows for documents whose postings never reached a
    /// segment, so the next crawl re-indexes those files.
    ///
    /// This exists because the two halves of a write have different
    /// durability. The catalogue commits every few thousand documents; the
    /// postings for those documents sit in the memtable until it fills. Kill
    /// the process in between and the catalogue claims a file is indexed while
    /// the inverted index has never heard of it — and because the crawl skips
    /// files whose size and mtime match the catalogue, that file stays
    /// unsearchable forever.
    ///
    /// Every flush records the highest document id it made durable. Anything
    /// above that watermark is a lie the catalogue is telling.
    ///
    /// Those rows are *tombstoned*, not deleted. Deleting them looked right
    /// and was wrong: an interrupted run can leave a partially written segment
    /// whose postings point at those ids, and with the catalogue rows gone
    /// those postings become orphans that no query can resolve and no merge
    /// knows to drop. Tombstoning routes them through the machinery that
    /// already exists for deleted files — filtered at query time, discarded at
    /// the next merge.
    pub fn recover(&self) -> Result<u64> {
        let durable = self.meta_i64("durable_doc")?.unwrap_or(0);
        let mut stmt = self
            .conn
            .prepare("SELECT id, len, deleted FROM docs WHERE id > ?1")?;
        let rows = stmt.query_map([durable], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })?;

        let mut lost = 0u64;
        let mut tokens = 0i64;
        let mut docs = 0i64;
        for row in rows {
            let (_, len, deleted) = row?;
            if deleted == 0 {
                lost += 1;
                docs += 1;
                tokens += len;
            }
        }
        if lost > 0 {
            self.conn.execute(
                "UPDATE docs SET deleted = 1 WHERE id > ?1 AND deleted = 0",
                [durable],
            )?;
            self.bump_corpus(-docs, -tokens)?;
        }
        Ok(lost)
    }

    /// Record that every document up to `id` now has postings on disk.
    /// Called inside the flush transaction so the watermark and the segment
    /// become durable together or not at all.
    pub fn set_durable_doc(&self, id: i64) -> Result<()> {
        self.set_meta("durable_doc", &id.to_string())
    }

    /// Every tombstoned document id.
    ///
    /// This is the deletion filter applied at query time. It is loaded per
    /// search, which is only sane because merges purge tombstones: the set is
    /// bounded by churn since the last merge, not by corpus size.
    pub fn tombstoned_ids(&self) -> Result<std::collections::HashSet<i64>> {
        let mut stmt = self.conn.prepare("SELECT id FROM docs WHERE deleted = 1")?;
        let ids = stmt.query_map([], |r| r.get::<_, i64>(0))?;
        let mut set = std::collections::HashSet::new();
        for id in ids {
            set.insert(id?);
        }
        Ok(set)
    }

    /// Drop tombstone rows whose postings are gone. Called after a full merge.
    pub fn purge_tombstones(&self) -> Result<usize> {
        Ok(self
            .conn
            .execute("DELETE FROM docs WHERE deleted = 1", [])?)
    }

    /// Every live path, for detecting deletions during a rescan.
    pub fn live_paths(&self) -> Result<std::collections::HashMap<String, (i64, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, mtime, size FROM docs WHERE deleted = 0")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?),
            ))
        })?;
        let mut map = std::collections::HashMap::new();
        for row in rows {
            let (p, v) = row?;
            map.insert(p, v);
        }
        Ok(map)
    }

    // -- segments ----------------------------------------------------------

    pub fn segments(&self) -> Result<Vec<SegmentInfo>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, level, docs, terms, postings, bytes FROM segments ORDER BY id")?;
        let rows = stmt.query_map([], |r| {
            Ok(SegmentInfo {
                id: r.get(0)?,
                level: r.get(1)?,
                docs: r.get(2)?,
                terms: r.get(3)?,
                postings: r.get(4)?,
                bytes: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn record_segment(&self, info: &SegmentInfo) -> Result<()> {
        self.conn.execute(
            "INSERT INTO segments(id, level, docs, terms, postings, bytes, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            (
                info.id,
                info.level,
                info.docs,
                info.terms,
                info.postings,
                info.bytes,
                now_secs(),
            ),
        )?;
        Ok(())
    }

    pub fn next_segment_id(&self) -> Result<i64> {
        let max: Option<i64> = self
            .conn
            .query_row("SELECT MAX(id) FROM segments", [], |r| r.get(0))
            .optional()?
            .flatten();
        Ok(max.unwrap_or(0) + 1)
    }

    pub fn drop_segments(&self, ids: &[i64]) -> Result<()> {
        for id in ids {
            self.conn
                .execute("DELETE FROM terms WHERE seg = ?1", [id])?;
            self.conn
                .execute("DELETE FROM segments WHERE id = ?1", [id])?;
        }
        Ok(())
    }

    /// On-disk size of the index in bytes (database plus WAL).
    pub fn index_bytes(&self) -> u64 {
        let mut total = 0;
        for name in [DB_FILE, "index.db-wal", "index.db-shm"] {
            if let Ok(md) = std::fs::metadata(self.dir.join(name)) {
                total += md.len();
            }
        }
        total
    }
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
