//! ferret — a local file search engine with a hand-built inverted index.
//!
//! The index is a small log-structured merge tree:
//!
//! ```text
//!   files ──► extract ──► tokenize ──► memtable (budgeted, in RAM)
//!                                          │  flush when full
//!                                          ▼
//!                              level 0 segments (immutable)
//!                                          │  4 segments at a level
//!                                          ▼
//!                              level 1 segments ──► level 2 ...
//! ```
//!
//! Everything under `segment`, `codec`, `memtable`, `merge` and `query` is the
//! index proper. `store` is the only module that knows SQLite exists, and it
//! uses it as a B-tree and a page cache — there is no FTS5 in this project.

pub mod codec;
pub mod extract;
pub mod index;
pub mod memtable;
pub mod merge;
pub mod query;
pub mod segment;
pub mod serve;
pub mod snippet;
pub mod store;
pub mod tokenize;
pub mod watch;

use std::fmt;

/// One error type for the whole crate. A search tool has three failure modes
/// that a user can act on — bad path, bad index, bad query — and everything
/// else is context attached to a message.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Db(rusqlite::Error),
    Message(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Db(e) => write!(f, "index error: {e}"),
            Error::Message(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Db(e)
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Message(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Message(s.to_string())
    }
}

/// Format a byte count for humans.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Thousands separators, because index statistics are unreadable without them.
pub fn commas(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_bytes() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MB");
    }

    #[test]
    fn formats_thousands() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1000), "1,000");
        assert_eq!(commas(1234567), "1,234,567");
    }
}
