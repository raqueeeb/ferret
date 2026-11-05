//! Postings lists and immutable segments.
//!
//! # Postings format
//!
//! A postings list is the entire on-disk representation of one term inside one
//! segment. Documents appear in ascending id order:
//!
//! ```text
//! repeat for each document:
//!   varint  doc_gap      // first entry is the raw id, then differences
//!   varint  tf           // term frequency in this document
//!   u8      norm         // quantised document length (codec::encode_norm)
//!   varint  pos_gap * tf // first is the raw position, then differences
//! ```
//!
//! Two things are worth defending here.
//!
//! *Positions are stored inline rather than in a parallel list.* It costs a
//! scan-and-skip on term-only queries (cheap: varints are self-delimiting, so
//! skipping is a branchy byte walk with no decode). It buys phrase queries
//! with no second seek, and phrase queries are most of why anyone wants this
//! over `grep -r`.
//!
//! *The norm is inline too.* See codec.rs — this is what removes random access
//! to the document table from the scoring loop.

use rusqlite::Connection;

use crate::codec::{get_varint, put_varint, varint_len};
use crate::store::{SegmentInfo, Store};
use crate::Result;

/// Builds one term's postings list incrementally, document by document.
#[derive(Debug, Default)]
pub struct PostingsBuilder {
    pub blob: Vec<u8>,
    pub df: u32,
    pub ttf: u64,
    last_doc: i64,
}

impl PostingsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one document. `positions` must be sorted ascending and unique;
    /// callers get that for free from the tokenizer's position counter.
    ///
    /// Returns the number of bytes added, so the memtable can charge its
    /// budget without re-measuring.
    pub fn push(&mut self, doc: i64, norm: u8, positions: &[u32]) -> usize {
        debug_assert!(doc > self.last_doc, "documents must arrive in id order");
        let before = self.blob.len();

        put_varint(&mut self.blob, (doc - self.last_doc) as u64);
        put_varint(&mut self.blob, positions.len() as u64);
        self.blob.push(norm);
        let mut prev = 0u32;
        for (i, &p) in positions.iter().enumerate() {
            let gap = if i == 0 { p } else { p - prev };
            put_varint(&mut self.blob, gap as u64);
            prev = p;
        }

        self.last_doc = doc;
        self.df += 1;
        self.ttf += positions.len() as u64;
        self.blob.len() - before
    }

    /// Append a document whose positions are already encoded.
    ///
    /// Position gaps are relative to the previous position *within the same
    /// document*, so they are independent of the document id. That means a
    /// merge can memcpy the position bytes straight across instead of
    /// decoding and re-encoding them — which is most of the work in a merge,
    /// since positions dominate the byte count of a postings list.
    pub fn push_encoded(&mut self, doc: i64, tf: u32, norm: u8, positions: &[u8]) -> usize {
        debug_assert!(doc > self.last_doc, "documents must arrive in id order");
        let before = self.blob.len();
        put_varint(&mut self.blob, (doc - self.last_doc) as u64);
        put_varint(&mut self.blob, tf as u64);
        self.blob.push(norm);
        self.blob.extend_from_slice(positions);
        self.last_doc = doc;
        self.df += 1;
        self.ttf += tf as u64;
        self.blob.len() - before
    }

    /// Bytes a `push` would add, without performing it.
    pub fn cost(&self, doc: i64, positions: &[u32]) -> usize {
        let mut n =
            varint_len((doc - self.last_doc) as u64) + varint_len(positions.len() as u64) + 1;
        let mut prev = 0u32;
        for (i, &p) in positions.iter().enumerate() {
            n += varint_len(if i == 0 { p as u64 } else { (p - prev) as u64 });
            prev = p;
        }
        n
    }
}

/// One decoded document entry, positions not yet read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub doc: i64,
    pub tf: u32,
    pub norm: u8,
}

/// Sequential cursor over an encoded postings list.
pub struct PostingsReader<'a> {
    buf: &'a [u8],
    pos: usize,
    last_doc: i64,
    pending_positions: u32,
}

impl<'a> PostingsReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        PostingsReader {
            buf,
            pos: 0,
            last_doc: 0,
            pending_positions: 0,
        }
    }

    /// Advance to the next document. Any positions left unread from the
    /// previous entry are skipped automatically.
    pub fn next_entry(&mut self) -> Option<Entry> {
        if self.pending_positions > 0 {
            self.skip_positions();
        }
        if self.pos >= self.buf.len() {
            return None;
        }
        let gap = get_varint(self.buf, &mut self.pos)?;
        let tf = get_varint(self.buf, &mut self.pos)? as u32;
        if self.pos >= self.buf.len() {
            return None;
        }
        let norm = self.buf[self.pos];
        self.pos += 1;
        self.last_doc += gap as i64;
        self.pending_positions = tf;
        Some(Entry {
            doc: self.last_doc,
            tf,
            norm,
        })
    }

    /// Walk past the current entry's positions without decoding them. Varints
    /// are self-delimiting, so this is a scan for bytes with the high bit
    /// clear — no arithmetic, no branches on value.
    pub fn skip_positions(&mut self) {
        let mut remaining = self.pending_positions;
        while remaining > 0 && self.pos < self.buf.len() {
            if self.buf[self.pos] & 0x80 == 0 {
                remaining -= 1;
            }
            self.pos += 1;
        }
        self.pending_positions = 0;
    }

    /// Decode the current entry's positions into `out` (cleared first).
    pub fn positions(&mut self, out: &mut Vec<u32>) {
        out.clear();
        let mut prev = 0u32;
        for i in 0..self.pending_positions {
            match get_varint(self.buf, &mut self.pos) {
                Some(gap) => {
                    let p = if i == 0 {
                        gap as u32
                    } else {
                        prev + gap as u32
                    };
                    out.push(p);
                    prev = p;
                }
                None => break,
            }
        }
        self.pending_positions = 0;
    }

    /// Raw bytes of the current entry's positions, for merge-time copying.
    pub fn positions_raw(&mut self) -> &'a [u8] {
        let start = self.pos;
        self.skip_positions();
        &self.buf[start..self.pos]
    }
}

/// Writes a sorted stream of terms into a new segment.
///
/// The whole segment is one transaction: either the segment exists completely
/// or it does not exist at all, which is what lets a crash mid-flush leave a
/// consistent index behind.
pub struct SegmentWriter<'a> {
    conn: &'a Connection,
    pub id: i64,
    level: i64,
    terms: i64,
    postings: i64,
    bytes: i64,
    docs: i64,
}

impl<'a> SegmentWriter<'a> {
    pub fn begin(conn: &'a Connection, id: i64, level: i64) -> Result<Self> {
        conn.execute_batch("BEGIN IMMEDIATE")?;
        Ok(SegmentWriter {
            conn,
            id,
            level,
            terms: 0,
            postings: 0,
            bytes: 0,
            docs: 0,
        })
    }

    pub fn add_term(&mut self, term: &str, df: u32, ttf: u64, blob: &[u8]) -> Result<()> {
        self.conn.execute(
            "INSERT INTO terms(term, seg, df, ttf, postings) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![term, self.id, df as i64, ttf as i64, blob],
        )?;
        self.terms += 1;
        self.postings += df as i64;
        self.bytes += blob.len() as i64;
        Ok(())
    }

    pub fn set_docs(&mut self, docs: i64) {
        self.docs = docs;
    }

    pub fn commit(self, store: &Store) -> Result<SegmentInfo> {
        let info = SegmentInfo {
            id: self.id,
            level: self.level,
            docs: self.docs,
            terms: self.terms,
            postings: self.postings,
            bytes: self.bytes,
        };
        store.record_segment(&info)?;
        self.conn.execute_batch("COMMIT")?;
        Ok(info)
    }

    pub fn rollback(self) -> Result<()> {
        self.conn.execute_batch("ROLLBACK")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postings_roundtrip() {
        let mut b = PostingsBuilder::new();
        b.push(1, 40, &[0, 5, 9]);
        b.push(7, 55, &[3]);
        b.push(1000, 60, &[1, 2, 3, 400]);

        let mut r = PostingsReader::new(&b.blob);
        let mut positions = Vec::new();

        let e = r.next_entry().unwrap();
        assert_eq!((e.doc, e.tf, e.norm), (1, 3, 40));
        r.positions(&mut positions);
        assert_eq!(positions, vec![0, 5, 9]);

        let e = r.next_entry().unwrap();
        assert_eq!((e.doc, e.tf), (7, 1));

        let e = r.next_entry().unwrap();
        assert_eq!(e.doc, 1000);
        r.positions(&mut positions);
        assert_eq!(positions, vec![1, 2, 3, 400]);

        assert!(r.next_entry().is_none());
        assert_eq!(b.df, 3);
        assert_eq!(b.ttf, 8);
    }

    #[test]
    fn skipping_positions_lands_on_next_doc() {
        let mut b = PostingsBuilder::new();
        b.push(1, 10, &[0, 300, 70000]);
        b.push(2, 11, &[4]);
        let mut r = PostingsReader::new(&b.blob);
        assert_eq!(r.next_entry().unwrap().doc, 1);
        // deliberately do not read positions
        assert_eq!(r.next_entry().unwrap().doc, 2);
    }

    #[test]
    fn cost_matches_actual_bytes() {
        let mut b = PostingsBuilder::new();
        let positions = [0u32, 130, 1000];
        let predicted = b.cost(9, &positions);
        let actual = b.push(9, 3, &positions);
        assert_eq!(predicted, actual);
    }
}
