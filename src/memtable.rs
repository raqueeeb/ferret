//! The in-memory index.
//!
//! Documents accumulate here in already-encoded form. When the estimated
//! footprint crosses a budget the whole thing is sorted and written out as one
//! immutable segment, then dropped. This is the LSM shape: writes are
//! sequential appends, reads fan out over segments, and merges keep the fan-out
//! bounded.
//!
//! The important property is that peak memory is a *configured constant*, not
//! a function of corpus size. Indexing 4 million files with a 64MB budget uses
//! 64MB, plus whatever one document's text needs.
//!
//! Postings are encoded as documents arrive rather than held as
//! `Vec<(doc, Vec<pos>)>`. A `Vec<u32>` of positions costs 4 bytes per
//! occurrence plus 24 bytes of header; the varint encoding costs about 1.4
//! bytes per occurrence and no header. That difference is roughly 3x more
//! documents per flush, which means fewer segments and less merging later.

use std::collections::HashMap;

use crate::codec::encode_norm;
use crate::segment::{PostingsBuilder, SegmentWriter};
use crate::store::{SegmentInfo, Store};
use crate::Result;

/// Charged per term for the `HashMap` slot, the `String` header, the
/// `PostingsBuilder` struct and the table's load-factor slack. `HashMap`
/// resizes at 87.5% occupancy, so a slot costs more than `size_of` suggests.
const ENTRY_OVERHEAD: usize = 96;

pub struct MemTable {
    terms: HashMap<String, PostingsBuilder>,
    /// Estimated heap bytes held by `terms`.
    bytes: usize,
    budget: usize,
    docs: i64,
    tokens: i64,
    last_doc: i64,
    /// Scratch reused across documents so the per-document map does not
    /// allocate a fresh table for every file.
    scratch: HashMap<String, Vec<u32>>,
}

impl MemTable {
    pub fn new(budget: usize) -> Self {
        MemTable {
            terms: HashMap::new(),
            bytes: 0,
            budget,
            docs: 0,
            tokens: 0,
            last_doc: 0,
            scratch: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.docs == 0
    }

    pub fn docs(&self) -> i64 {
        self.docs
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Ground truth for the accounting above: what the term map actually owns.
    /// Walked only by tests, since it is O(terms).
    pub fn allocated_bytes(&self) -> usize {
        self.terms
            .iter()
            .map(|(k, v)| k.len() + v.blob.capacity() + ENTRY_OVERHEAD)
            .sum()
    }

    pub fn should_flush(&self) -> bool {
        self.bytes >= self.budget
    }

    /// Index one document's text. `doc` must be larger than any id added
    /// before, which the catalogue guarantees by never reusing ids.
    ///
    /// Returns the token count.
    pub fn add_document(&mut self, doc: i64, text: &str) -> u32 {
        assert!(doc > self.last_doc, "document ids must increase");

        // Pass 1: group positions by term for this document only.
        for v in self.scratch.values_mut() {
            v.clear();
        }
        let mut length = 0u32;
        let scratch = &mut self.scratch;
        crate::tokenize::tokenize(text, |term, pos| {
            length = length.max(pos + 1);
            match scratch.get_mut(term) {
                Some(v) => {
                    // Sub-token expansion can emit the same term twice at one
                    // position (`foo_foo`). Positions must stay strictly
                    // ascending for the gap encoding, so collapse duplicates.
                    if v.last() != Some(&pos) {
                        v.push(pos);
                    }
                }
                None => {
                    scratch.insert(term.to_string(), vec![pos]);
                }
            }
        });

        // Pass 2: append to the global postings, charging the budget.
        let norm = encode_norm(length);
        let mut added = 0usize;
        for (term, positions) in self.scratch.iter() {
            if positions.is_empty() {
                continue;
            }
            // Charge *capacity*, not length. A `Vec` doubles when it grows,
            // so a postings list holding 9MB of data can own 16MB of heap.
            // Budgeting against length was the difference between a stated
            // 64MB budget and 229MB of measured resident memory.
            match self.terms.get_mut(term) {
                Some(builder) => {
                    let before = builder.blob.capacity();
                    builder.push(doc, norm, positions);
                    added += builder.blob.capacity() - before;
                }
                None => {
                    let mut builder = PostingsBuilder::new();
                    builder.push(doc, norm, positions);
                    added += builder.blob.capacity() + term.len() + ENTRY_OVERHEAD;
                    self.terms.insert(term.clone(), builder);
                }
            }
        }

        self.bytes += added;
        self.docs += 1;
        self.tokens += length as i64;
        self.last_doc = doc;
        length
    }

    /// Write everything out as one segment at level 0 and reset.
    ///
    /// Terms are sorted before insertion. SQLite would accept them in any
    /// order, but a `WITHOUT ROWID` table is a B-tree keyed by term, and
    /// inserting in key order turns page splits into appends. On a 200k-term
    /// flush this is worth roughly 2x.
    pub fn flush(&mut self, store: &Store) -> Result<Option<SegmentInfo>> {
        if self.terms.is_empty() {
            return Ok(None);
        }
        let seg_id = store.next_segment_id()?;
        let mut keys: Vec<&String> = self.terms.keys().collect();
        keys.sort_unstable();

        let mut writer = SegmentWriter::begin(&store.conn, seg_id, 0)?;
        writer.set_docs(self.docs);
        // Inside the segment's transaction: the watermark and the postings it
        // describes commit atomically. See Store::recover.
        store.set_durable_doc(self.last_doc)?;
        for key in keys {
            let builder = &self.terms[key];
            if let Err(e) = writer.add_term(key, builder.df, builder.ttf, &builder.blob) {
                writer.rollback()?;
                return Err(e);
            }
        }
        let info = writer.commit(store)?;

        self.terms = HashMap::new();
        self.scratch = HashMap::new();
        self.bytes = 0;
        self.docs = 0;
        self.tokens = 0;
        Ok(Some(info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounting_tracks_real_allocation() {
        // The budget must not undercount, or the flush trigger fires late and
        // the process uses several times the memory it was asked to.
        let mut m = MemTable::new(1 << 30);
        for i in 1..500 {
            m.add_document(i, "alpha beta gamma delta epsilon zeta eta theta iota kappa");
        }
        let charged = m.bytes();
        let actual = m.allocated_bytes();
        assert!(
            charged >= actual,
            "budget undercounts real allocation: charged {charged}, allocated {actual}"
        );
        // ...and must not overcount so wildly that flushes happen far too
        // early, which would fragment the index into tiny segments.
        assert!(charged < actual * 3, "charged {charged} vs allocated {actual}");
    }

    #[test]
    fn budget_grows_with_content() {
        let mut m = MemTable::new(1 << 20);
        m.add_document(1, "alpha beta gamma");
        let after_one = m.bytes();
        m.add_document(2, "delta epsilon zeta eta theta");
        assert!(m.bytes() > after_one);
        assert_eq!(m.docs(), 2);
    }

    #[test]
    fn flush_is_requested_once_over_budget() {
        let mut m = MemTable::new(256);
        assert!(!m.should_flush());
        for i in 1..200 {
            m.add_document(i, "some words that take up a little space each time");
        }
        assert!(m.should_flush());
    }

    #[test]
    fn repeated_subtoken_positions_collapse() {
        // `foo_foo` emits "foo" twice at position 0; the encoder requires
        // strictly ascending positions.
        let mut m = MemTable::new(1 << 20);
        let len = m.add_document(1, "foo_foo");
        assert_eq!(len, 1);
    }
}
