//! Segment merging.
//!
//! # Why merge at all
//!
//! Every flush creates a segment, and every query has to consult every
//! segment that contains the term. Left alone, an index that has absorbed
//! 500 flushes answers each query with 500 B-tree probes. Merging trades
//! write amplification for read cost, and the tiered policy below bounds the
//! number of segments at roughly `MERGE_FACTOR * log_MF(N)`.
//!
//! # Why this merge is memory-bounded
//!
//! The naive implementation loads each input segment's terms into a map and
//! merges in RAM, which needs memory proportional to the *merged segment*,
//! i.e. eventually proportional to the whole corpus. That is the thing to
//! avoid.
//!
//! The trick is that the `terms` table is `WITHOUT ROWID` with primary key
//! `(term, seg)`, so a scan of it emits rows already grouped by term and
//! ordered by segment inside each group. One statement:
//!
//! ```sql
//! SELECT term, seg, df, ttf, postings FROM terms WHERE seg IN (...)
//! ORDER BY term, seg
//! ```
//!
//! is satisfied by a single sequential B-tree walk with no sort buffer. The
//! merge consumes it as a stream: gather the rows for one term, merge those
//! postings lists by document id, write one output row, drop everything, move
//! to the next term. Peak memory is *the largest single term's postings*, not
//! the segment and not the corpus.
//!
//! The read runs on a separate connection so it sees a WAL snapshot from
//! before the merge started, while the writer inserts the new segment into the
//! same table. Readers and the writer never block each other, and a crash
//! mid-merge leaves the input segments intact because the new segment is one
//! transaction and the deletion of the inputs is another.

use rusqlite::Connection;
use std::collections::HashSet;
use std::time::Instant;

use crate::segment::{PostingsBuilder, PostingsReader, SegmentWriter};
use crate::store::Store;
use crate::Result;

/// Segments per level before that level is compacted upward.
pub const MERGE_FACTOR: usize = 4;

#[derive(Debug, Default, Clone)]
pub struct MergeStats {
    pub merges: u64,
    pub segments_in: u64,
    pub terms_out: u64,
    pub postings_in: u64,
    pub postings_out: u64,
    pub bytes_out: u64,
    pub peak_term_bytes: usize,
    pub elapsed_ms: u128,
}

/// Run any merges the tiered policy calls for. Cheap when there is nothing to
/// do, so it is safe to call after every flush.
pub fn maybe_merge(store: &Store, verbose: bool) -> Result<MergeStats> {
    let mut stats = MergeStats::default();
    loop {
        let segments = store.segments()?;
        let mut by_level: std::collections::BTreeMap<i64, Vec<i64>> = Default::default();
        for s in &segments {
            by_level.entry(s.level).or_default().push(s.id);
        }
        let target = by_level
            .iter()
            .find(|(_, ids)| ids.len() >= MERGE_FACTOR)
            .map(|(level, ids)| (*level, ids.clone()));

        match target {
            Some((level, ids)) => {
                let s = merge_segments(store, &ids, level + 1, verbose)?;
                accumulate(&mut stats, &s);
            }
            None => break,
        }
    }
    Ok(stats)
}

/// Collapse every segment into one. This is the "optimise" operation: best
/// possible query latency and smallest index, paid for with a full rewrite.
pub fn merge_all(store: &Store, verbose: bool) -> Result<MergeStats> {
    let segments = store.segments()?;
    // One segment is still worth rewriting if it carries postings for deleted
    // documents — that rewrite is the only thing that reclaims their space and
    // shrinks the tombstone set every query has to consult.
    let dead = store.tombstoned_ids()?.len();
    if segments.is_empty() || (segments.len() < 2 && dead == 0) {
        return Ok(MergeStats::default());
    }
    let ids: Vec<i64> = segments.iter().map(|s| s.id).collect();
    let level = segments.iter().map(|s| s.level).max().unwrap_or(0) + 1;
    let mut stats = MergeStats::default();
    let s = merge_segments(store, &ids, level, verbose)?;
    accumulate(&mut stats, &s);

    // Every posting has been rewritten, so no segment references a tombstoned
    // document any more and the catalogue rows can go.
    let purged = store.purge_tombstones()?;
    if verbose && purged > 0 {
        eprintln!("  purged {purged} tombstones");
    }
    Ok(stats)
}

fn accumulate(total: &mut MergeStats, one: &MergeStats) {
    total.merges += one.merges;
    total.segments_in += one.segments_in;
    total.terms_out += one.terms_out;
    total.postings_in += one.postings_in;
    total.postings_out += one.postings_out;
    total.bytes_out += one.bytes_out;
    total.peak_term_bytes = total.peak_term_bytes.max(one.peak_term_bytes);
    total.elapsed_ms += one.elapsed_ms;
}

/// Merge `ids` into a single new segment at `out_level`.
pub fn merge_segments(
    store: &Store,
    ids: &[i64],
    out_level: i64,
    verbose: bool,
) -> Result<MergeStats> {
    let start = Instant::now();
    let mut stats = MergeStats { merges: 1, segments_in: ids.len() as u64, ..Default::default() };
    if ids.is_empty() {
        return Ok(MergeStats::default());
    }

    let dead = store.tombstoned_ids()?;
    let docs_in: i64 = store
        .segments()?
        .iter()
        .filter(|s| ids.contains(&s.id))
        .map(|s| s.docs)
        .sum();

    let reader = store.reader()?;
    let new_id = store.next_segment_id()?;

    // Scope the reader statement so its snapshot is released before the input
    // segments are deleted.
    {
        let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT term, seg, postings FROM terms WHERE seg IN ({placeholders}) ORDER BY term, seg"
        );
        let mut stmt = reader.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
        let mut rows = stmt.query(params.as_slice())?;

        let mut writer = SegmentWriter::begin(&store.conn, new_id, out_level)?;
        writer.set_docs(docs_in);

        // Rows for the term currently being merged. This is the entire working
        // set of the merge.
        let mut current_term: Option<String> = None;
        let mut group: Vec<Vec<u8>> = Vec::new();

        let result = (|| -> Result<()> {
            while let Some(row) = rows.next()? {
                let term: String = row.get(0)?;
                let postings: Vec<u8> = row.get(2)?;

                match &current_term {
                    Some(t) if *t == term => group.push(postings),
                    Some(t) => {
                        let t = t.clone();
                        emit(&mut writer, &t, &group, &dead, &mut stats)?;
                        group.clear();
                        group.push(postings);
                        current_term = Some(term);
                    }
                    None => {
                        group.push(postings);
                        current_term = Some(term);
                    }
                }
            }
            if let Some(t) = current_term.take() {
                emit(&mut writer, &t, &group, &dead, &mut stats)?;
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                let info = writer.commit(store)?;
                stats.bytes_out = info.bytes as u64;
                stats.terms_out = info.terms as u64;
            }
            Err(e) => {
                writer.rollback()?;
                return Err(e);
            }
        }
    }
    drop(reader);

    store.drop_segments(ids)?;
    stats.elapsed_ms = start.elapsed().as_millis();

    if verbose {
        eprintln!(
            "  merged {} segments -> {} at level {} ({} terms, {} postings, {:.1} MB, peak term {:.0} KB, {} ms)",
            ids.len(),
            new_id,
            out_level,
            stats.terms_out,
            stats.postings_out,
            stats.bytes_out as f64 / 1e6,
            stats.peak_term_bytes as f64 / 1e3,
            stats.elapsed_ms
        );
    }
    Ok(stats)
}

/// Merge one term's postings lists and write the result.
///
/// `lists` are each ascending by document id; segments hold disjoint id
/// ranges, but a merged segment can interleave with a newer one, so this does
/// a real k-way merge rather than a concatenation.
fn emit(
    writer: &mut SegmentWriter<'_>,
    term: &str,
    lists: &[Vec<u8>],
    dead: &HashSet<i64>,
    stats: &mut MergeStats,
) -> Result<()> {
    let peak: usize = lists.iter().map(|l| l.len()).sum();
    stats.peak_term_bytes = stats.peak_term_bytes.max(peak);

    let mut readers: Vec<PostingsReader<'_>> = lists.iter().map(|b| PostingsReader::new(b)).collect();
    let mut heads: Vec<Option<crate::segment::Entry>> =
        readers.iter_mut().map(|r| r.next_entry()).collect();
    stats.postings_in += heads.iter().filter(|h| h.is_some()).count() as u64;

    let mut out = PostingsBuilder::new();
    loop {
        // k is the number of segments being merged, single digits by policy,
        // so a linear scan beats a binary heap on both constant factor and
        // clarity.
        let mut best: Option<usize> = None;
        for (i, h) in heads.iter().enumerate() {
            if let Some(entry) = h {
                if best.map_or(true, |b| entry.doc < heads[b].as_ref().unwrap().doc) {
                    best = Some(i);
                }
            }
        }
        let Some(i) = best else { break };
        let entry = heads[i].unwrap();

        if dead.contains(&entry.doc) {
            // Compaction: the document was deleted or superseded, so its
            // postings die here rather than being carried forward.
            readers[i].skip_positions();
        } else {
            let positions = readers[i].positions_raw();
            out.push_encoded(entry.doc, entry.tf, entry.norm, positions);
            stats.postings_out += 1;
        }
        heads[i] = readers[i].next_entry();
        if heads[i].is_some() {
            stats.postings_in += 1;
        }
    }

    // A term whose every posting was deleted disappears from the index.
    if out.df > 0 {
        writer.add_term(term, out.df, out.ttf, &out.blob)?;
    }
    Ok(())
}

/// Report how SQLite plans the merge scan. Used by `ferret explain` to show
/// that the ordering comes from the B-tree and not from a sort buffer.
pub fn explain_merge_plan(conn: &Connection, ids: &[i64]) -> Result<Vec<String>> {
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "EXPLAIN QUERY PLAN SELECT term, seg, postings FROM terms WHERE seg IN ({placeholders}) ORDER BY term, seg"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::ToSql> = ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
    let rows = stmt.query_map(params.as_slice(), |r| r.get::<_, String>(3))?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}
