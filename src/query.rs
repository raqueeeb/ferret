//! Query parsing and execution.
//!
//! # Ranking
//!
//! Okapi BM25, computed as
//!
//! ```text
//! idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
//! score(d) = Σ_t idf(t) * ( tf * (k1 + 1) ) / ( tf + k1 * (1 - b + b * len/avgdl) )
//! ```
//!
//! with `k1 = 1.2`, `b = 0.75`. The `ln(1 + ...)` form is used instead of the
//! textbook `ln((N - df + 0.5)/(df + 0.5))` because the textbook version goes
//! negative for terms present in more than half the corpus, which lets a
//! common word actively push a document *down* the ranking. For a corpus with
//! no stopword list — and this one has none, on purpose — that matters.
//!
//! # Execution
//!
//! Term-at-a-time. For each term, walk its postings across every segment and
//! accumulate into a score map. Document-at-a-time with WAND skipping would
//! win on very long lists, but it needs skip pointers in the postings format,
//! and at the corpus sizes this targets the accumulator never gets large
//! enough for the difference to show. Noted as the next thing to build, not
//! hand-waved as unnecessary.

use rusqlite::Connection;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::time::Instant;

use crate::codec::norm_table;
use crate::segment::PostingsReader;
use crate::store::Store;
use crate::tokenize;
use crate::Result;

pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;
/// Ceiling on how many terms a single `prefix*` may expand to.
pub const MAX_PREFIX_EXPANSION: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    Should,
    Must,
    MustNot,
}

#[derive(Debug, Clone)]
pub enum Clause {
    Term { text: String, occur: Occur },
    Prefix { text: String, occur: Occur },
    Phrase { terms: Vec<String>, occur: Occur },
}

#[derive(Debug, Clone, Default)]
pub struct Filters {
    /// Restrict to files with one of these extensions.
    pub ext: Vec<String>,
    /// Restrict to paths containing this substring.
    pub path: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Query {
    pub clauses: Vec<Clause>,
    pub filters: Filters,
}

/// What the planner did, surfaced to the user. This is the difference between
/// "it is fast" and "here is why it is fast".
#[derive(Debug, Clone, Default)]
pub struct Plan {
    pub terms: Vec<TermPlan>,
    pub segments: usize,
    pub postings_scanned: u64,
    pub candidates: usize,
    pub tombstones_skipped: u64,
    pub prefix_expansions: usize,
    pub parse_us: u128,
    pub search_us: u128,
}

#[derive(Debug, Clone)]
pub struct TermPlan {
    pub term: String,
    pub df: u64,
    pub idf: f32,
    pub segments: usize,
    pub bytes: usize,
}

/// The documents a phrase occurs in, and what the phrase contributes to each
/// one's score.
type PhraseMatches = (HashSet<i64>, Vec<(i64, f32)>);

#[derive(Debug, Clone)]
pub struct Hit {
    pub doc: i64,
    pub path: String,
    pub score: f32,
}

/// Parse a query string.
///
/// * `word` — should match, contributes to the score
/// * `+word` — required
/// * `-word` — excluded
/// * `"two words"` — phrase, required
/// * `word*` — prefix expansion
/// * `ext:rs` / `path:src` — filters, not scored
pub fn parse(input: &str) -> Query {
    let mut query = Query::default();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let mut occur = Occur::Should;
        match chars[i] {
            '+' => {
                occur = Occur::Must;
                i += 1;
            }
            '-' => {
                occur = Occur::MustNot;
                i += 1;
            }
            _ => {}
        }
        if i >= chars.len() {
            break;
        }

        if chars[i] == '"' {
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            let raw: String = chars[start..i].iter().collect();
            if i < chars.len() {
                i += 1;
            }
            let terms = tokenize::query_tokens(&raw);
            if terms.len() == 1 {
                query.clauses.push(Clause::Term {
                    text: terms[0].clone(),
                    occur: if occur == Occur::Should {
                        Occur::Must
                    } else {
                        occur
                    },
                });
            } else if !terms.is_empty() {
                query.clauses.push(Clause::Phrase {
                    terms,
                    occur: if occur == Occur::Should {
                        Occur::Must
                    } else {
                        occur
                    },
                });
            }
            continue;
        }

        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        let raw: String = chars[start..i].iter().collect();

        if let Some(rest) = raw.strip_prefix("ext:") {
            let e = rest.trim_start_matches('.').to_lowercase();
            if !e.is_empty() {
                query.filters.ext.push(e);
            }
            continue;
        }
        if let Some(rest) = raw.strip_prefix("path:") {
            if !rest.is_empty() {
                query.filters.path = Some(rest.to_lowercase());
            }
            continue;
        }

        if let Some(stem) = raw.strip_suffix('*') {
            let toks = tokenize::query_tokens(stem);
            if let Some(t) = toks.first() {
                query.clauses.push(Clause::Prefix {
                    text: t.clone(),
                    occur,
                });
                continue;
            }
        }
        for t in tokenize::query_tokens(&raw) {
            query.clauses.push(Clause::Term { text: t, occur });
        }
    }
    query
}

/// A warm searcher. Construction loads the corpus statistics and the tombstone
/// set; everything else is read per query straight out of the page cache.
///
/// Keeping one of these alive is the difference between the numbers `ferret
/// serve` reports and the numbers a cold CLI invocation reports.
pub struct Searcher<'a> {
    conn: &'a Connection,
    pub num_docs: u64,
    pub avg_len: f32,
    dead: HashSet<i64>,
    norms: [f32; 256],
    pub segment_count: usize,
}

impl<'a> Searcher<'a> {
    pub fn new(store: &'a Store) -> Result<Searcher<'a>> {
        let (num_docs, _) = store.corpus_stats()?;
        Ok(Searcher {
            conn: &store.conn,
            num_docs,
            avg_len: store.avg_doc_len()?,
            dead: store.tombstoned_ids()?,
            norms: norm_table(),
            segment_count: store.segments()?.len(),
        })
    }

    fn idf(&self, df: u64) -> f32 {
        let n = self.num_docs.max(1) as f32;
        let df = df as f32;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// All postings rows for one term, across every segment.
    fn postings_for(&self, term: &str) -> Result<Vec<(u64, Vec<u8>)>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT df, postings FROM terms WHERE term = ?1")?;
        let rows = stmt.query_map([term], |r| {
            Ok((r.get::<_, i64>(0)? as u64, r.get::<_, Vec<u8>>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Terms sharing a prefix, via a range scan on the clustered index.
    fn expand_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        let upper = prefix_upper_bound(prefix);
        let mut stmt = self.conn.prepare_cached(
            "SELECT DISTINCT term FROM terms WHERE term >= ?1 AND term < ?2 LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            rusqlite::params![prefix, upper, MAX_PREFIX_EXPANSION as i64],
            |r| r.get::<_, String>(0),
        )?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Run `query` and return the top `limit` documents.
    pub fn search(&self, query: &Query, limit: usize) -> Result<(Vec<Hit>, Plan)> {
        let start = Instant::now();
        let mut plan = Plan {
            segments: self.segment_count,
            ..Default::default()
        };

        let mut scores: HashMap<i64, f32> = HashMap::new();
        let mut required: Option<HashSet<i64>> = None;
        let mut excluded: HashSet<i64> = HashSet::new();
        let mut matched_any = false;

        for clause in &query.clauses {
            match clause {
                Clause::Term { text, occur } => {
                    let docs = self.score_term(text, &mut scores, &mut plan, *occur)?;
                    self.apply_occur(*occur, docs, &mut required, &mut excluded, &mut matched_any);
                }
                Clause::Prefix { text, occur } => {
                    let expansions = self.expand_prefix(text)?;
                    plan.prefix_expansions += expansions.len();
                    let mut union: HashSet<i64> = HashSet::new();
                    for term in &expansions {
                        let docs = self.score_term(term, &mut scores, &mut plan, *occur)?;
                        union.extend(docs);
                    }
                    self.apply_occur(
                        *occur,
                        union,
                        &mut required,
                        &mut excluded,
                        &mut matched_any,
                    );
                }
                Clause::Phrase { terms, occur } => {
                    let (docs, contribution) = self.match_phrase(terms, &mut plan)?;
                    if *occur != Occur::MustNot {
                        for (doc, score) in contribution {
                            *scores.entry(doc).or_insert(0.0) += score;
                        }
                    }
                    self.apply_occur(*occur, docs, &mut required, &mut excluded, &mut matched_any);
                }
            }
        }

        // Filters resolve to a document id set with one indexed catalogue
        // query, rather than a per-candidate path lookup.
        let allowed = self.filter_set(&query.filters)?;

        let mut heap: BinaryHeap<Ranked> = BinaryHeap::with_capacity(limit + 1);
        plan.candidates = scores.len();
        for (doc, score) in scores {
            if excluded.contains(&doc) {
                continue;
            }
            if let Some(req) = &required {
                if !req.contains(&doc) {
                    continue;
                }
            }
            if let Some(allow) = &allowed {
                if !allow.contains(&doc) {
                    continue;
                }
            }
            // A min-heap capped at `limit`: O(n log k) with k in the tens,
            // instead of sorting a candidate list that can run to six figures.
            heap.push(Ranked { score: -score, doc });
            if heap.len() > limit {
                heap.pop();
            }
        }

        // `Ranked.score` is negated, so ascending order on it is descending
        // order on the real score.
        let mut ranked: Vec<Ranked> = heap.into_vec();
        ranked.sort();

        let mut hits = Vec::with_capacity(ranked.len());
        let mut stmt = self
            .conn
            .prepare_cached("SELECT path FROM docs WHERE id = ?1")?;
        for r in ranked {
            let path: String = match stmt.query_row([r.doc], |row| row.get(0)) {
                Ok(p) => p,
                Err(_) => continue,
            };
            hits.push(Hit {
                doc: r.doc,
                path,
                score: -r.score,
            });
        }

        plan.search_us = start.elapsed().as_micros();
        Ok((hits, plan))
    }

    fn apply_occur(
        &self,
        occur: Occur,
        docs: HashSet<i64>,
        required: &mut Option<HashSet<i64>>,
        excluded: &mut HashSet<i64>,
        matched_any: &mut bool,
    ) {
        match occur {
            Occur::Must => {
                *required = Some(match required.take() {
                    Some(existing) => existing.intersection(&docs).copied().collect(),
                    None => docs,
                });
                *matched_any = true;
            }
            Occur::MustNot => excluded.extend(docs),
            Occur::Should => {}
        }
    }

    /// Score one term into the accumulator, returning the documents it hit.
    fn score_term(
        &self,
        term: &str,
        scores: &mut HashMap<i64, f32>,
        plan: &mut Plan,
        occur: Occur,
    ) -> Result<HashSet<i64>> {
        let lists = self.postings_for(term)?;
        let df: u64 = lists.iter().map(|(df, _)| df).sum();
        if df == 0 {
            plan.terms.push(TermPlan {
                term: term.to_string(),
                df: 0,
                idf: 0.0,
                segments: 0,
                bytes: 0,
            });
            return Ok(HashSet::new());
        }
        let idf = self.idf(df);
        plan.terms.push(TermPlan {
            term: term.to_string(),
            df,
            idf,
            segments: lists.len(),
            bytes: lists.iter().map(|(_, b)| b.len()).sum(),
        });

        let mut docs = HashSet::with_capacity(df as usize);
        for (_, blob) in &lists {
            let mut reader = PostingsReader::new(blob);
            while let Some(e) = reader.next_entry() {
                plan.postings_scanned += 1;
                if self.dead.contains(&e.doc) {
                    plan.tombstones_skipped += 1;
                    continue;
                }
                docs.insert(e.doc);
                if occur == Occur::MustNot {
                    continue;
                }
                // The length norm rides along with the posting, so this loop
                // touches nothing outside the blob it is already scanning.
                let len = self.norms[e.norm as usize].max(1.0);
                let tf = e.tf as f32;
                let denom = tf + K1 * (1.0 - B + B * len / self.avg_len);
                *scores.entry(e.doc).or_insert(0.0) += idf * (tf * (K1 + 1.0)) / denom;
            }
        }
        Ok(docs)
    }

    /// Positional phrase match.
    ///
    /// Terms are processed rarest-first and positions are shifted by each
    /// term's offset in the phrase, so "the quick brown fox" starts from
    /// whichever of those four terms has the smallest posting list rather than
    /// from `the`. After the shift, a phrase occurrence is exactly a position
    /// value present in every term's set, which turns matching into repeated
    /// sorted-list intersection.
    ///
    /// Returns the documents the phrase occurs in, and the per-document score
    /// contribution of the phrase.
    fn match_phrase(&self, terms: &[String], plan: &mut Plan) -> Result<PhraseMatches> {
        let mut loaded: Vec<(usize, u64, Vec<Vec<u8>>)> = Vec::new();
        for (offset, term) in terms.iter().enumerate() {
            let lists = self.postings_for(term)?;
            let df: u64 = lists.iter().map(|(df, _)| df).sum();
            plan.terms.push(TermPlan {
                term: term.clone(),
                df,
                idf: self.idf(df),
                segments: lists.len(),
                bytes: lists.iter().map(|(_, b)| b.len()).sum(),
            });
            if df == 0 {
                return Ok((HashSet::new(), Vec::new()));
            }
            loaded.push((offset, df, lists.into_iter().map(|(_, b)| b).collect()));
        }
        loaded.sort_by_key(|(_, df, _)| *df);

        let mut acc: HashMap<i64, Vec<u32>> = HashMap::new();
        let mut positions = Vec::new();

        for (nth, (offset, _, blobs)) in loaded.iter().enumerate() {
            let mut next: HashMap<i64, Vec<u32>> = HashMap::new();
            for blob in blobs {
                let mut reader = PostingsReader::new(blob);
                while let Some(e) = reader.next_entry() {
                    plan.postings_scanned += 1;
                    if self.dead.contains(&e.doc) {
                        plan.tombstones_skipped += 1;
                        continue;
                    }
                    if nth > 0 && !acc.contains_key(&e.doc) {
                        continue;
                    }
                    reader.positions(&mut positions);
                    // Shift into a common frame: a phrase occurrence is a
                    // value shared by every term after shifting.
                    let shifted: Vec<u32> = positions
                        .iter()
                        .filter(|&&p| p >= *offset as u32)
                        .map(|&p| p - *offset as u32)
                        .collect();
                    if shifted.is_empty() {
                        continue;
                    }
                    if nth == 0 {
                        next.insert(e.doc, shifted);
                    } else if let Some(prev) = acc.get(&e.doc) {
                        let common = intersect_sorted(prev, &shifted);
                        if !common.is_empty() {
                            next.insert(e.doc, common);
                        }
                    }
                }
            }
            acc = next;
            if acc.is_empty() {
                return Ok((HashSet::new(), Vec::new()));
            }
        }

        // The phrase's own document frequency is now known exactly, so its idf
        // is exact rather than approximated from the component terms.
        let phrase_df = acc.len() as u64;
        let idf = self.idf(phrase_df);
        let docs: HashSet<i64> = acc.keys().copied().collect();
        let contribution = acc
            .into_iter()
            .map(|(doc, hits)| {
                let tf = hits.len() as f32;
                // Phrases are worth more than the sum of their words; the
                // multiplier is a tuning choice, not a result from the paper.
                (
                    doc,
                    idf * (tf * (K1 + 1.0)) / (tf + K1) * terms.len() as f32,
                )
            })
            .collect();
        Ok((docs, contribution))
    }

    /// Resolve path and extension filters to a document id set, or `None` when
    /// no filter is active.
    fn filter_set(&self, filters: &Filters) -> Result<Option<HashSet<i64>>> {
        if filters.ext.is_empty() && filters.path.is_none() {
            return Ok(None);
        }
        let mut set = HashSet::new();
        let mut stmt = self
            .conn
            .prepare("SELECT id, path FROM docs WHERE deleted = 0")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        for row in rows {
            let (id, path) = row?;
            let lower = path.to_lowercase();
            if !filters.ext.is_empty() {
                let ok = filters
                    .ext
                    .iter()
                    .any(|e| lower.ends_with(&format!(".{e}")));
                if !ok {
                    continue;
                }
            }
            if let Some(p) = &filters.path {
                if !lower.contains(p) {
                    continue;
                }
            }
            set.insert(id);
        }
        Ok(Some(set))
    }
}

/// Smallest string greater than every string starting with `prefix`, used as
/// the exclusive upper bound of the range scan.
fn prefix_upper_bound(prefix: &str) -> String {
    let mut chars: Vec<char> = prefix.chars().collect();
    while let Some(last) = chars.pop() {
        if let Some(next) = char::from_u32(last as u32 + 1) {
            chars.push(next);
            return chars.into_iter().collect();
        }
    }
    // Empty prefix: unbounded above.
    "\u{10FFFF}".to_string()
}

fn intersect_sorted(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    out
}

/// Heap entry. `score` is negated so `BinaryHeap` (a max-heap) evicts the
/// worst candidate.
#[derive(Debug, PartialEq)]
struct Ranked {
    score: f32,
    doc: i64,
}
impl Eq for Ranked {}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.doc.cmp(&other.doc))
    }
}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_operators() {
        let q = parse("alpha +beta -gamma \"delta epsilon\" ext:rs path:src pre*");
        assert_eq!(q.filters.ext, vec!["rs"]);
        assert_eq!(q.filters.path.as_deref(), Some("src"));
        let kinds: Vec<&str> = q
            .clauses
            .iter()
            .map(|c| match c {
                Clause::Term {
                    occur: Occur::Should,
                    ..
                } => "should",
                Clause::Term {
                    occur: Occur::Must, ..
                } => "must",
                Clause::Term {
                    occur: Occur::MustNot,
                    ..
                } => "not",
                Clause::Phrase { .. } => "phrase",
                Clause::Prefix { .. } => "prefix",
            })
            .collect();
        assert_eq!(kinds, vec!["should", "must", "not", "phrase", "prefix"]);
    }

    #[test]
    fn single_word_quotes_become_a_required_term() {
        let q = parse("\"solo\"");
        assert!(matches!(
            &q.clauses[0],
            Clause::Term {
                occur: Occur::Must,
                ..
            }
        ));
    }

    #[test]
    fn prefix_bounds_are_exclusive_and_tight() {
        assert_eq!(prefix_upper_bound("abc"), "abd");
        assert_eq!(prefix_upper_bound("az"), "a{");
    }

    #[test]
    fn sorted_intersection() {
        assert_eq!(intersect_sorted(&[1, 3, 5, 7], &[3, 4, 5, 9]), vec![3, 5]);
        assert!(intersect_sorted(&[1, 2], &[3, 4]).is_empty());
    }
}
