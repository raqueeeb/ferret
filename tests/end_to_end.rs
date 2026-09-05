//! End-to-end tests over a real index on a real temporary directory.
//!
//! The unit tests cover the codecs in isolation; these cover the properties
//! that only appear once the pieces are assembled — that a merge does not
//! change results, that re-indexing a changed file removes its old terms, and
//! that a tiny memtable budget produces the same answers as a large one.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use ferret::index::{Indexer, Options};
use ferret::merge;
use ferret::query::{self, Searcher};
use ferret::store::Store;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

struct TempCorpus(PathBuf);

impl TempCorpus {
    fn new(name: &str) -> TempCorpus {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("ferret-test-{}-{}-{}", name, std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempCorpus(dir)
    }

    fn write(&self, rel: &str, body: &str) {
        let path = self.0.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, body).unwrap();
    }

    fn remove(&self, rel: &str) {
        std::fs::remove_file(self.0.join(rel)).unwrap();
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempCorpus {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn build(corpus: &TempCorpus, budget: usize) -> Store {
    let store = Store::open(corpus.path()).unwrap();
    let opts = Options { budget, threads: 2, auto_merge: false, ..Default::default() };
    let mut indexer = Indexer::new(&store, opts);
    indexer.crawl().unwrap();
    indexer.finish().unwrap();
    store
}

fn hits(store: &Store, q: &str) -> Vec<String> {
    let searcher = Searcher::new(store).unwrap();
    let parsed = query::parse(q);
    let (hits, _) = searcher.search(&parsed, 1000).unwrap();
    hits.into_iter().map(|h| h.path).collect()
}

#[test]
fn finds_a_document_by_a_word_it_contains() {
    let c = TempCorpus::new("basic");
    c.write("a.txt", "the inverted index maps terms to documents");
    c.write("b.txt", "unrelated content about gardening");
    let store = build(&c, 1 << 20);

    assert_eq!(hits(&store, "inverted"), vec!["a.txt"]);
    assert_eq!(hits(&store, "gardening"), vec!["b.txt"]);
    assert!(hits(&store, "absent").is_empty());
}

#[test]
fn phrase_queries_respect_word_order() {
    let c = TempCorpus::new("phrase");
    c.write("ordered.txt", "a memory bounded merge is the goal");
    c.write("scrambled.txt", "merge bounded memory appear here but not adjacent");
    c.write("split.txt", "memory is bounded and then we merge");
    let store = build(&c, 1 << 20);

    assert_eq!(hits(&store, "\"memory bounded merge\""), vec!["ordered.txt"]);
    // The bare terms match all three.
    assert_eq!(hits(&store, "memory bounded merge").len(), 3);
}

#[test]
fn bm25_prefers_denser_and_shorter_documents() {
    let c = TempCorpus::new("bm25");
    let padding = "filler words ".repeat(400);
    c.write("dense.txt", "postings postings postings");
    c.write("diluted.txt", &format!("postings {padding}"));
    let store = build(&c, 1 << 20);

    let ranked = hits(&store, "postings");
    assert_eq!(ranked[0], "dense.txt", "length normalisation should favour the short document");
    assert_eq!(ranked.len(), 2);
}

#[test]
fn required_and_excluded_terms_filter_results() {
    let c = TempCorpus::new("occur");
    c.write("both.txt", "alpha beta");
    c.write("alpha_only.txt", "alpha gamma");
    c.write("beta_only.txt", "beta delta");
    let store = build(&c, 1 << 20);

    assert_eq!(hits(&store, "+alpha +beta"), vec!["both.txt"]);
    let mut minus = hits(&store, "alpha -beta");
    minus.sort();
    assert_eq!(minus, vec!["alpha_only.txt"]);
}

#[test]
fn prefix_queries_expand_over_the_term_range() {
    let c = TempCorpus::new("prefix");
    c.write("a.txt", "tokenizer");
    c.write("b.txt", "tokenize");
    c.write("c.txt", "token");
    c.write("d.txt", "unrelated");
    let store = build(&c, 1 << 20);

    let mut found = hits(&store, "token*");
    found.sort();
    assert_eq!(found, vec!["a.txt", "b.txt", "c.txt"]);
}

#[test]
fn extension_and_path_filters_narrow_results() {
    let c = TempCorpus::new("filters");
    c.write("src/main.rs", "shared keyword here");
    c.write("docs/guide.md", "shared keyword here");
    let store = build(&c, 1 << 20);

    assert_eq!(hits(&store, "keyword ext:rs"), vec!["src/main.rs"]);
    assert_eq!(hits(&store, "keyword path:docs"), vec!["docs/guide.md"]);
    assert_eq!(hits(&store, "keyword").len(), 2);
}

#[test]
fn identifier_parts_are_searchable() {
    let c = TempCorpus::new("identifiers");
    c.write("code.rs", "fn parse_json_body(req: HttpRequest) {}");
    let store = build(&c, 1 << 20);

    assert_eq!(hits(&store, "json"), vec!["code.rs"]);
    assert_eq!(hits(&store, "http"), vec!["code.rs"]);
    assert_eq!(hits(&store, "parse_json_body"), vec!["code.rs"]);
}

#[test]
fn reindexing_a_changed_file_retires_its_old_terms() {
    let c = TempCorpus::new("incremental");
    c.write("notes.txt", "original vocabulary");
    let store = build(&c, 1 << 20);
    assert_eq!(hits(&store, "original"), vec!["notes.txt"]);
    drop(store);

    // mtime resolution is one second in the catalogue, so a same-second edit
    // must still be detected. Size differs here, which is the other half of
    // the change check.
    c.write("notes.txt", "replacement vocabulary entirely different now");
    let store = build(&c, 1 << 20);

    assert!(hits(&store, "original").is_empty(), "stale posting survived");
    assert_eq!(hits(&store, "replacement"), vec!["notes.txt"]);
    let (docs, _) = store.corpus_stats().unwrap();
    assert_eq!(docs, 1, "re-indexing must not double-count the document");
}

#[test]
fn deleting_a_file_removes_it_from_results() {
    let c = TempCorpus::new("deletion");
    c.write("keep.txt", "shared term");
    c.write("drop.txt", "shared term");
    let store = build(&c, 1 << 20);
    assert_eq!(hits(&store, "shared").len(), 2);
    drop(store);

    c.remove("drop.txt");
    let store = build(&c, 1 << 20);
    assert_eq!(hits(&store, "shared"), vec!["keep.txt"]);
}

#[test]
fn a_small_budget_produces_many_segments_and_identical_answers() {
    let c = TempCorpus::new("segments");
    for i in 0..60 {
        c.write(&format!("f{i}.txt"), &format!("common term document number {i} unique{i}"));
    }

    // 4KB budget forces repeated flushes mid-crawl.
    let store = build(&c, 4096);
    let segments = store.segments().unwrap().len();
    assert!(segments > 1, "expected multiple segments, got {segments}");

    let before = hits(&store, "common");
    assert_eq!(before.len(), 60);
    assert_eq!(hits(&store, "unique7"), vec!["f7.txt"]);
}

#[test]
fn merging_preserves_results_and_shrinks_the_segment_count() {
    let c = TempCorpus::new("merge");
    for i in 0..80 {
        c.write(&format!("f{i}.txt"), &format!("shared vocabulary term unique{i} padding padding"));
    }
    let store = build(&c, 4096);

    let searcher = Searcher::new(&store).unwrap();
    let (before, _) = searcher
        .search(&query::parse("shared"), 100)
        .unwrap();
    let before: Vec<(String, f32)> = before.into_iter().map(|h| (h.path, h.score)).collect();
    let segments_before = store.segments().unwrap().len();
    drop(searcher);

    merge::merge_all(&store, false).unwrap();
    let segments_after = store.segments().unwrap().len();
    assert!(segments_after < segments_before, "{segments_before} -> {segments_after}");
    assert_eq!(segments_after, 1);

    let searcher = Searcher::new(&store).unwrap();
    let (after, _) = searcher.search(&query::parse("shared"), 100).unwrap();
    let after: Vec<(String, f32)> = after.into_iter().map(|h| (h.path, h.score)).collect();

    assert_eq!(before.len(), after.len());
    for ((pb, sb), (pa, sa)) in before.iter().zip(after.iter()) {
        assert_eq!(pb, pa, "merge reordered results");
        assert!((sb - sa).abs() < 1e-4, "merge changed score: {sb} vs {sa}");
    }
    assert_eq!(hits(&store, "unique42"), vec!["f42.txt"]);
}

#[test]
fn merging_drops_postings_for_deleted_documents() {
    let c = TempCorpus::new("compaction");
    for i in 0..40 {
        c.write(&format!("f{i}.txt"), "recurring term in every file");
    }
    let store = build(&c, 4096);
    let postings_before: i64 = store.segments().unwrap().iter().map(|s| s.postings).sum();
    drop(store);

    for i in 0..20 {
        c.remove(&format!("f{i}.txt"));
    }
    let store = build(&c, 4096);
    let stats = merge::merge_all(&store, false).unwrap();
    let postings_after: i64 = store.segments().unwrap().iter().map(|s| s.postings).sum();

    assert!(
        postings_after < postings_before,
        "compaction kept dead postings: {postings_before} -> {postings_after}"
    );
    assert_eq!(hits(&store, "recurring").len(), 20);
    assert!(stats.postings_out < stats.postings_in);

    // With every posting rewritten there is nothing left to tombstone.
    let dead: i64 = store
        .conn
        .query_row("SELECT COUNT(*) FROM docs WHERE deleted = 1", [], |r| r.get(0))
        .unwrap();
    assert_eq!(dead, 0);
}

#[test]
fn merge_working_set_stays_far_below_the_index_size() {
    let c = TempCorpus::new("bounded");
    // Many documents sharing one term: the merge must stream, not buffer.
    for i in 0..400 {
        c.write(
            &format!("d{i}.txt"),
            &format!("ubiquitous {} filler{i}", "ubiquitous ".repeat(20)),
        );
    }
    let store = build(&c, 8192);
    assert!(store.segments().unwrap().len() > 3);
    let index_bytes: i64 = store.segments().unwrap().iter().map(|s| s.bytes).sum();

    let stats = merge::merge_all(&store, false).unwrap();
    assert!(
        (stats.peak_term_bytes as i64) < index_bytes,
        "peak working set {} was not below index size {}",
        stats.peak_term_bytes,
        index_bytes
    );
    assert_eq!(hits(&store, "ubiquitous").len(), 400);
}

#[test]
fn binary_files_are_skipped_but_text_beside_them_is_not() {
    let c = TempCorpus::new("binary");
    std::fs::write(c.path().join("blob.bin"), [0u8, 1, 2, 3, 0, 5]).unwrap();
    std::fs::write(c.path().join("image.png"), b"\x89PNG\r\n\x1a\n mentionable").unwrap();
    c.write("readme.md", "mentionable text");
    let store = build(&c, 1 << 20);

    assert_eq!(hits(&store, "mentionable"), vec!["readme.md"]);
}

#[test]
fn build_directories_are_not_crawled() {
    let c = TempCorpus::new("skipdirs");
    c.write("src/lib.rs", "distinctive marker");
    c.write("node_modules/dep/index.js", "distinctive marker");
    c.write("target/debug/build.rs", "distinctive marker");
    let store = build(&c, 1 << 20);

    assert_eq!(hits(&store, "distinctive"), vec!["src/lib.rs"]);
}

#[test]
fn a_second_crawl_reindexes_nothing() {
    let c = TempCorpus::new("idempotent");
    for i in 0..20 {
        c.write(&format!("f{i}.txt"), "stable content");
    }
    let store = build(&c, 1 << 20);
    drop(store);

    let store = Store::open(c.path()).unwrap();
    let mut indexer = Indexer::new(&store, Options { auto_merge: false, ..Default::default() });
    indexer.crawl().unwrap();
    indexer.finish().unwrap();

    assert_eq!(indexer.stats.indexed, 0, "unchanged files were re-indexed");
    assert_eq!(indexer.stats.unchanged, 20);
}
