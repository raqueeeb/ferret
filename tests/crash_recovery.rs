//! Crash-consistency tests.
//!
//! The index is written in two places with different durability — the
//! catalogue in SQLite and the postings in a memtable that flushes on a size
//! trigger. These tests simulate losing the process in between.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferret::index::{Indexer, Options};
use ferret::query::{self, Searcher};
use ferret::store::Store;

static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_dir(name: &str) -> PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "ferret-crash-{}-{}-{}",
        name,
        std::process::id(),
        id
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn found(store: &Store, q: &str) -> usize {
    let searcher = Searcher::new(store).unwrap();
    searcher.search(&query::parse(q), 1000).unwrap().0.len()
}

#[test]
fn documents_lost_before_a_flush_are_reindexed_not_skipped() {
    let dir = temp_dir("watermark");
    for i in 0..30 {
        std::fs::write(
            dir.join(format!("f{i}.txt")),
            format!("searchable content {i}"),
        )
        .unwrap();
    }

    // Index without ever flushing: the catalogue is written, the postings are
    // not. Dropping the indexer here is the same thing a kill -9 does.
    {
        let store = Store::open(&dir).unwrap();
        let opts = Options {
            budget: 1 << 30,
            threads: 1,
            auto_merge: false,
            ..Default::default()
        };
        let mut indexer = Indexer::new(&store, opts);
        indexer.crawl().unwrap();
        // deliberately no finish()/flush()
        std::mem::forget(indexer);

        let catalogued: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM docs", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            catalogued, 30,
            "the catalogue should have recorded the files"
        );
        assert_eq!(found(&store, "searchable"), 0, "no postings were flushed");
    }

    // A fresh run must notice and redo the work.
    let store = Store::open(&dir).unwrap();
    let lost = store.recover().unwrap();
    assert_eq!(lost, 30);
    // Tombstoned rather than deleted, so any postings a partial flush did
    // write are still reachable by the deletion filter and the merge.
    let dead: i64 = store
        .conn
        .query_row("SELECT COUNT(*) FROM docs WHERE deleted = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(dead, 30);

    let mut indexer = Indexer::new(
        &store,
        Options {
            threads: 1,
            ..Default::default()
        },
    );
    indexer.crawl().unwrap();
    indexer.finish().unwrap();
    assert_eq!(
        indexer.stats.indexed, 30,
        "files must be re-read, not treated as unchanged"
    );
    assert_eq!(found(&store, "searchable"), 30);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_flushed_prefix_survives_and_is_not_redone() {
    let dir = temp_dir("prefix");
    for i in 0..40 {
        std::fs::write(dir.join(format!("f{i}.txt")), format!("durable marker {i}")).unwrap();
    }

    {
        let store = Store::open(&dir).unwrap();
        let mut indexer = Indexer::new(
            &store,
            Options {
                budget: 1 << 30,
                threads: 1,
                auto_merge: false,
                ..Default::default()
            },
        );
        indexer.crawl().unwrap();
        indexer.flush().unwrap(); // everything is durable now
        std::mem::forget(indexer);
    }

    let store = Store::open(&dir).unwrap();
    assert_eq!(
        store.recover().unwrap(),
        0,
        "nothing should be discarded after a clean flush"
    );
    assert_eq!(found(&store, "durable"), 40);

    let mut indexer = Indexer::new(
        &store,
        Options {
            threads: 1,
            ..Default::default()
        },
    );
    indexer.crawl().unwrap();
    assert_eq!(indexer.stats.indexed, 0);
    assert_eq!(indexer.stats.unchanged, 40);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn corpus_statistics_stay_consistent_after_recovery() {
    let dir = temp_dir("stats");
    for i in 0..25 {
        std::fs::write(dir.join(format!("f{i}.txt")), "one two three four five").unwrap();
    }

    {
        let store = Store::open(&dir).unwrap();
        let mut indexer = Indexer::new(
            &store,
            Options {
                budget: 1 << 30,
                threads: 1,
                auto_merge: false,
                ..Default::default()
            },
        );
        indexer.crawl().unwrap();
        std::mem::forget(indexer);
    }

    let store = Store::open(&dir).unwrap();
    store.recover().unwrap();
    let (docs, tokens) = store.corpus_stats().unwrap();
    assert_eq!(docs, 0, "document count must not include discarded rows");
    assert_eq!(tokens, 0, "token total must not include discarded rows");

    // BM25 divides by the average document length; a stale total here would
    // quietly skew every score in the index.
    assert_eq!(store.avg_doc_len().unwrap(), 1.0);

    let _ = std::fs::remove_dir_all(&dir);
}
