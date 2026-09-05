//! Command line shell.
//!
//! Argument parsing is hand-written. `clap` would be three lines shorter per
//! flag and pull in about forty crates, which for a tool whose selling point
//! is "I built the parts myself" is the wrong trade.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Instant;

use ferret::index::{Indexer, Options};
use ferret::query::{self, Searcher};
use ferret::store::Store;
use ferret::{commas, human_bytes, merge, serve, snippet, watch, Result};

const USAGE: &str = "\
ferret — local file search with a hand-built inverted index

usage:
  ferret index  [PATH] [options]      build or update the index
  ferret search QUERY [options]       search the index
  ferret watch  [PATH] [options]      re-index as files change
  ferret serve  [PATH] [--port N]     browser UI on localhost
  ferret stats  [PATH]                index size, segments, term counts
  ferret merge  [PATH] [--all]        compact segments
  ferret bench  [PATH] [--runs N]     query latency distribution
  ferret explain [PATH]               show how SQLite executes the hot paths

query syntax:
  word            contributes to the score
  +word           required
  -word           excluded
  \"two words\"     phrase, matched by position
  word*           prefix expansion
  ext:rs path:src filter by extension or path fragment

index options:
  --budget MB     memtable size before flushing        (default 64)
  --threads N     extractor threads                    (default: cores, max 8)
  --hidden        include dotfiles and dot-directories
  --no-merge      skip compaction after indexing
  -v, --verbose   report flushes and merges

search options:
  -n N            results to show                      (default 10)
  --plan          print term statistics and timings
  --no-snippet    paths and scores only
";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{USAGE}");
        return;
    }
    if let Err(e) = dispatch(&args) {
        eprintln!("ferret: {e}");
        std::process::exit(1);
    }
}

fn dispatch(args: &[String]) -> Result<()> {
    let rest = &args[1..];
    match args[0].as_str() {
        "index" => cmd_index(rest),
        "search" | "s" => cmd_search(rest),
        "watch" => cmd_watch(rest),
        "serve" => cmd_serve(rest),
        "stats" => cmd_stats(rest),
        "merge" => cmd_merge(rest),
        "bench" => cmd_bench(rest),
        "explain" => cmd_explain(rest),
        other => Err(format!("unknown command `{other}` — run `ferret --help`").into()),
    }
}

// ---------------------------------------------------------------------------
// argument helpers
// ---------------------------------------------------------------------------

struct Args {
    positional: Vec<String>,
    flags: Vec<String>,
    values: Vec<(String, String)>,
}

fn parse_args(args: &[String], value_flags: &[&str]) -> Args {
    let mut out = Args { positional: Vec::new(), flags: Vec::new(), values: Vec::new() };
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(name) = a.strip_prefix("--").or_else(|| a.strip_prefix('-')) {
            if let Some((k, v)) = name.split_once('=') {
                out.values.push((k.to_string(), v.to_string()));
            } else if value_flags.contains(&name) && i + 1 < args.len() {
                out.values.push((name.to_string(), args[i + 1].clone()));
                i += 1;
            } else {
                out.flags.push(name.to_string());
            }
        } else {
            out.positional.push(a.clone());
        }
        i += 1;
    }
    out
}

impl Args {
    fn has(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }
    fn value(&self, name: &str) -> Option<&str> {
        self.values.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
    fn num(&self, name: &str) -> Option<u64> {
        self.value(name).and_then(|v| v.parse().ok())
    }
    fn root(&self) -> PathBuf {
        self.value("in")
            .map(PathBuf::from)
            .or_else(|| self.positional.first().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("."))
    }
}

fn index_options(args: &Args) -> Options {
    let mut opts = Options {
        hidden: args.has("hidden"),
        auto_merge: !args.has("no-merge"),
        verbose: args.has("v") || args.has("verbose"),
        ..Default::default()
    };
    if let Some(mb) = args.num("budget") {
        opts.budget = (mb as usize).max(1) * 1024 * 1024;
    }
    if let Some(n) = args.num("threads") {
        opts.threads = (n as usize).max(1);
    }
    opts
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

fn cmd_index(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["budget", "threads", "in"]);
    let root = args.root();
    let store = Store::open(&root)?;
    let opts = index_options(&args);

    println!("indexing {}", store.root.display());
    println!(
        "  memtable budget {} · {} extractor threads",
        human_bytes(opts.budget as u64),
        opts.threads
    );

    let mut indexer = Indexer::new(&store, opts);
    indexer.crawl()?;
    indexer.finish()?;

    let s = indexer.stats.clone();
    let (docs, tokens) = store.corpus_stats()?;
    let index_bytes = store.index_bytes();
    let secs = (s.elapsed_ms as f64 / 1000.0).max(0.001);

    println!();
    println!(
        "  {} files indexed, {} unchanged, {} skipped, {} removed, {} errors",
        commas(s.indexed),
        commas(s.unchanged),
        commas(s.skipped),
        commas(s.removed),
        commas(s.errors)
    );
    println!(
        "  {} tokens from {} of text in {:.1}s ({:.0} files/s)",
        commas(s.tokens),
        human_bytes(s.bytes_read),
        secs,
        s.indexed as f64 / secs
    );
    println!(
        "  index is {} for {} live documents — {:.1}% of the text it covers",
        human_bytes(index_bytes),
        commas(docs),
        if s.bytes_read > 0 { index_bytes as f64 / s.bytes_read as f64 * 100.0 } else { 0.0 }
    );
    let _ = tokens;
    println!("  {} segments after merge", store.segments()?.len());
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["n", "in", "limit"]);
    if args.positional.is_empty() {
        return Err("nothing to search for — try `ferret search \"postings list\"`".into());
    }
    let root = args.value("in").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let query_text = args.positional.join(" ");
    let limit = args.num("n").or_else(|| args.num("limit")).unwrap_or(10) as usize;

    let open_start = Instant::now();
    let store = Store::open_existing(&root)?;
    let parsed = query::parse(&query_text);
    let searcher = Searcher::new(&store)?;
    let open_us = open_start.elapsed().as_micros();

    let (hits, plan) = searcher.search(&parsed, limit)?;

    let color = std::io::stdout().is_terminal();
    let (bold, dim, mark, reset) = if color {
        ("\x1b[1m", "\x1b[2m", "\x1b[30;43m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };

    if hits.is_empty() {
        println!("no matches for {query_text}");
        if plan.terms.iter().all(|t| t.df == 0) && !plan.terms.is_empty() {
            println!("{dim}none of those terms are in the index{reset}");
        }
        return Ok(());
    }

    let highlight: Vec<String> = plan.terms.iter().map(|t| t.term.clone()).collect();
    let show_snippets = !args.has("no-snippet");

    for hit in &hits {
        println!("{bold}{}{reset}  {dim}{:.3}{reset}", hit.path, hit.score);
        if show_snippets {
            let snip = snippet::for_file(&store.root.join(&hit.path), &highlight);
            if !snip.text.is_empty() {
                let mut line = String::new();
                let mut cursor = 0;
                for &(a, b) in &snip.marks {
                    if a < cursor || b > snip.text.len() {
                        continue;
                    }
                    line.push_str(&snip.text[cursor..a]);
                    line.push_str(mark);
                    line.push_str(&snip.text[a..b]);
                    line.push_str(reset);
                    cursor = b;
                }
                line.push_str(&snip.text[cursor..]);
                println!("  {dim}{}:{reset} {}", snip.line, line);
            }
        }
    }

    println!();
    println!(
        "{dim}{} matched · {} postings scanned · {} segments · {:.2} ms search, {:.1} ms to open the index{reset}",
        commas(plan.candidates as u64),
        commas(plan.postings_scanned),
        plan.segments,
        plan.search_us as f64 / 1000.0,
        open_us as f64 / 1000.0
    );

    if args.has("plan") {
        println!();
        println!("  {:<28} {:>9} {:>7} {:>9} {:>5}", "term", "docs", "idf", "postings", "segs");
        for t in &plan.terms {
            println!(
                "  {:<28} {:>9} {:>7.3} {:>9} {:>5}",
                truncate(&t.term, 28),
                commas(t.df),
                t.idf,
                human_bytes(t.bytes as u64),
                t.segments
            );
        }
        if plan.tombstones_skipped > 0 {
            println!(
                "  {} stale postings skipped (files edited since the last merge)",
                commas(plan.tombstones_skipped)
            );
        }
        if plan.prefix_expansions > 0 {
            println!("  prefix expanded to {} terms", plan.prefix_expansions);
        }
    }
    Ok(())
}

fn cmd_watch(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["budget", "threads", "in"]);
    let store = Store::open(&args.root())?;
    watch::run(&store, index_options(&args))
}

fn cmd_serve(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["port", "in"]);
    let store = Store::open_existing(&args.root())?;
    let port = args.num("port").unwrap_or(7777) as u16;
    serve::run(store, port)
}

fn cmd_stats(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["in"]);
    let store = Store::open_existing(&args.root())?;
    let (docs, tokens) = store.corpus_stats()?;
    let segments = store.segments()?;
    let terms: i64 = store.conn.query_row("SELECT COUNT(*) FROM terms", [], |r| r.get(0))?;
    let distinct: i64 = store
        .conn
        .query_row("SELECT COUNT(DISTINCT term) FROM terms", [], |r| r.get(0))?;
    let postings: i64 = segments.iter().map(|s| s.postings).sum();
    let dead: i64 = store
        .conn
        .query_row("SELECT COUNT(*) FROM docs WHERE deleted = 1", [], |r| r.get(0))?;
    let index_bytes = store.index_bytes();

    println!("{}", store.root.display());
    println!("  {:<22} {}", "live documents", commas(docs));
    println!("  {:<22} {}", "tombstoned", commas(dead as u64));
    println!("  {:<22} {}", "tokens", commas(tokens));
    println!("  {:<22} {}", "distinct terms", commas(distinct as u64));
    println!("  {:<22} {}", "postings", commas(postings as u64));
    println!(
        "  {:<22} {} across {} segments",
        "term rows",
        commas(terms as u64),
        segments.len()
    );
    println!("  {:<22} {}", "index on disk", human_bytes(index_bytes));
    if postings > 0 {
        let posting_bytes: i64 = segments.iter().map(|s| s.bytes).sum();
        println!(
            "  {:<22} {:.2} bytes/posting (positions included)",
            "compression",
            posting_bytes as f64 / postings as f64
        );
    }
    if docs > 0 {
        println!("  {:<22} {:.0} tokens", "average document", tokens as f64 / docs as f64);
    }

    if !segments.is_empty() {
        println!();
        println!("  {:>4} {:>6} {:>10} {:>12} {:>12} {:>10}", "seg", "level", "docs", "terms", "postings", "bytes");
        for s in &segments {
            println!(
                "  {:>4} {:>6} {:>10} {:>12} {:>12} {:>10}",
                s.id,
                s.level,
                commas(s.docs as u64),
                commas(s.terms as u64),
                commas(s.postings as u64),
                human_bytes(s.bytes as u64)
            );
        }
    }
    Ok(())
}

fn cmd_merge(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["in"]);
    let store = Store::open_existing(&args.root())?;
    let before = store.segments()?.len();
    let stats = if args.has("all") {
        merge::merge_all(&store, true)?
    } else {
        merge::maybe_merge(&store, true)?
    };
    let after = store.segments()?.len();
    if stats.merges == 0 {
        println!("nothing to merge — {before} segments, all levels under the merge factor");
        return Ok(());
    }
    println!(
        "{} merges: {before} segments -> {after}, {} postings kept of {} read, peak working set {}",
        stats.merges,
        commas(stats.postings_out),
        commas(stats.postings_in),
        human_bytes(stats.peak_term_bytes as u64)
    );
    println!("  {} ms", stats.elapsed_ms);
    // VACUUM reclaims the pages the dropped segments used, but leaves the
    // write-ahead log holding every page it touched doing so. Without the
    // checkpoint the index looks three times bigger than it is.
    store.conn.execute_batch("VACUUM")?;
    store.conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")?;
    println!("  index now {}", human_bytes(store.index_bytes()));
    Ok(())
}

/// Query latency distribution over terms sampled from the index itself.
fn cmd_bench(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["runs", "in", "terms"]);
    let store = Store::open_existing(&args.root())?;
    let runs = args.num("runs").unwrap_or(200) as usize;

    // Sample across the frequency spectrum: rare terms exercise the B-tree
    // descent, common terms exercise the postings scan. Benchmarking only one
    // of the two is how people end up reporting misleading numbers.
    let mut sample: Vec<(String, i64)> = Vec::new();
    let mut stmt = store.conn.prepare(
        "SELECT term, SUM(df) AS d FROM terms GROUP BY term ORDER BY d DESC LIMIT 4000",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    for row in rows {
        sample.push(row?);
    }
    if sample.is_empty() {
        return Err("index is empty".into());
    }

    let pick: Vec<(String, i64)> = (0..runs)
        .map(|i| sample[(i * 7919) % sample.len()].clone())
        .collect();

    let searcher = Searcher::new(&store)?;
    let mut single = Vec::with_capacity(runs);
    let mut pairs = Vec::with_capacity(runs);
    let mut phrases = Vec::with_capacity(runs);

    for (i, (term, _)) in pick.iter().enumerate() {
        let other = &pick[(i + 1) % pick.len()].0;

        let q = query::parse(term);
        let t = Instant::now();
        let _ = searcher.search(&q, 10)?;
        single.push(t.elapsed().as_secs_f64() * 1000.0);

        let q = query::parse(&format!("{term} {other}"));
        let t = Instant::now();
        let _ = searcher.search(&q, 10)?;
        pairs.push(t.elapsed().as_secs_f64() * 1000.0);

        let q = query::parse(&format!("\"{term} {other}\""));
        let t = Instant::now();
        let _ = searcher.search(&q, 10)?;
        phrases.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    let (docs, _) = store.corpus_stats()?;
    println!(
        "{} documents, {} segments, {} warm queries per row",
        commas(docs),
        store.segments()?.len(),
        runs
    );
    println!();
    println!("  {:<16} {:>8} {:>8} {:>8} {:>8}", "query kind", "p50", "p90", "p99", "max");
    for (name, mut v) in [("single term", single), ("two terms", pairs), ("phrase", phrases)] {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {:<16} {:>7.2}ms {:>7.2}ms {:>7.2}ms {:>7.2}ms",
            name,
            percentile(&v, 0.50),
            percentile(&v, 0.90),
            percentile(&v, 0.99),
            v[v.len() - 1]
        );
    }

    let t = Instant::now();
    let cold = Store::open_existing(&args.root())?;
    let _ = Searcher::new(&cold)?;
    println!();
    println!(
        "  opening the index costs {:.2} ms — `ferret serve` pays it once, the CLI pays it per query",
        t.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Show the SQLite query plans behind the two hot paths, because "the ordering
/// comes from the B-tree, not a sort buffer" is a claim that should be
/// checkable rather than asserted.
fn cmd_explain(args: &[String]) -> Result<()> {
    let args = parse_args(args, &["in"]);
    let store = Store::open_existing(&args.root())?;

    println!("term lookup:");
    let mut stmt = store
        .conn
        .prepare("EXPLAIN QUERY PLAN SELECT df, postings FROM terms WHERE term = ?1")?;
    let rows = stmt.query_map(["example"], |r| r.get::<_, String>(3))?;
    for row in rows {
        println!("  {}", row?);
    }

    let ids: Vec<i64> = store.segments()?.iter().map(|s| s.id).collect();
    if ids.len() >= 2 {
        println!("merge scan:");
        for line in merge::explain_merge_plan(&store.conn, &ids)? {
            println!("  {line}");
        }
        println!("  (no SCAN ... USING TEMP B-TREE means the merge streams in term order)");
    }
    Ok(())
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}
