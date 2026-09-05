# ferret

A local file search engine. Point it at a folder; it indexes text, code and PDFs, and answers ranked queries in well under a millisecond.

The inverted index is built from scratch — tokenizer, postings codec, BM25, segment merging, crash recovery. SQLite is used as a B-tree and a page cache. There is no Lucene, no Tantivy, and no FTS5 anywhere in this project.

```
$ ferret index ~/notes
$ ferret search "memory bounded merge"
$ ferret serve            # browser UI on localhost:7777
$ ferret watch            # keeps the index live as files change
```

---

## Measured on 100,300 files

A synthetic corpus with Zipf-distributed vocabulary and log-normally distributed file sizes: 456 MB of text across 100,000 markdown and Rust files plus 300 PDFs. All figures below were produced by `ferret index`, `ferret stats` and `ferret bench` on a **single-core** VM — the parallel extractor had one thread to work with, so the build throughput is a floor, not a ceiling.

### Index size against corpus size

| | |
|---|---|
| Corpus text | 456.0 MB |
| Index on disk, after merge | **168.2 MB — 36.9% of the corpus** |
| Postings alone | 126.4 MB |
| Documents / distinct terms / postings | 100,300 / 209,732 / 21,066,329 |
| **Bytes per posting, positions included** | **6.00** |

Six bytes per posting covers a document-id gap, a term frequency, a length norm, and every position that term occupies in that document — averaging 1.9 positions each. Delta encoding plus varints is doing the work: raw `u32` fields for the same data would cost 24 bytes.

The index is smaller than the corpus despite storing positions, because it stores each *distinct* term once per document rather than every occurrence, and never stores the document text. Snippets are produced by re-reading the top 25 files at display time.

### Query latency

300 warm queries per row, terms sampled across the whole frequency spectrum, top-10 retrieval with full BM25 scoring:

| query | p50 | p90 | p99 | max |
|---|---|---|---|---|
| single term | **0.15 ms** | 0.76 ms | 4.17 ms | 22.2 ms |
| two terms | 0.25 ms | 1.67 ms | 9.97 ms | 17.5 ms |
| phrase, positional | 0.16 ms | 1.15 ms | 9.93 ms | 11.6 ms |

Opening the index costs 0.30 ms, paid once by `ferret serve` and per-invocation by the CLI.

The tail is honest and explainable: p99 lands on terms appearing in half the corpus, where the cost is scanning a 50,000-entry postings list, not finding it. That is the case block-max WAND exists to fix, and it is the next thing I would build (see below).

**Segment count barely moves p50.** At 17 segments instead of 1, single-term p50 was 0.16 ms — unchanged — while p99 went from 4.2 ms to 9.7 ms. That flatness is a direct consequence of the primary key ordering described below.

### Memory

The memtable budget is a hard input, and the process respects it:

| budget | peak anonymous RSS, indexing | segments produced |
|---|---|---|
| 64 MB | 175 MB | 4 |
| 16 MB | 111 MB | 17 |

The gap between budget and RSS is SQLite's own 64 MB page cache plus allocator slack, not runaway index growth — halving the budget moves RSS by the amount you would expect. Anonymous RSS is reported deliberately; total RSS reads ~290 MB because a memory-mapped database counts file-backed pages that the kernel can reclaim at any time.

Merging 17 segments into 1 — 21 million postings, a 126 MB output segment — held a **peak per-term working set of 4.8 MB** and 112 MB process RSS, in 15.9 s.

### Against the tool it replaces

| | grep -r | ferret |
|---|---|---|
| term in 49,708 of 100,300 files | 510 ms | **6.3 ms** |
| term in 6 files | 400 ms | **0.32 ms** |
| ranking | none | BM25 |
| inside PDFs | invisible | indexed |

grep's cost is the corpus; ferret's cost is the answer. The last row is not a footnote — the six-document query above was matched entirely inside compressed PDF streams that grep cannot read at all.

---

## How it works

```
files ─► extract ─► tokenize ─► memtable (budgeted, in RAM)
                                    │ flush when full
                                    ▼
                        level 0 segments (immutable)
                                    │ 4 at a level
                                    ▼
                        level 1 ─► level 2 ─► ...
```

A log-structured merge tree. Writes are sequential appends into a memory-budgeted table; reads fan out across immutable segments; merges keep the fan-out bounded.

### Postings format

One term inside one segment:

```
repeat per document:
  varint  doc_gap       first entry is the raw id, then differences
  varint  tf
  u8      norm          quantised document length
  varint  pos_gap × tf  first is the raw position, then differences
```

**Why the norm is inline.** BM25 needs `len/avgdl` for every document it scores. Reading that from a document table is a random lookup per candidate, and it dominates tail latency. Instead each posting carries a one-byte quantised length — `log2(len+1) × 16`, accurate to 4.4% — so the scoring loop touches nothing outside the blob it is already scanning. BM25 multiplies length by `b = 0.75` and divides by the corpus average, so a 4% length error moves a score by well under 1%, far below the noise floor of ranking itself.

**Why positions are inline rather than in a parallel file.** Term-only queries pay a skip, but varints are self-delimiting so skipping is a byte walk with no decoding. In exchange, phrase queries need no second seek — and phrase queries are most of why anyone wants this over grep.

### The one schema decision that matters

```sql
CREATE TABLE terms (
    term TEXT, seg INTEGER, df INTEGER, ttf INTEGER, postings BLOB,
    PRIMARY KEY (term, seg)
) WITHOUT ROWID;
```

`WITHOUT ROWID` makes this a clustered B-tree keyed by the term itself, so a postings lookup is one descent plus a short sequential run — not one lookup per segment, and no secondary-index hop to fetch the blob. Ordering `term` before `seg` is why query cost tracks *term count* rather than *segment count*, and why p50 was identical at 1 and 17 segments.

The same ordering makes the merge free of a sort. `ferret explain` prints the plans so the claim is checkable rather than asserted:

```
term lookup:
  SEARCH terms USING PRIMARY KEY (term=?)
```

### The memory-bounded merge

The naive merge loads each input segment's terms into a map and merges in RAM, which needs memory proportional to the merged segment — eventually, to the whole corpus. That is the thing to avoid.

Because the table is clustered on `(term, seg)`, a single statement

```sql
SELECT term, seg, postings FROM terms WHERE seg IN (...) ORDER BY term, seg
```

is satisfied by one sequential B-tree walk with **no sort buffer**. The merge consumes it as a stream: gather the rows for one term, k-way merge those postings by document id, write one output row, drop everything, advance. Peak memory is the largest single term's postings — 4.8 MB on this corpus — regardless of index size.

The scan runs on a second connection so it reads a WAL snapshot from before the merge started while the writer inserts the new segment into the same table. Neither blocks the other. Crash safety falls out of the ordering: the new segment is one transaction and dropping the inputs is another, so an interrupted merge leaves the inputs intact.

Deletions are compacted here. A document edited or removed is tombstoned, its postings linger until the merge that rewrites them, and the merge simply declines to copy them forward.

### Incremental re-indexing

`ferret watch` sits on inotify (FSEvents on macOS, ReadDirectoryChangesW on Windows). Three details separate a demo from something you can leave running:

- **Ignore your own writes.** The index lives under the watched root, so every commit fires events that would retrigger indexing forever. Paths under `.ferret/` are dropped first.
- **Debounce.** One editor save is often a write, a rename, a chmod and a swap-file delete. A build touches thousands of files at once. Events collect for 400 ms of quiet before anything runs.
- **Coalesce.** The buffer is a set, so a file saved forty times during a build is indexed once.

A crawl skips files whose size and mtime match the catalogue. Content hashing would catch mtime-preserving edits, but it costs a full read of every file on every crawl, which defeats the purpose.

### Crash recovery

This one came out of an actual crash during benchmarking, not from a design document.

The two halves of a write have different durability. The catalogue commits every few thousand documents; the postings for those documents sit in the memtable until it fills. Kill the process in between and the catalogue claims a file is indexed while the inverted index has never heard of it — and because the crawl skips files whose size and mtime match, **that file stays unsearchable forever.**

Every flush now records the highest document id it made durable, inside the segment's own transaction. Anything above that watermark is tombstoned at startup and re-indexed. Tombstoned rather than deleted, deliberately: a partially written segment may hold postings pointing at those ids, and deleting the catalogue rows would orphan them beyond the reach of any query or merge. Tombstoning routes them through machinery that already exists.

### Tokenizer

Splits on non-alphanumerics, folds case (Unicode-aware), drops tokens over 64 bytes because those are hashes and minified bundles. Compound identifiers additionally emit their parts at the same position, so `parse_json_body` is findable as `json`, and `HTTPServerConfig` as `http`, `server` and `config`.

No stemming and no stopword list. Both are one-liners and both are wrong here: stemming mangles identifiers in a corpus that is half source code, and dropping stopwords breaks the phrase queries this index exists to support. Their absence is why BM25 uses the `ln(1 + …)` idf form — the textbook form goes negative for terms in more than half the corpus, letting a common word push documents *down* the ranking.

### PDF extraction

Written from scratch: scan for `stream … endstream` spans, inflate `/FlateDecode`, interpret the text-showing operators (`Tj`, `TJ`, `'`, `"`) and enough positioning to know where whitespace goes. Literal strings with octal escapes, hex strings, UTF-16BE byte-order marks, and TJ kerning arrays where a large negative kern means a word gap.

It skips the cross-reference table on purpose. The xref is the correct entry point and the first thing that breaks in real files — incremental updates, linearisation, broken generators. A linear scan works on files a strict parser rejects.

**Validated against poppler:** output is token-for-token identical to `pdftotext` on both literal-string and hex-plus-kerning PDFs.

---

## Query syntax

| | |
|---|---|
| `word` | contributes to the score |
| `+word` | required |
| `-word` | excluded |
| `"two words"` | phrase, matched by position |
| `word*` | prefix expansion, range scan on the clustered index |
| `ext:rs` `path:src` | filters, not scored |

Phrase matching shifts each term's positions by its offset in the phrase and processes terms rarest-first, so `"the quick brown fox"` starts from `fox` rather than `the`. After the shift, an occurrence is a position value present in every term's set, which reduces matching to sorted-list intersection.

---

## Tests

58 tests, `cargo test`. The ones that matter are the properties that only appear once the pieces are assembled:

- merging preserves result order **and scores to 1e-4**, and shrinks the segment count
- a 4 KB memtable budget gives the same answers as a 1 MB one
- merge peak working set stays below index size
- re-indexing a changed file retires its old terms and does not double-count it
- compaction actually drops postings for deleted documents
- documents lost before a flush are re-indexed rather than skipped, and corpus statistics stay consistent afterwards
- build directories and binary files are excluded; text beside them is not
- a second crawl re-indexes nothing

Three real bugs surfaced this way and are worth naming, since the point of the exercise is understanding rather than a green checkmark:

1. **Ranking was inverted.** A stray `.reverse()` on the heap sort put the worst-scoring documents first. Caught by a test asserting a short dense document outranks a long diluted one — precisely what BM25 length normalisation is for.
2. **A single segment holding tombstones could never be compacted.** `merge_all` bailed at `segments.len() < 2`, so deleted documents' postings were immortal.
3. **The memtable budget undercounted by 3x.** It charged encoded bytes while `Vec` doubles its capacity, so a 64 MB budget used 229 MB. Charging capacity brought it to 175 MB and made the budget mean something.

---

## Known limits

Stated rather than hidden.

- **The memtable is not queryable.** Documents become searchable when they flush. `ferret watch` flushes at the end of each debounced batch, so edits appear within about a second, but a bulk `ferret index` leaves the last partial memtable invisible until it commits.
- **PDF text with custom font encodings and no usable `ToUnicode` map comes out as mojibake**, and image-only scans yield nothing — correctly, since they contain no text. Those files are still catalogued, never silently dropped. Hard-wrapped PDF lines also split words across the break; real documents hyphenate, and de-hyphenation is not implemented.
- **Query execution is term-at-a-time** with a full accumulator. Fine at this corpus size; it is what the p99 column is paying for.
- **No stemming, no synonyms, no field weighting.** A title match scores the same as a footnote match.
- **`ferret serve` binds to loopback with no authentication**, which is appropriate for a tool that indexes private files and has no remote surface.

## What I would build next, in order

1. **Block-max WAND.** Store per-block maximum scores in the postings and skip blocks that cannot enter the top-k. This is the fix for the p99 column and needs skip pointers in the format — the format change is the work, not the algorithm.
2. **A queryable memtable**, so writes are visible before they flush.
3. **Content hashing behind a flag**, for editors and build tools that preserve mtime.
4. **A real `ToUnicode` CMap parser** for the PDFs that currently come out garbled.

---

## Why not just call Lucene

Lucene is better than this at nearly everything, and for production I would use it. The reasons not to here are specific:

- **The exercise is the point.** "I used a library that implements BM25" and "I implemented BM25, and here is why the idf form differs from the paper" are different sentences.
- **Dependency weight.** Four crates: `rusqlite`, `notify`, `walkdir`, `flate2`. The argument parser, HTTP server, JSON writer and PDF extractor are all hand-written. A tool that indexes a person's private files should be auditable in an afternoon.
- **A single file you can copy.** The whole index is one SQLite database. No segment directory to keep consistent, no lock files, no JVM.

Lucene would win on: block-max WAND and other skip-list optimisations, a far better analyzer chain, faceting, and a decade of correctness work on file formats. Those are real, and none of them are things I would have learned by importing them.

---

## Layout

```
src/
  codec.rs      varint and delta encoding, length-norm quantisation
  tokenize.rs   Unicode folding, identifier splitting
  memtable.rs   budgeted in-memory index
  segment.rs    postings encode/decode, segment writer
  merge.rs      streaming k-way merge, tiered compaction
  query.rs      parsing, BM25, phrase and prefix execution
  snippet.rs    best-window selection with highlighting
  extract/      file classification; pdf.rs is a from-scratch PDF text parser
  index.rs      crawler; parallel extraction, single-writer indexing
  watch.rs      inotify/FSEvents incremental re-indexing
  store.rs      SQLite schema, pragmas, crash recovery
  serve.rs      dependency-free HTTP server
  ui.html       browser UI
tests/          integration and crash-consistency tests
scripts/        corpus generator for benchmarking
```

## Commands

```
ferret index  [PATH] [--budget MB] [--threads N] [--hidden] [--no-merge] [-v]
ferret search QUERY [--in PATH] [-n N] [--plan] [--no-snippet]
ferret watch  [PATH]
ferret serve  [PATH] [--port N]
ferret stats  [PATH]
ferret merge  [PATH] [--all]
ferret bench  [PATH] [--runs N]
ferret explain [PATH]
```

Reproduce the benchmarks:

```
python3 scripts/gen_corpus.py /tmp/corpus --files 100000 --pdfs 300
cargo build --release
./target/release/ferret index /tmp/corpus -v
./target/release/ferret bench /tmp/corpus --runs 300
```
