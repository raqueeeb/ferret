//! A local HTTP shell over the index.
//!
//! Written against `std::net` rather than a web framework, for two reasons
//! that are not stubbornness. First, the whole point of `serve` is that the
//! searcher stays *warm* — an async runtime and a connection pool would add
//! moving parts around a process whose entire job is to hold one SQLite
//! connection and one tombstone set in memory. Second, a search tool for
//! private files should be auditable at a glance, and this is about 200 lines
//! of request parsing with no transitive dependencies.
//!
//! It binds to loopback only. There is no authentication, because there is no
//! remote surface to authenticate.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::query::{self, Searcher};
use crate::snippet;
use crate::store::Store;
use crate::Result;

const UI: &str = include_str!("ui.html");

pub fn run(store: Store, port: u16) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("cannot bind 127.0.0.1:{port}: {e}"))?;
    println!(
        "ferret serving http://127.0.0.1:{port}  (index: {})",
        store.root.display()
    );

    // One connection behind a mutex. SQLite connections are `Send` but not
    // `Sync`, and serialising queries is correct here anyway: the index is
    // read-mostly and a local UI issues one query at a time.
    let shared = Arc::new(Mutex::new(store));

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let shared = Arc::clone(&shared);
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, shared) {
                eprintln!("request failed: {e}");
            }
        });
    }
    Ok(())
}

fn handle(mut stream: TcpStream, shared: Arc<Mutex<Store>>) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(());
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("/");

    // Drain headers; nothing here needs them, but the socket must be read to
    // the end of the request before replying or clients see a reset.
    let mut header = String::new();
    loop {
        header.clear();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
    }

    if method != "GET" {
        return respond(&mut stream, 405, "text/plain", b"only GET is supported");
    }

    let (path, params) = split_query(target);
    match path {
        "/" | "/index.html" => respond(&mut stream, 200, "text/html; charset=utf-8", UI.as_bytes()),
        "/api/stats" => {
            let store = shared.lock().unwrap();
            let body = stats_json(&store)?;
            respond(&mut stream, 200, "application/json", body.as_bytes())
        }
        "/api/search" => {
            let q = params
                .iter()
                .find(|(k, _)| k == "q")
                .map(|(_, v)| v.clone())
                .unwrap_or_default();
            let limit = params
                .iter()
                .find(|(k, _)| k == "limit")
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .unwrap_or(25)
                .clamp(1, 200);
            let store = shared.lock().unwrap();
            match search_json(&store, &q, limit) {
                Ok(body) => respond(&mut stream, 200, "application/json", body.as_bytes()),
                Err(e) => respond(&mut stream, 500, "text/plain", e.to_string().as_bytes()),
            }
        }
        _ => respond(&mut stream, 404, "text/plain", b"not found"),
    }
}

fn respond(stream: &mut TcpStream, code: u16, mime: &str, body: &[u8]) -> Result<()> {
    let reason = match code {
        200 => "OK",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

fn stats_json(store: &Store) -> Result<String> {
    let (docs, tokens) = store.corpus_stats()?;
    let segments = store.segments()?;
    let terms: i64 = store
        .conn
        .query_row("SELECT COUNT(*) FROM terms", [], |r| r.get(0))
        .unwrap_or(0);
    Ok(format!(
        r#"{{"root":{},"docs":{},"tokens":{},"terms":{},"segments":{},"index_bytes":{}}}"#,
        json_string(&store.root.to_string_lossy()),
        docs,
        tokens,
        terms,
        segments.len(),
        store.index_bytes()
    ))
}

fn search_json(store: &Store, q: &str, limit: usize) -> Result<String> {
    let started = std::time::Instant::now();
    let parsed = query::parse(q);
    let searcher = Searcher::new(store)?;
    let (hits, plan) = searcher.search(&parsed, limit)?;

    // Terms to highlight: everything the parser produced, minus exclusions.
    let mut highlight: Vec<String> = Vec::new();
    for clause in &parsed.clauses {
        match clause {
            query::Clause::Term { text, occur } | query::Clause::Prefix { text, occur } => {
                if *occur != query::Occur::MustNot {
                    highlight.push(text.clone());
                }
            }
            query::Clause::Phrase { terms, occur } => {
                if *occur != query::Occur::MustNot {
                    highlight.extend(terms.iter().cloned());
                }
            }
        }
    }

    let mut items = Vec::with_capacity(hits.len());
    for hit in &hits {
        let full = store.root.join(&hit.path);
        let snip = snippet::for_file(&full, &highlight);
        items.push(format!(
            r#"{{"path":{},"score":{:.6},"line":{},"snippet":{}}}"#,
            json_string(&hit.path),
            hit.score,
            snip.line,
            json_string(&highlight_html(&snip))
        ));
    }

    let terms = plan
        .terms
        .iter()
        .map(|t| {
            format!(
                r#"{{"term":{},"df":{},"idf":{:.4},"segments":{},"bytes":{}}}"#,
                json_string(&t.term),
                t.df,
                t.idf,
                t.segments,
                t.bytes
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    Ok(format!(
        r#"{{"query":{},"took_ms":{:.3},"total":{},"hits":[{}],"plan":{{"terms":[{}],"segments":{},"postings_scanned":{},"candidates":{},"tombstones_skipped":{}}}}}"#,
        json_string(q),
        started.elapsed().as_secs_f64() * 1000.0,
        plan.candidates,
        items.join(","),
        terms,
        plan.segments,
        plan.postings_scanned,
        plan.candidates,
        plan.tombstones_skipped
    ))
}

/// Wrap matched ranges in `<mark>`, escaping everything else. Escaping happens
/// here rather than in the browser because the marks are byte offsets into the
/// raw snippet, and escaping first would invalidate them.
fn highlight_html(s: &snippet::Snippet) -> String {
    let mut out = String::with_capacity(s.text.len() + 32);
    let mut cursor = 0usize;
    for &(start, end) in &s.marks {
        if start < cursor || end > s.text.len() || start >= end {
            continue;
        }
        out.push_str(&html_escape(&s.text[cursor..start]));
        out.push_str("<mark>");
        out.push_str(&html_escape(&s.text[start..end]));
        out.push_str("</mark>");
        cursor = end;
    }
    out.push_str(&html_escape(&s.text[cursor..]));
    out
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn split_query(target: &str) -> (&str, Vec<(String, String)>) {
    match target.split_once('?') {
        None => (target, Vec::new()),
        Some((path, qs)) => {
            let params = qs
                .split('&')
                .filter(|p| !p.is_empty())
                .map(|pair| {
                    let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
                    (percent_decode(k), percent_decode(v))
                })
                .collect();
            (path, params)
        }
    }
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Read a whole request body. Unused by the current routes but kept honest:
/// without it a client sending a body would desynchronise the stream.
#[allow(dead_code)]
fn drain(reader: &mut impl Read, len: usize) {
    let mut buf = vec![0u8; len];
    let _ = reader.read_exact(&mut buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_query_strings() {
        let (path, params) = split_query("/api/search?q=hello%20world&limit=5");
        assert_eq!(path, "/api/search");
        assert_eq!(params[0], ("q".into(), "hello world".into()));
        assert_eq!(params[1], ("limit".into(), "5".into()));
    }

    #[test]
    fn decodes_quoted_phrases() {
        let (_, params) = split_query("/api/search?q=%22two+words%22");
        assert_eq!(params[0].1, "\"two words\"");
    }

    #[test]
    fn escapes_around_marks() {
        let s = snippet::Snippet {
            text: "a <b> c".to_string(),
            marks: vec![(2, 5)],
            line: 1,
        };
        assert_eq!(highlight_html(&s), "a <mark>&lt;b&gt;</mark> c");
    }
}
