//! Snippet generation.
//!
//! Snippets are built by re-reading the source file at display time, not from
//! stored text. That is a deliberate trade: it keeps the index roughly 3-5x
//! smaller than one that stores documents, at the cost of one file read per
//! *displayed* result — twenty reads for a page of results, not twenty
//! thousand for the candidate set. Since only the top-k are ever rendered,
//! the cost is bounded by the page size and nothing else.

use std::path::Path;

/// Target snippet width in characters.
const WINDOW: usize = 240;

#[derive(Debug, Clone)]
pub struct Snippet {
    pub text: String,
    /// Byte ranges within `text` that matched, for highlighting.
    pub marks: Vec<(usize, usize)>,
    /// 1-based line number where the snippet starts, or 0 if unknown.
    pub line: usize,
}

impl Snippet {
    pub fn empty() -> Snippet {
        Snippet {
            text: String::new(),
            marks: Vec::new(),
            line: 0,
        }
    }
}

/// Build a snippet for `path` around occurrences of `terms`.
pub fn for_file(path: &Path, terms: &[String]) -> Snippet {
    let text = match crate::extract::extract_file(path) {
        Ok(Some(t)) => t,
        _ => return Snippet::empty(),
    };
    from_text(&text, terms)
}

/// Build a snippet from text already in hand.
pub fn from_text(text: &str, terms: &[String]) -> Snippet {
    if terms.is_empty() || text.is_empty() {
        return head(text);
    }
    let hits = find_matches(text, terms);
    if hits.is_empty() {
        return head(text);
    }

    // Pick the window containing the most *distinct* terms, breaking ties on
    // raw match count. Rewarding distinct terms is what stops a snippet from
    // being 240 characters of the single most common query word.
    let mut best = (0usize, 0usize, 0usize); // (distinct, count, start hit)
    for i in 0..hits.len() {
        let window_end = hits[i].0 + WINDOW;
        let mut seen = 0u64;
        let mut count = 0usize;
        for hit in &hits[i..] {
            if hit.1 > window_end {
                break;
            }
            seen |= 1u64 << (hit.2 % 64);
            count += 1;
        }
        let distinct = seen.count_ones() as usize;
        if (distinct, count) > (best.0, best.1) {
            best = (distinct, count, i);
        }
    }

    let anchor = hits[best.2].0;
    let start = floor_char_boundary(text, anchor.saturating_sub(WINDOW / 4));
    let end = ceil_char_boundary(text, (start + WINDOW).min(text.len()));
    let slice = &text[start..end];

    let mut marks = Vec::new();
    for &(s, e, _) in &hits {
        if s >= start && e <= end {
            marks.push((s - start, e - start));
        }
    }

    let line = text[..start].bytes().filter(|&b| b == b'\n').count() + 1;
    let (collapsed, marks) = collapse_whitespace(slice, &marks);
    Snippet {
        text: collapsed,
        marks,
        line,
    }
}

fn head(text: &str) -> Snippet {
    let end = ceil_char_boundary(text, text.len().min(WINDOW));
    let (collapsed, marks) = collapse_whitespace(&text[..end], &[]);
    Snippet {
        text: collapsed,
        marks,
        line: 1,
    }
}

/// Case-insensitive whole-token matches, with the index of the term matched.
fn find_matches(text: &str, terms: &[String]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut i = 0usize;

    // One pass over the text, comparing each token against the term list. The
    // term list is single digits long, so a per-token linear scan beats
    // building an automaton.
    while i <= bytes.len() {
        let at_boundary = i == bytes.len() || !is_word_byte(bytes[i]);
        if at_boundary {
            if i > start {
                let token = &text[start..i];
                for (ti, term) in terms.iter().enumerate() {
                    if token.len() == term.len() && token.eq_ignore_ascii_case(term) {
                        out.push((start, i, ti));
                        break;
                    }
                    // Also mark the term when it appears inside a compound
                    // identifier, matching what the tokenizer indexed.
                    if token.len() > term.len() && token.to_lowercase().contains(term.as_str()) {
                        out.push((start, i, ti));
                        break;
                    }
                }
            }
            start = i + 1;
        }
        i += 1;
    }
    out
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b >= 0x80
}

/// Fold runs of whitespace to single spaces, keeping mark offsets correct.
fn collapse_whitespace(s: &str, marks: &[(usize, usize)]) -> (String, Vec<(usize, usize)>) {
    let mut out = String::with_capacity(s.len());
    let mut map: Vec<usize> = Vec::with_capacity(s.len() + 1);
    let mut last_space = true; // trims leading whitespace

    for (i, c) in s.char_indices() {
        while map.len() <= i {
            map.push(out.len());
        }
        if c.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(c);
            last_space = false;
        }
    }
    while map.len() <= s.len() {
        map.push(out.len());
    }

    let mapped = marks
        .iter()
        .filter_map(|&(a, b)| {
            let (na, nb) = (*map.get(a)?, *map.get(b)?);
            if nb > na {
                Some((na, nb))
            } else {
                None
            }
        })
        .collect();
    (out.trim_end().to_string(), mapped)
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centres_on_the_densest_window() {
        let filler = "lorem ipsum ".repeat(60);
        let text = format!("{filler} the inverted index and the postings list {filler}");
        let s = from_text(&text, &["inverted".into(), "postings".into()]);
        assert!(s.text.contains("inverted"), "{}", s.text);
        assert!(!s.marks.is_empty());
    }

    #[test]
    fn marks_land_on_the_right_bytes() {
        let s = from_text("alpha beta gamma", &["beta".into()]);
        let (a, b) = s.marks[0];
        assert_eq!(&s.text[a..b], "beta");
    }

    #[test]
    fn survives_multibyte_text() {
        let text = "café ☕ inverted index café ☕".repeat(20);
        let s = from_text(&text, &["inverted".into()]);
        assert!(s.text.contains("inverted"));
    }

    #[test]
    fn empty_terms_give_a_head_snippet() {
        let s = from_text("hello world", &[]);
        assert_eq!(s.text, "hello world");
    }
}
