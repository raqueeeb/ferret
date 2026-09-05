//! Tokenizer.
//!
//! The corpus is prose *and* source code, which pulls in two directions. Prose
//! wants aggressive folding; code wants `HttpRequest` to be findable as
//! `http`, `request`, or `httprequest`. The compromise here:
//!
//! 1. Split on anything that is not alphanumeric.
//! 2. Lowercase (Unicode-aware, so `Straße` and `ÉCOLE` behave).
//! 3. If a token is a compound identifier, also emit its parts at the *same*
//!    position range, so `parse_json_body` matches a search for `json`.
//! 4. Drop tokens longer than 64 bytes — those are base64 payloads, minified
//!    bundles and hashes, and they are pure index bloat with no query value.
//!
//! Deliberately absent: stemming and stopword removal. See README, both are
//! defensible one-liners to add and both are wrong for a corpus that is half
//! source code.

/// Longest token kept. Anything past this is a hash or a minified blob.
pub const MAX_TOKEN_LEN: usize = 64;

/// A token and its ordinal position in the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub pos: u32,
}

/// Tokenize `text`, calling `sink` for each token in position order.
///
/// Streaming rather than returning a `Vec` because a 4MB log file produces
/// ~600k tokens and materialising them all doubles peak memory for no reason.
pub fn tokenize<F: FnMut(&str, u32)>(text: &str, mut sink: F) {
    let mut pos: u32 = 0;
    let mut buf = String::with_capacity(32);

    for chunk in split_raw(text) {
        buf.clear();
        for c in chunk.chars() {
            for lc in c.to_lowercase() {
                buf.push(lc);
            }
        }
        if buf.is_empty() || buf.len() > MAX_TOKEN_LEN {
            continue;
        }
        sink(&buf, pos);

        // Sub-tokens share the whole token's position. Two consequences worth
        // knowing: a phrase query for "parse json" will match `parse_json`
        // (arguably correct), and term frequency for the compound is counted
        // once per part. Both are acceptable; the alternative is a positional
        // gap scheme that complicates phrase matching for little gain.
        let parts = split_identifier(chunk);
        if parts.len() > 1 {
            for part in parts {
                let lowered: String = part.chars().flat_map(|c| c.to_lowercase()).collect();
                if !lowered.is_empty() && lowered.len() <= MAX_TOKEN_LEN && lowered != buf {
                    sink(&lowered, pos);
                }
            }
        }
        pos += 1;
    }
}

/// Convenience wrapper for tests and small inputs.
pub fn tokens(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    tokenize(text, |t, p| {
        out.push(Token { text: t.to_string(), pos: p })
    });
    out
}

/// Query-side tokenization: same rules, but sub-tokens are dropped so that
/// searching for `HttpRequest` does not silently become three OR'd terms.
pub fn query_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in split_raw(text) {
        let lowered: String = chunk.chars().flat_map(|c| c.to_lowercase()).collect();
        if !lowered.is_empty() && lowered.len() <= MAX_TOKEN_LEN {
            out.push(lowered);
        }
    }
    out
}

/// Split on non-alphanumeric boundaries. `_` is a separator here but
/// [`split_identifier`] sees the original chunk, so `snake_case` still yields
/// its parts.
fn split_raw(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_matches('_'))
        .filter(|s| !s.is_empty())
}

/// Break an identifier into its parts: `snake_case`, `camelCase`,
/// `PascalCase`, `SCREAMING_SNAKE`, and the acronym boundary in `HTTPServer`
/// (-> `http`, `server`). Digits attach to the preceding run: `utf8Decode`
/// gives `utf8`, `decode`.
fn split_identifier(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    for segment in s.split('_').filter(|x| !x.is_empty()) {
        let bytes: Vec<(usize, char)> = segment.char_indices().collect();
        let mut start = 0usize;
        for i in 1..bytes.len() {
            let (idx, cur) = bytes[i];
            let (_, prev) = bytes[i - 1];
            let boundary = if prev.is_lowercase() && cur.is_uppercase() {
                // fooBar
                true
            } else if prev.is_uppercase() && cur.is_uppercase() {
                // HTTPServer: break before the S, which needs one char lookahead
                bytes.get(i + 1).map_or(false, |(_, next)| next.is_lowercase())
            } else if prev.is_numeric() != cur.is_numeric() {
                // utf8Decode / decode2
                cur.is_alphabetic() && prev.is_numeric()
            } else {
                false
            };
            if boundary {
                parts.push(&segment[start..idx]);
                start = idx;
            }
        }
        parts.push(&segment[start..]);
    }
    parts.retain(|p| !p.is_empty());
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(s: &str) -> Vec<String> {
        tokens(s).into_iter().map(|t| t.text).collect()
    }

    #[test]
    fn splits_plain_prose() {
        assert_eq!(texts("The quick, brown fox!"), vec!["the", "quick", "brown", "fox"]);
    }

    #[test]
    fn positions_are_sequential() {
        let t = tokens("alpha beta gamma");
        assert_eq!(t.iter().map(|x| x.pos).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn emits_compound_and_parts() {
        let t = texts("parse_json_body");
        assert!(t.contains(&"parse_json_body".to_string()));
        assert!(t.contains(&"json".to_string()));
        assert!(t.contains(&"body".to_string()));
    }

    #[test]
    fn handles_camel_and_acronyms() {
        let t = texts("HTTPServerConfig");
        assert!(t.contains(&"http".to_string()), "{t:?}");
        assert!(t.contains(&"server".to_string()), "{t:?}");
        assert!(t.contains(&"config".to_string()), "{t:?}");
    }

    #[test]
    fn subtokens_share_a_position() {
        let t = tokens("readFile now");
        let read = t.iter().find(|x| x.text == "read").unwrap();
        let file = t.iter().find(|x| x.text == "file").unwrap();
        let now = t.iter().find(|x| x.text == "now").unwrap();
        assert_eq!(read.pos, file.pos);
        assert_eq!(now.pos, read.pos + 1);
    }

    #[test]
    fn drops_absurd_tokens() {
        let long = "a".repeat(200);
        assert!(texts(&long).is_empty());
    }

    #[test]
    fn unicode_folds() {
        assert_eq!(texts("ÉCOLE Straße"), vec!["école", "straße"]);
    }

    #[test]
    fn query_side_keeps_compounds_whole() {
        assert_eq!(query_tokens("HttpRequest timeout"), vec!["httprequest", "timeout"]);
    }
}
