//! A small PDF text extractor.
//!
//! Not a PDF renderer, and not trying to be. The index needs a bag of words
//! with roughly correct ordering; it does not need glyph positions, layout, or
//! reading order across columns.
//!
//! Strategy, and why:
//!
//! * **Scan for streams instead of parsing the cross-reference table.** The
//!   xref is the "correct" entry point, but it is also the first thing that
//!   breaks in real-world PDFs — incremental updates, linearised files, broken
//!   generators. Scanning every `stream ... endstream` span costs one linear
//!   pass and works on files that a strict parser rejects.
//! * **Inflate anything marked `/FlateDecode`,** which is essentially all
//!   modern content streams.
//! * **Interpret only the text-showing operators** (`Tj`, `TJ`, `'`, `"`) plus
//!   enough positioning operators (`Td`, `TD`, `T*`, `ET`) to know where to
//!   insert whitespace.
//!
//! Known limits, stated rather than hidden: text in fonts with a custom
//! encoding and no usable `ToUnicode` map comes out as mojibake, and so does
//! text stored as glyph outlines by some LaTeX and scanner pipelines. Scanned
//! image-only PDFs produce nothing at all, correctly — there is no text in
//! them. Those files are still catalogued, so they are never silently skipped.

use flate2::read::ZlibDecoder;
use std::io::Read;

/// Extract text from raw PDF bytes. Returns an empty string when the document
/// carries no extractable text (image-only scans, for example).
pub fn extract(bytes: &[u8]) -> String {
    let mut out = String::new();
    for stream in find_streams(bytes) {
        let data = match decode_stream(&stream) {
            Some(d) => d,
            None => continue,
        };
        // Content streams contain a BT/ET text block. Skip fonts, images and
        // metadata without paying for a full operator walk.
        if !contains(&data, b"BT") {
            continue;
        }
        extract_text_ops(&data, &mut out);
    }
    out
}

struct RawStream<'a> {
    dict: &'a [u8],
    data: &'a [u8],
}

/// Locate every `stream ... endstream` span and the dictionary preceding it.
fn find_streams(bytes: &[u8]) -> Vec<RawStream<'_>> {
    let mut streams = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = find(&bytes[i..], b"stream") {
        let kw = i + rel;
        // `endstream` also contains "stream"; skip those hits.
        if kw >= 3 && &bytes[kw - 3..kw] == b"end" {
            i = kw + 6;
            continue;
        }
        // The dictionary is whatever precedes the keyword, back to the object
        // header. Bounded look-back keeps this linear.
        let dict_start = kw.saturating_sub(2048);
        let dict = &bytes[dict_start..kw];

        // Per the spec the keyword is followed by CRLF or LF (not bare CR).
        let mut start = kw + 6;
        if bytes.get(start) == Some(&b'\r') {
            start += 1;
        }
        if bytes.get(start) == Some(&b'\n') {
            start += 1;
        }

        let end = match find(&bytes[start..], b"endstream") {
            Some(rel) => start + rel,
            None => break,
        };
        let mut data_end = end;
        while data_end > start && (bytes[data_end - 1] == b'\n' || bytes[data_end - 1] == b'\r') {
            data_end -= 1;
        }
        if data_end > start {
            streams.push(RawStream { dict, data: &bytes[start..data_end] });
        }
        i = end + 9;
    }
    streams
}

fn decode_stream(s: &RawStream<'_>) -> Option<Vec<u8>> {
    if contains(s.dict, b"/FlateDecode") {
        // Some writers emit a stray leading byte or a raw deflate stream, so
        // fall back rather than dropping the object.
        inflate(s.data)
            .or_else(|| s.data.first().and_then(|_| inflate(&s.data[1..])))
            .or_else(|| inflate_raw(s.data))
    } else if s.dict.contains(&b'/') && contains(s.dict, b"Decode") {
        // LZW, RunLength, DCT, JBIG2 and friends: not text, or not worth it.
        None
    } else {
        Some(s.data.to_vec())
    }
}

fn inflate(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut d = ZlibDecoder::new(data);
    match d.read_to_end(&mut out) {
        // Truncated streams still yield usable text up to the break.
        Ok(_) => Some(out),
        Err(_) if !out.is_empty() => Some(out),
        Err(_) => None,
    }
}

fn inflate_raw(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut d = flate2::read::DeflateDecoder::new(data);
    match d.read_to_end(&mut out) {
        Ok(_) => Some(out),
        Err(_) if !out.is_empty() => Some(out),
        Err(_) => None,
    }
}

/// Walk a content stream, collecting the arguments of text-showing operators.
fn extract_text_ops(data: &[u8], out: &mut String) {
    let mut i = 0usize;
    // Strings seen since the last operator, in order. Kerning large enough to
    // read as a word gap is pushed in as a literal space, which keeps the
    // spacing in the right *place* rather than at the end of the operator.
    let mut operands: Vec<String> = Vec::new();

    while i < data.len() {
        match data[i] {
            b'(' => {
                let (s, next) = read_literal_string(data, i + 1);
                operands.push(s);
                i = next;
            }
            b'<' if data.get(i + 1) != Some(&b'<') => {
                let (s, next) = read_hex_string(data, i + 1);
                operands.push(s);
                i = next;
            }
            b'-' | b'0'..=b'9' | b'.' => {
                let start = i;
                while i < data.len() && matches!(data[i], b'-' | b'+' | b'.' | b'0'..=b'9') {
                    i += 1;
                }
                if let Ok(text) = std::str::from_utf8(&data[start..i]) {
                    if let Ok(v) = text.parse::<f32>() {
                        // TJ kerning is in thousandths of an em; negative
                        // values move the pen forward. Anything past a third
                        // of an em is a word gap in practice.
                        if v <= -100.0 && operands.last().map_or(false, |s| s != " ") {
                            operands.push(" ".to_string());
                        }
                    }
                }
            }
            b'%' => {
                while i < data.len() && data[i] != b'\n' {
                    i += 1;
                }
            }
            c if c.is_ascii_alphabetic() || c == b'\'' || c == b'"' => {
                let start = i;
                while i < data.len()
                    && (data[i].is_ascii_alphanumeric()
                        || data[i] == b'*'
                        || data[i] == b'\''
                        || data[i] == b'"')
                {
                    i += 1;
                }
                let op = &data[start..i];
                match op {
                    b"Tj" | b"'" | b"\"" => {
                        if op != b"Tj" {
                            out.push('\n');
                        }
                        for s in operands.drain(..) {
                            out.push_str(&s);
                        }
                    }
                    b"TJ" => {
                        for s in operands.drain(..) {
                            out.push_str(&s);
                        }
                    }
                    b"Td" | b"TD" | b"T*" | b"ET" | b"BT" => {
                        out.push('\n');
                        operands.clear();
                    }
                    _ => operands.clear(),
                }
            }
            _ => i += 1,
        }
    }
}

/// PDF literal string: parenthesised, backslash escapes, balanced inner parens.
fn read_literal_string(data: &[u8], mut i: usize) -> (String, usize) {
    let mut depth = 1;
    let mut bytes: Vec<u8> = Vec::new();
    while i < data.len() {
        match data[i] {
            b'\\' => {
                i += 1;
                match data.get(i) {
                    Some(b'n') => bytes.push(b'\n'),
                    Some(b'r') => bytes.push(b'\r'),
                    Some(b't') => bytes.push(b'\t'),
                    Some(b'b') => bytes.push(8),
                    Some(b'f') => bytes.push(12),
                    Some(b'\n') => {}
                    Some(c @ b'0'..=b'7') => {
                        // Up to three octal digits.
                        let mut v = (c - b'0') as u32;
                        for _ in 0..2 {
                            match data.get(i + 1) {
                                Some(d @ b'0'..=b'7') => {
                                    v = v * 8 + (d - b'0') as u32;
                                    i += 1;
                                }
                                _ => break,
                            }
                        }
                        bytes.push((v & 0xff) as u8);
                    }
                    Some(c) => bytes.push(*c),
                    None => break,
                }
                i += 1;
            }
            b'(' => {
                depth += 1;
                bytes.push(b'(');
                i += 1;
            }
            b')' => {
                depth -= 1;
                i += 1;
                if depth == 0 {
                    break;
                }
                bytes.push(b')');
            }
            c => {
                bytes.push(c);
                i += 1;
            }
        }
    }
    (decode_pdf_bytes(&bytes), i)
}

/// PDF hex string: `<48656C6C6F>`.
fn read_hex_string(data: &[u8], mut i: usize) -> (String, usize) {
    let mut nibbles: Vec<u8> = Vec::new();
    while i < data.len() && data[i] != b'>' {
        let c = data[i];
        if let Some(v) = (c as char).to_digit(16) {
            nibbles.push(v as u8);
        }
        i += 1;
    }
    if i < data.len() {
        i += 1; // consume '>'
    }
    if nibbles.len() % 2 == 1 {
        nibbles.push(0);
    }
    let bytes: Vec<u8> = nibbles.chunks(2).map(|p| (p[0] << 4) | p[1]).collect();
    (decode_pdf_bytes(&bytes), i)
}

/// Interpret string bytes. UTF-16BE when the byte-order mark is present,
/// otherwise Latin-1, which agrees with WinAnsiEncoding across the ASCII range
/// and most of the accented range.
fn decode_pdf_bytes(bytes: &[u8]) -> String {
    if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
        let units: Vec<u16> = bytes[2..]
            .chunks_exact(2)
            .map(|p| u16::from_be_bytes([p[0], p[1]]))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    bytes.iter().map(|&b| b as char).collect()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    find(haystack, needle).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_literal_strings_with_escapes() {
        let (s, _) = read_literal_string(b"Hello \\(world\\) \\101)", 0);
        assert_eq!(s, "Hello (world) A");
    }

    #[test]
    fn reads_hex_strings() {
        let (s, _) = read_hex_string(b"48656C6C6F>", 0);
        assert_eq!(s, "Hello");
    }

    #[test]
    fn reads_utf16_hex_strings() {
        let (s, _) = read_hex_string(b"FEFF00480069>", 0);
        assert_eq!(s, "Hi");
    }

    #[test]
    fn pulls_text_out_of_a_content_stream() {
        let stream = b"BT /F1 12 Tf 72 720 Td (inverted index) Tj ET";
        let mut out = String::new();
        extract_text_ops(stream, &mut out);
        assert!(out.contains("inverted index"), "{out:?}");
    }

    #[test]
    fn tj_arrays_get_word_spacing() {
        let stream = b"BT [(post)-300(ings)] TJ ET";
        let mut out = String::new();
        extract_text_ops(stream, &mut out);
        let joined: String = out.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(joined.contains("post ings"), "{joined:?}");
    }
}
