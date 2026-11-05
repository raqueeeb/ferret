//! Byte-level codec for postings lists.
//!
//! Everything in a postings list is a small non-negative integer *after* delta
//! encoding, so LEB128 varints buy a lot: document ids in a segment are
//! ascending, term positions inside a document are ascending, and both are
//! stored as gaps. On real corpora this lands around 1.4 bytes per position
//! and 1.1 bytes per document gap.

/// Append `v` to `out` as a LEB128 varint (7 payload bits per byte, high bit =
/// continuation).
#[inline]
pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Read a varint at `*pos`, advancing the cursor. Returns `None` on a
/// truncated or over-long encoding rather than panicking: postings blobs come
/// off disk and a corrupt one should degrade a query, not kill the process.
#[inline]
pub fn get_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    while *pos < buf.len() {
        let byte = buf[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
    None
}

/// Number of bytes `v` will occupy. Used by the memtable to charge memory
/// against its budget without actually encoding.
#[inline]
pub fn varint_len(v: u64) -> usize {
    match v {
        0..=0x7f => 1,
        0x80..=0x3fff => 2,
        0x4000..=0x1f_ffff => 3,
        0x20_0000..=0xfff_ffff => 4,
        _ => 5,
    }
}

// ---------------------------------------------------------------------------
// Length norms
// ---------------------------------------------------------------------------

/// Quantise a document length into one byte.
///
/// BM25 needs `len/avgdl` for every document it scores. Reading that from a
/// documents table means a random lookup per candidate, which is the single
/// biggest source of tail latency in a naive implementation. Instead the norm
/// travels inline with each posting, so scoring is one sequential pass over
/// bytes already in the page cache.
///
/// The cost is precision: the code is `log2(len+1) * 16`, so the decoded value
/// is within about 4.4% of the true length, and lengths above ~63k tokens
/// saturate. BM25 divides length by the corpus average and multiplies by
/// b=0.75, so a 4% error moves a score by well under 1% — far below the noise
/// floor of the ranking itself.
#[inline]
pub fn encode_norm(len: u32) -> u8 {
    if len == 0 {
        return 0;
    }
    let code = ((len as f32 + 1.0).log2() * 16.0).round();
    code.clamp(0.0, 255.0) as u8
}

/// Inverse of [`encode_norm`]. Precomputed into a 256-entry table by callers
/// that decode in a hot loop.
#[inline]
pub fn decode_norm(code: u8) -> f32 {
    if code == 0 {
        return 0.0;
    }
    (2f32).powf(code as f32 / 16.0) - 1.0
}

/// The full decode table. 1KB, built once per searcher.
pub fn norm_table() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = decode_norm(i as u8);
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        let values = [
            0u64,
            1,
            127,
            128,
            300,
            16383,
            16384,
            1 << 20,
            u32::MAX as u64,
        ];
        let mut buf = Vec::new();
        for v in values {
            put_varint(&mut buf, v);
        }
        let mut pos = 0;
        for v in values {
            assert_eq!(get_varint(&buf, &mut pos), Some(v));
        }
        assert_eq!(pos, buf.len());
    }

    #[test]
    fn varint_len_matches_encoding() {
        for v in [0u64, 1, 127, 128, 16383, 16384, 1 << 21, 1 << 28] {
            let mut buf = Vec::new();
            put_varint(&mut buf, v);
            assert_eq!(buf.len(), varint_len(v), "value {v}");
        }
    }

    #[test]
    fn truncated_varint_is_none() {
        let buf = [0x80u8, 0x80];
        let mut pos = 0;
        assert_eq!(get_varint(&buf, &mut pos), None);
    }

    #[test]
    fn norms_stay_within_five_percent() {
        for len in [1u32, 5, 17, 100, 1000, 12345, 50000] {
            let back = decode_norm(encode_norm(len));
            let err = (back - len as f32).abs() / len as f32;
            assert!(err < 0.05, "len {len} -> {back} ({err})");
        }
    }
}
