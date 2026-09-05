#!/usr/bin/env python3
"""Generate a synthetic corpus for benchmarking.

Real corpora have two properties that random text does not, and both change
the numbers a search engine reports:

  * Term frequencies follow Zipf's law. A uniform vocabulary gives every term
    the same posting list length, which hides exactly the tail-latency
    behaviour a benchmark is supposed to expose.
  * File sizes are heavy-tailed. A handful of large files dominate the index.

So the generator draws words from a Zipf distribution and file lengths from a
log-normal one, mixes prose with source-like files, and plants a few known
phrases so correctness can be checked against a known answer.

usage: gen_corpus.py OUTDIR --files 20000 [--seed 7]
"""

import argparse
import math
import os
import random
import zlib

SYLL_A = "ba ca da fa ga ha ka la ma na pa ra sa ta va za bi ci di fi gi hi ki li mi ni".split()
SYLL_B = "ble ck ct dge ft ght lm nd ng nk pt rk sh sk sp st th tion ment ness able".split()

CODE_TEMPLATE = """// {name}.rs — generated for corpus benchmarking
use std::collections::HashMap;

/// {doc}
pub struct {Type} {{
    {field_a}: HashMap<String, u64>,
    {field_b}: Vec<{Type}Entry>,
}}

impl {Type} {{
    pub fn new() -> Self {{
        Self {{ {field_a}: HashMap::new(), {field_b}: Vec::new() }}
    }}

    /// {doc2}
    pub fn {method}(&mut self, {arg}: &str) -> Option<u64> {{
        self.{field_a}.get({arg}).copied()
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;
    #[test]
    fn {method}_returns_none_when_absent() {{
        let mut subject = {Type}::new();
        assert!(subject.{method}("{arg}").is_none());
    }}
}}
"""


def build_vocabulary(size, rng):
    words = set()
    while len(words) < size:
        n = rng.randint(1, 3)
        w = "".join(rng.choice(SYLL_A) + rng.choice(SYLL_B) for _ in range(n))
        words.add(w[:14])
    return sorted(words)


def zipf_table(n, s=1.07):
    """Cumulative weights for a Zipf(s) distribution over n ranks."""
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    acc, out = 0.0, []
    for w in weights:
        acc += w / total
        out.append(acc)
    return out


def pick(vocab, cdf, rng):
    import bisect
    return vocab[bisect.bisect_left(cdf, rng.random())]


def make_pdf(text, style="literal"):
    """A minimal but valid single-page PDF with a Flate-compressed content
    stream, to exercise the extractor's real path rather than a toy one.

    Two styles, because they take different paths through the extractor:
    `literal` writes `(text) Tj`, and `hex_tj` writes `[<hex> -320 <hex>] TJ`,
    which is what most real generators emit and what forces the extractor to
    infer word gaps from kerning."""
    lines = [text[i:i + 70] for i in range(0, len(text), 70)][:40]
    content = "BT /F1 11 Tf 40 750 Td 14 TL\n"
    for line in lines:
        if style == "hex_tj":
            parts = line.split(" ")
            chunks = []
            for w in parts:
                if not w:
                    continue
                chunks.append("<" + w.encode("latin-1", "replace").hex() + ">")
                chunks.append("-320")
            content += "[" + " ".join(chunks[:-1]) + "] TJ T*\n"
        else:
            safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            content += f"({safe}) Tj T*\n"
    content += "ET"
    stream = zlib.compress(content.encode("latin-1", "replace"))

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" /Filter /FlateDecode >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--files", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--vocab", type=int, default=40000)
    ap.add_argument("--pdfs", type=int, default=50)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    vocab = build_vocabulary(args.vocab, rng)
    cdf = zipf_table(len(vocab))

    os.makedirs(args.outdir, exist_ok=True)
    total_bytes = 0

    for i in range(args.files):
        # Fan out over directories: 200 files per leaf keeps the tree
        # realistic and stops any single readdir from dominating the crawl.
        d = os.path.join(args.outdir, f"pkg{i // 4000:02d}", f"mod{(i // 200) % 20:02d}")
        os.makedirs(d, exist_ok=True)

        kind = rng.random()
        if kind < 0.35:
            # Source-like file, exercising the identifier splitter.
            name = pick(vocab, cdf, rng)
            body = CODE_TEMPLATE.format(
                name=name,
                Type=name.capitalize() + pick(vocab, cdf, rng).capitalize(),
                doc=" ".join(pick(vocab, cdf, rng) for _ in range(12)),
                doc2=" ".join(pick(vocab, cdf, rng) for _ in range(10)),
                field_a=pick(vocab, cdf, rng) + "_" + pick(vocab, cdf, rng),
                field_b=pick(vocab, cdf, rng) + "_count",
                method="get_" + pick(vocab, cdf, rng),
                arg=pick(vocab, cdf, rng),
            )
            path = os.path.join(d, f"{name}_{i}.rs")
        else:
            # Prose. Log-normal length: median ~400 words, long tail.
            words = int(math.exp(rng.gauss(math.log(400), 0.9)))
            words = max(30, min(words, 60000))
            paragraphs = []
            for _ in range(max(1, words // 90)):
                paragraphs.append(" ".join(pick(vocab, cdf, rng) for _ in range(90)))
            body = f"# note {i}\n\n" + "\n\n".join(paragraphs) + "\n"
            path = os.path.join(d, f"note_{i}.md")

        # A phrase with a known, exact answer, planted in 1 file in 5000.
        if i % 5000 == 0:
            body += "\nthe memory bounded merge writes one term at a time\n"

        with open(path, "w") as fh:
            fh.write(body)
        total_bytes += len(body)

    for i in range(args.pdfs):
        # Planted phrase goes first so it survives the one-page cap.
        text = "the memory bounded merge writes one term at a time "
        text += " ".join(pick(vocab, cdf, rng) for _ in range(400))
        blob = make_pdf(text, "hex_tj" if i % 4 == 0 else "literal")
        p = os.path.join(args.outdir, "papers", f"paper_{i:03d}.pdf")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(blob)
        total_bytes += len(blob)

    print(f"{args.files + args.pdfs} files, {total_bytes / 1e6:.1f} MB in {args.outdir}")


if __name__ == "__main__":
    main()
