//! Turning a file on disk into indexable text.

pub mod pdf;

use std::path::Path;

/// Files above this are read but truncated. A 200MB log file has nothing in
/// its tail that a search needs, and reading it whole stalls the crawler.
pub const MAX_BYTES: usize = 16 * 1024 * 1024;

/// How many leading bytes are inspected to decide whether a file is text.
const SNIFF: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Text,
    Pdf,
    Skip,
}

/// Extensions that are always treated as text. Everything else without an
/// extension gets sniffed; everything else with a known-binary extension is
/// skipped without being opened.
const TEXT_EXT: &[&str] = &[
    // prose and markup
    "txt", "md", "markdown", "rst", "org", "adoc", "tex", "bib", "log",
    // structured data
    "json", "jsonl", "ndjson", "yaml", "yml", "toml", "ini", "cfg", "conf", "env", "properties",
    "csv", "tsv", "xml", "svg", "html", "htm", "xhtml", "css", "scss", "sass", "less",
    // code
    "rs", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "go", "py", "pyi", "rb", "java", "kt", "kts",
    "scala", "swift", "m", "mm", "cs", "fs", "php", "pl", "pm", "lua", "r", "jl", "hs", "ml",
    "elm", "ex", "exs", "erl", "clj", "cljs", "dart", "zig", "nim", "v", "sv", "vhd",
    "js", "jsx", "mjs", "cjs", "ts", "tsx", "vue", "svelte",
    "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd", "make", "mk", "cmake", "gradle",
    "sql", "graphql", "gql", "proto", "thrift", "dockerfile", "gitignore", "patch", "diff",
];

const BINARY_EXT: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "webp", "tiff", "avif", "heic",
    "mp3", "mp4", "wav", "flac", "ogg", "avi", "mov", "mkv", "webm", "m4a",
    "zip", "gz", "bz2", "xz", "zst", "7z", "rar", "tar", "jar", "war",
    "exe", "dll", "so", "dylib", "bin", "o", "a", "obj", "class", "pyc", "wasm",
    "ttf", "otf", "woff", "woff2", "eot",
    "db", "sqlite", "sqlite3", "mdb", "pack", "idx", "lock",
    // Office XML: a zip container, worth a real reader rather than a bad one.
    "docx", "xlsx", "pptx", "odt", "ods", "odp", "doc", "xls", "ppt",
];

/// Filenames without a useful extension that are still worth indexing.
const TEXT_NAMES: &[&str] = &[
    "makefile", "dockerfile", "readme", "license", "licence", "changelog", "authors",
    "contributing", "notice", "todo", "cargo.lock", "gemfile", "rakefile", "procfile",
];

pub fn classify(path: &Path) -> Kind {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    if name.starts_with('.') && !TEXT_NAMES.contains(&name.trim_start_matches('.')) {
        // Dotfiles are usually config, which is worth indexing, but dot-
        // directories were already pruned by the crawler. Index them.
        if name.len() > 1 {
            return Kind::Text;
        }
    }

    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    if ext == "pdf" {
        return Kind::Pdf;
    }
    if BINARY_EXT.contains(&ext.as_str()) {
        return Kind::Skip;
    }
    if TEXT_EXT.contains(&ext.as_str()) {
        return Kind::Text;
    }
    if TEXT_NAMES.contains(&name.as_str()) {
        return Kind::Text;
    }
    // Unknown extension: read it and let the binary sniffer decide.
    Kind::Text
}

/// Read and extract. `Ok(None)` means "correctly skipped", which is different
/// from an error and is not reported as a failure.
pub fn extract_file(path: &Path) -> std::io::Result<Option<String>> {
    match classify(path) {
        Kind::Skip => Ok(None),
        Kind::Pdf => {
            let bytes = read_capped(path)?;
            Ok(Some(pdf::extract(&bytes)))
        }
        Kind::Text => {
            let bytes = read_capped(path)?;
            if looks_binary(&bytes) {
                return Ok(None);
            }
            // Lossy on purpose: a stray invalid byte in an otherwise useful
            // source file should not cost the whole file.
            Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
        }
    }
}

fn read_capped(path: &Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.take(MAX_BYTES as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

/// A NUL byte in the first few KB is the classic and still the most reliable
/// binary signal. UTF-16 text trips it, which is a known and accepted miss.
fn looks_binary(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(SNIFF)];
    if head.contains(&0) {
        return true;
    }
    // A high proportion of control characters means binary even without a NUL.
    let control = head
        .iter()
        .filter(|&&b| b < 9 || (b > 13 && b < 32))
        .count();
    !head.is_empty() && control * 100 / head.len() > 5
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn classifies_by_extension() {
        assert_eq!(classify(&PathBuf::from("a/b/main.rs")), Kind::Text);
        assert_eq!(classify(&PathBuf::from("paper.pdf")), Kind::Pdf);
        assert_eq!(classify(&PathBuf::from("logo.png")), Kind::Skip);
        assert_eq!(classify(&PathBuf::from("Makefile")), Kind::Text);
    }

    #[test]
    fn detects_binary_payloads() {
        assert!(looks_binary(&[0x7f, b'E', b'L', b'F', 0, 0, 0]));
        assert!(!looks_binary(b"fn main() { println!(\"hi\"); }"));
    }
}
