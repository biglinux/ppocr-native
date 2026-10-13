//! Read text from images with PP-OCRv6, answering in JSON lines: bigocrpdf's
//! OCR worker.
//!
//! ```text
//! ppocr-native [OPTIONS] IMAGE...      one JSON line per image
//! ppocr-native --persistent [OPTIONS]  requests on standard input
//! ```
//!
//! Images are binary 8-bit PPM or PGM. An answer is `{"boxes": [...],
//! "txts": [...], "scores": [...]}`, with `"vertical_reoriented": N` when the
//! vertical pass replaced N lines, `{"boxes": null}` for a page without
//! text, or `{"error": "..."}`.
//!
//! The persistent worker first writes `{"ready": true, "tier": N}`, N being
//! the kernel tier it runs (see `PPOCR_NATIVE_TIER` below), or
//! `{"fatal": "..."}` and exits, then answers one line per request line:
//!
//! - `PATH`: the page's text, as above;
//! - `detect<TAB>PATH<TAB>MAX_SIDE`: `{"boxes": [...]}`, the detector's boxes
//!   alone on the page fitted within `MAX_SIDE` pixels;
//! - `probmap<TAB>PATH<TAB>OUT`: the detector's probability map of an image
//!   already at its input size, normalized in R, G, B order, written to `OUT`
//!   as little-endian `f32` rows; answers `{"height": H, "width": W}`.
//!
//! The worker exits when standard input closes.
//!
//! Options:
//!   --threads N          recognisers working on a page's lines (default: the
//!                        CPUs this process may use)
//!   --detect-only        load the detector alone; only `detect` and
//!                        `probmap` requests are answered
//!   --box-thresh F       mean probability a box needs (0.5)
//!   --unclip-ratio F     how far boxes grow (1.5)
//!   --text-score F       score a line needs to be kept (0.3)
//!   --limit-side-len N   longest page side, larger pages are scaled down (4096)
//!   --full-resolution    detect on the page as given instead of at most 2000
//!                        pixels, scaling only short sides up to 960 while
//!                        the result stays within the page limit squared
//!
//! `PPOCR_NATIVE_PLANS` names the directory holding the plans and the
//! character list (default `/usr/share/ppocr-native`); `PPOCR_NATIVE_TIER=0..3`
//! caps the kernel tier.

mod ocr;
mod pnm;

use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use ppocr_native::{DbOptions, Image, Limit, PageOptions, Quad};

use ocr::{Options, Page, Reader};

const DEFAULT_PLANS: &str = "/usr/share/ppocr-native";

struct Args {
    persistent: bool,
    detect_only: bool,
    threads: usize,
    options: Options,
    images: Vec<PathBuf>,
}

fn main() -> ExitCode {
    let args = match parse(std::env::args_os().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!(
                "ppocr-native: {message}\nusage: ppocr-native [--persistent] [OPTIONS] [IMAGE...]"
            );
            return ExitCode::from(2);
        }
    };
    let plans = std::env::var_os("PPOCR_NATIVE_PLANS")
        .map_or_else(|| PathBuf::from(DEFAULT_PLANS), PathBuf::from);
    let threads = if args.detect_only { 0 } else { args.threads };
    let loaded = Reader::load(&plans, threads, args.options);
    let result = match (loaded, args.persistent) {
        (Ok(mut reader), true) => serve(&mut reader),
        (Ok(mut reader), false) => {
            let mut out = io::stdout().lock();
            args.images.iter().try_for_each(|path| {
                writeln!(out, "{}", page_reply(&mut reader, path))?;
                out.flush()
            })
        }
        (Err(message), persistent) => {
            let key = if persistent { "fatal" } else { "error" };
            println!("{{{}: {}}}", json_str(key), json_str(&message));
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        // The reader went away: nobody is left to tell.
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ppocr-native: {e}");
            ExitCode::FAILURE
        }
    }
}

fn parse(mut argv: impl Iterator<Item = OsString>) -> Result<Args, String> {
    let mut args = Args {
        persistent: false,
        detect_only: false,
        threads: std::thread::available_parallelism().map_or(1, usize::from),
        options: Options {
            page: PageOptions::default(),
            limit: Limit::Max,
            db: DbOptions::default(),
            text_score: 0.3,
        },
        images: Vec::new(),
    };
    while let Some(arg) = argv.next() {
        let mut value = |name: &str| match argv.next().map(OsString::into_string) {
            Some(Ok(text)) => Ok(text),
            Some(Err(text)) => Err(format!("{name}: not a number: {}", text.display())),
            None => Err(format!("{name} needs a value")),
        };
        let number = |name: &str, text: String| -> Result<f64, String> {
            text.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .ok_or(format!("{name}: not a number: {text}"))
        };
        let name = arg.to_str().unwrap_or_default();
        match name {
            "--persistent" => args.persistent = true,
            "--detect-only" => args.detect_only = true,
            "--full-resolution" => args.options.limit = Limit::Min(960),
            "--threads" => {
                let text = value(name)?;
                args.threads = text
                    .parse()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or(format!("--threads: not a count: {text}"))?;
            }
            "--box-thresh" => args.options.db.box_thresh = number(name, value(name)?)?,
            "--unclip-ratio" => args.options.db.unclip_ratio = number(name, value(name)?)?,
            "--text-score" => args.options.text_score = number(name, value(name)?)?,
            "--limit-side-len" => {
                let text = value(name)?;
                args.options.page.max_side = text
                    .parse()
                    .ok()
                    .filter(|&n| n >= 32)
                    .ok_or(format!("--limit-side-len: not a length: {text}"))?;
            }
            "--" => args.images.extend(argv.by_ref().map(PathBuf::from)),
            option if option.starts_with("--") => return Err(format!("unknown option {option}")),
            _ => args.images.push(PathBuf::from(arg)),
        }
    }
    if args.persistent != args.images.is_empty() {
        return Err("give images, or --persistent and none".into());
    }
    Ok(args)
}

/// Answer requests from standard input until it closes.
fn serve(reader: &mut Reader) -> io::Result<()> {
    let mut out = io::stdout().lock();
    let tier = ppocr_native::engine::simd_tier();
    writeln!(out, "{{\"ready\": true, \"tier\": {tier}}}")?;
    out.flush()?;
    for line in io::stdin().lock().split(b'\n') {
        let line = line?;
        let line = String::from_utf8_lossy(&line);
        let request = line.trim();
        if request.is_empty() {
            continue;
        }
        let reply = match request.split('\t').collect::<Vec<_>>()[..] {
            ["detect", path, max_side] => detect_reply(reader, Path::new(path), max_side),
            ["probmap", path, output] => probmap_reply(reader, Path::new(path), Path::new(output)),
            _ => page_reply(reader, Path::new(request)),
        };
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

fn load(path: &Path) -> Result<Image, String> {
    let failed = |e: String| format!("Could not load image: {}: {e}", path.display());
    let bytes = std::fs::read(path).map_err(|e| failed(e.to_string()))?;
    pnm::decode(bytes).map_err(failed)
}

fn error_reply(message: &str) -> String {
    format!("{{\"error\": {}}}", json_str(message))
}

fn page_reply(reader: &mut Reader, path: &Path) -> String {
    match load(path).and_then(|image| reader.read(&image)) {
        Ok(page) => page_json(&page),
        Err(message) => error_reply(&message),
    }
}

fn detect_reply(reader: &mut Reader, path: &Path, max_side: &str) -> String {
    let Some(max_side) = max_side.parse().ok().filter(|&n: &usize| n >= 32) else {
        return error_reply(&format!("not a side length: {max_side}"));
    };
    match load(path).and_then(|image| reader.boxes(&image, max_side)) {
        Ok(boxes) => {
            let mut json = String::from("{\"boxes\": ");
            push_boxes(&mut json, boxes.iter());
            json.push('}');
            json
        }
        Err(message) => error_reply(&message),
    }
}

fn probmap_reply(reader: &mut Reader, path: &Path, output: &Path) -> String {
    let image = match load(path) {
        Ok(image) => image,
        Err(message) => return error_reply(&message),
    };
    let written = reader.probability_map(&image).and_then(|map| {
        let bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(output, bytes).map_err(|e| format!("{}: {e}", output.display()))
    });
    match written {
        Ok(()) => format!(
            "{{\"height\": {}, \"width\": {}}}",
            image.height, image.width
        ),
        Err(message) => error_reply(&message),
    }
}

/// The page as `json.dumps` writes the worker's dict: Python's separators,
/// floats as `repr` writes them, non-ASCII escaped.
fn page_json(page: &Page) -> String {
    if page.lines.is_empty() {
        return "{\"boxes\": null}".into();
    }
    let mut json = String::from("{\"boxes\": ");
    push_boxes(&mut json, page.lines.iter().map(|line| &line.quad));
    json.push_str(", \"txts\": [");
    for (i, line) in page.lines.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        json.push_str(&json_str(&line.text));
    }
    json.push_str("], \"scores\": [");
    for (i, line) in page.lines.iter().enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        let _ = write!(json, "{:?}", line.score);
    }
    json.push(']');
    if page.reoriented > 0 {
        let _ = write!(json, ", \"vertical_reoriented\": {}", page.reoriented);
    }
    json.push('}');
    json
}

fn push_boxes<'a>(json: &mut String, boxes: impl Iterator<Item = &'a Quad>) {
    json.push('[');
    for (i, quad) in boxes.enumerate() {
        if i > 0 {
            json.push_str(", ");
        }
        // `{:?}` of the f64 is Python's `repr` for these values: the shortest
        // digits that read back, with ".0" on whole numbers.
        let [a, b, c, d] = quad.map(|[x, y]| (f64::from(x), f64::from(y)));
        let _ = write!(
            json,
            "[[{:?}, {:?}], [{:?}, {:?}], [{:?}, {:?}], [{:?}, {:?}]]",
            a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1
        );
    }
    json.push(']');
}

/// A JSON string as Python's `json.dumps` writes it with `ensure_ascii`.
fn json_str(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                for unit in c.encode_utf16(&mut [0; 2]) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    use super::{json_str, parse};

    #[test]
    fn strings_escape_like_python() {
        // json.dumps('a"\\\n\x01é😀\x7f')
        assert_eq!(
            json_str("a\"\\\n\u{1}é😀\u{7f}"),
            "\"a\\\"\\\\\\n\\u0001\\u00e9\\ud83d\\ude00\\u007f\""
        );
    }

    #[test]
    fn options_are_checked() {
        let args = |list: &[&str]| parse(list.iter().map(OsString::from));
        assert!(args(&["--persistent", "--threads", "3"]).is_ok_and(|a| a.threads == 3));
        for bad in [
            &["--persistent", "--threads", "0"][..],
            &["--persistent", "--box-thresh", "nan"],
            &["--persistent", "--limit-side-len", "16"],
            &["--persistent", "--bogus"],
            &["--persistent", "page.ppm"],
            &[],
        ] {
            assert!(args(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn non_utf8_paths_are_images() {
        let latin1 = || OsString::from_vec(b"caf\xe9.ppm".to_vec());
        assert!(parse([latin1()].into_iter()).is_ok_and(|a| a.images == [latin1()]));
        let threads = ["--persistent".into(), "--threads".into(), latin1()];
        assert!(parse(threads.into_iter()).is_err());
    }
}
