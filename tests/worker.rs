//! The `ppocr-native` worker end to end, as bigocrpdf drives it: requests on
//! standard input, one JSON answer per line.
//!
//! It needs the exported plans, which the repository does not keep: run it
//! with `--include-ignored` and `PPOCR_NATIVE_PLANS` naming them. The text it
//! expects is RapidOCR's, recorded by `tools/make_fixtures.py`.

#[expect(dead_code, reason = "tests/parity.rs uses the rest of the helpers")]
mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use common::npy::{Npy, Result};
use ppocr_native::Image;
use serde_json::{json, Value};

/// A binary PPM of a BGR image, `width` by `height` pixels from its top left.
fn write_ppm(path: &Path, image: &Image, width: usize, height: usize) -> Result<()> {
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    for row in image.data.chunks_exact(image.width * 3).take(height) {
        for px in row[..width * 3].as_chunks::<3>().0 {
            bytes.extend_from_slice(&[px[2], px[1], px[0]]);
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// A new, empty directory in Cargo's private temporary directory for this
/// target, apart from other runs of the tests.
fn scratch_dir(name: &str) -> Result<PathBuf> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir(&dir)?;
    Ok(dir)
}

#[test]
#[ignore = "needs PPOCR_NATIVE_PLANS"]
fn worker_answers_each_request() -> Result<()> {
    let plans = std::env::var_os("PPOCR_NATIVE_PLANS").ok_or("PPOCR_NATIVE_PLANS is not set")?;
    let dir = scratch_dir("worker")?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let expected: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures.join("fixtures.json"))?)?;
    let expected = &expected["worker"];
    let page = Npy::read(&fixtures.join("page.npy"))?.image()?;
    let page_path = dir.join("page.ppm");
    write_ppm(&page_path, &page, page.width, page.height)?;
    let caption = Npy::read(&fixtures.join("caption.npy"))?.image()?;
    let caption_path = dir.join("caption.ppm");
    write_ppm(&caption_path, &caption, caption.width, caption.height)?;
    // The probability map is asked for at the detector's input size, whole
    // multiples of 32.
    let sized_path = dir.join("sized.ppm");
    write_ppm(&sized_path, &page, 192, 96)?;
    let png_path = dir.join("page.png");
    std::fs::write(&png_path, b"\x89PNG\r\n\x1a\n")?;
    let map_path: PathBuf = dir.join("map.f32");

    let mut child = Command::new(env!("CARGO_BIN_EXE_ppocr-native"))
        .args(["--persistent", "--threads", "2"])
        .env("PPOCR_NATIVE_PLANS", plans)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let requests = format!(
        "{page}\n{png}\n\ndetect\t{page}\t1536\nprobmap\t{sized}\t{map}\n{caption}\n",
        page = page_path.display(),
        caption = caption_path.display(),
        png = png_path.display(),
        sized = sized_path.display(),
        map = map_path.display(),
    );
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(requests.as_bytes())?;
    let output = child.wait_with_output()?;
    assert!(output.status.success());
    let answers: Vec<Value> = String::from_utf8(output.stdout)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(answers.len(), 6, "{answers:?}");

    // The worker inherits PPOCR_NATIVE_TIER, so it settles on this process's tier.
    assert_eq!(
        answers[0],
        json!({"ready": true, "tier": ppocr_native::engine::simd_tier()})
    );

    // RapidOCR's lines: the same boxes and texts, scores to the recogniser's
    // precision.
    let read = &answers[1];
    let page = &expected["page"];
    assert_eq!(read["boxes"], page["boxes"], "{read}");
    assert_eq!(read["txts"], page["txts"], "{read}");
    let scores = read["scores"].as_array().ok_or("scores")?;
    let page_scores = page["scores"].as_array().ok_or("scores")?;
    assert_eq!(scores.len(), page_scores.len());
    for (score, page_score) in scores.iter().zip(page_scores) {
        let (a, b) = (
            score.as_f64().ok_or("score")?,
            page_score.as_f64().ok_or("score")?,
        );
        assert!((a - b).abs() <= 1e-3, "{read}");
    }
    assert_eq!(read.get("vertical_reoriented"), None);
    let boxes = read["boxes"].as_array().ok_or("boxes")?;

    let error = answers[2]["error"].as_str().ok_or("error answer")?;
    assert!(error.starts_with("Could not load image"), "{error}");

    let detected = answers[3].as_object().ok_or("detect answer")?;
    assert_eq!(detected.keys().collect::<Vec<_>>(), ["boxes"]);
    // Lines scoring under --text-score are dropped from a read, never from detection.
    assert!(detected["boxes"]
        .as_array()
        .is_some_and(|d| d.len() >= boxes.len()));

    assert_eq!(answers[4], json!({"height": 96, "width": 192}));
    let map = std::fs::read(&map_path)?;
    assert_eq!(map.len(), 96 * 192 * 4);
    assert!(map
        .as_chunks::<4>()
        .0
        .iter()
        .all(|b| (0.0..=1.0).contains(&f32::from_le_bytes(*b))));

    // The caption reads upside down until the vertical pass turns it.
    let caption = &answers[5];
    assert_eq!(
        caption["txts"],
        json!([expected["caption"]["text"]]),
        "{caption}"
    );
    assert_eq!(caption["vertical_reoriented"], 1, "{caption}");

    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
#[ignore = "needs PPOCR_NATIVE_PLANS"]
fn full_resolution_strip_stays_bounded() -> Result<()> {
    let plans = std::env::var_os("PPOCR_NATIVE_PLANS").ok_or("PPOCR_NATIVE_PLANS is not set")?;
    let dir = scratch_dir("strip")?;
    // Its short side scaled up to 960 would make a 960x368640 input.
    let strip = dir.join("strip.ppm");
    write_ppm(&strip, &Image::zeroed(10, 4096), 10, 4096)?;
    let output = Command::new(env!("CARGO_BIN_EXE_ppocr-native"))
        .args(["--full-resolution", "--threads", "1"])
        .arg(&strip)
        .env("PPOCR_NATIVE_PLANS", plans)
        .output()?;
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout)?, "{\"boxes\": null}\n");
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
