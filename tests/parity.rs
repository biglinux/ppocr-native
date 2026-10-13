//! Does each stage still compute what RapidOCR and OpenCV compute?
//!
//! The fixtures were written by `tools/make_fixtures.py` from the libraries
//! this crate reproduces (OpenCV 5.0, pyclipper 1.4, RapidOCR 3.9, numpy
//! 2.5), on synthetic inputs and on parts of a real page. Where the crate
//! matches bit for bit the test demands it; the two known gaps, OpenCV's
//! tie-breaking inside `minAreaRect` and its bicubic rounding, are bounded
//! by what `tools/check_parity.py` measured on 30 full pages.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::npy::{Npy, Result};
use ppocr_native::{
    boxes_from_prob, crop_text, ctc_decode, ctc_decode_probs, det_input, det_size, find_contours,
    min_area_rect, pad_vertically, page_lines, prepare_page, rec_input, resize_linear, unclip,
    DbOptions, Dictionary, Image, Limit, PageOptions, PreparedPage, Quad,
};
use serde_json::Value;

fn path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fixtures() -> Result<Value> {
    Ok(serde_json::from_str(&fs::read_to_string(path(
        "fixtures.json",
    ))?)?)
}

fn npy(name: &Value) -> Result<Npy> {
    Npy::read(&path(name.as_str().ok_or("file name")?))
}

fn points(v: &Value) -> Vec<[i32; 2]> {
    serde_json::from_value(v.clone()).expect("integer points")
}

fn quad(v: &Value) -> Quad {
    serde_json::from_value(v.clone()).expect("four float points")
}

#[test]
fn resize_matches_opencv() -> Result<()> {
    let f = fixtures()?;
    let image = Npy::read(&path("noise.npy"))?.image()?;
    for case in f["resize"].as_array().ok_or("resize")? {
        let (w, h) = (
            case["width"].as_u64().ok_or("w")? as usize,
            case["height"].as_u64().ok_or("h")? as usize,
        );
        let expected = npy(&case["file"])?.image()?;
        assert_eq!(
            resize_linear(&image, w, h),
            expected,
            "40x28 noise to {w}x{h}"
        );
    }
    Ok(())
}

#[test]
fn det_input_matches_rapidocr() -> Result<()> {
    let f = fixtures()?;
    let image = Npy::read(&path("noise.npy"))?.image()?;
    let mean: [f64; 3] = serde_json::from_value(f["det_input"]["mean"].clone())?;
    let std: [f64; 3] = serde_json::from_value(f["det_input"]["std"].clone())?;
    let (w, h) = det_size(image.width, image.height, Limit::Max).ok_or("size")?;
    assert_eq!((w, h), (32, 32), "40x28 snaps to 32x32");
    let expected = npy(&f["det_input"]["file"])?.f32s()?;
    assert!(
        det_input(&image, w, h, mean, std) == expected,
        "detector tensor differs"
    );
    Ok(())
}

#[test]
fn contours_match_opencv() -> Result<()> {
    let f = fixtures()?;
    for (i, case) in f["contours"]
        .as_array()
        .ok_or("contours")?
        .iter()
        .enumerate()
    {
        let mask = npy(&case["file"])?;
        let [h, w] = mask.shape[..] else {
            return Err("mask shape".into());
        };
        let expected: Vec<Vec<[i32; 2]>> = case["contours"]
            .as_array()
            .ok_or("list")?
            .iter()
            .map(points)
            .collect();
        assert_eq!(find_contours(&mask.data, w, h), expected, "mask {i}");
    }
    Ok(())
}

#[test]
fn db_boxes_match_rapidocr() -> Result<()> {
    let f = fixtures()?;
    let case = &f["boxes"];
    let prob = npy(&case["file"])?;
    let [h, w] = prob.shape[..] else {
        return Err("map shape".into());
    };
    let [dest_w, dest_h]: [usize; 2] = serde_json::from_value(case["dest"].clone())?;
    let expected: Vec<Quad> = case["boxes"]
        .as_array()
        .ok_or("boxes")?
        .iter()
        .map(quad)
        .collect();
    let boxes = boxes_from_prob(&prob.f32s()?, w, h, dest_w, dest_h, &DbOptions::default());
    assert_eq!(boxes, expected);
    Ok(())
}

#[test]
fn min_area_rect_matches_opencv() -> Result<()> {
    let f = fixtures()?;
    for case in f["min_area_rect"].as_array().ok_or("cases")? {
        assert_eq!(
            min_area_rect(&points(&case["points"])).points(),
            quad(&case["box"]),
            "{}",
            case["points"]
        );
    }
    Ok(())
}

#[test]
fn unclip_matches_pyclipper() -> Result<()> {
    let f = fixtures()?;
    for case in f["unclip"].as_array().ok_or("cases")? {
        assert_eq!(
            unclip(&quad(&case["quad"]), 1.5),
            points(&case["ring"]),
            "{}",
            case["quad"]
        );
    }
    Ok(())
}

#[test]
fn crops_match_opencv() -> Result<()> {
    let f = fixtures()?;
    let page = Npy::read(&path("page.npy"))?.image()?;
    for case in f["crops"].as_array().ok_or("crops")? {
        let expected = npy(&case["file"])?.image()?;
        let got = crop_text(&page, &quad(&case["quad"]));
        assert_eq!(
            (got.width, got.height),
            (expected.width, expected.height),
            "{}",
            case["quad"]
        );
        // OpenCV 5 rounds bicubic sums near a half its own way: one level,
        // on 0.003 % of the reference pages' crop pixels.
        let off: Vec<i16> = got
            .data
            .iter()
            .zip(&expected.data)
            .map(|(&a, &b)| i16::from(a) - i16::from(b))
            .filter(|&d| d != 0)
            .collect();
        assert!(
            off.iter().all(|d| d.abs() <= 1),
            "{}: {off:?}",
            case["quad"]
        );
        assert!(
            off.len() * 100 <= got.data.len(),
            "{}: {} values differ",
            case["quad"],
            off.len()
        );
    }
    Ok(())
}

#[test]
fn rec_input_matches_rapidocr() -> Result<()> {
    let f = fixtures()?;
    for case in f["rec_input"].as_array().ok_or("cases")? {
        let crop = npy(&case["crop"])?.image()?;
        let expected = npy(&case["input"])?;
        let (tensor, width) = rec_input(&crop);
        assert_eq!(
            width, expected.shape[3],
            "width for a {}x{} crop",
            crop.width, crop.height
        );
        assert!(
            tensor == expected.f32s()?,
            "tensor for a {}x{} crop",
            crop.width,
            crop.height
        );
    }
    Ok(())
}

#[test]
fn ctc_matches_rapidocr() -> Result<()> {
    let f = fixtures()?;
    let dictionary = Dictionary::from_metadata(&fs::read_to_string(path("characters.txt"))?);
    assert_eq!(dictionary.len(), 18710, "classes of PP-OCRv6 small");
    for case in f["ctc"].as_array().ok_or("cases")? {
        let stem = case["stem"].as_str().ok_or("stem")?;
        let best: Vec<u32> = Npy::read(&path(&format!("{stem}_argmax.npy")))?
            .i32s()?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let prob = Npy::read(&path(&format!("{stem}_maxprob.npy")))?.f32s()?;
        let (text, score) = ctc_decode(&best, &prob, &dictionary);
        assert_eq!(text, case["text"].as_str().ok_or("text")?, "{stem}");
        assert_eq!(score, case["score"].as_f64().ok_or("score")?, "{stem}");
    }
    Ok(())
}

#[test]
fn dictionary_splits_like_python() {
    let d = Dictionary::from_metadata("a\r\nb\x1cc\u{2028}\nd\n");
    let classes: Vec<&str> = (0..d.len()).map(|i| d.class(i)).collect();
    assert_eq!(classes, ["blank", "a", "b", "c", "", "d", " "]);
}

#[test]
fn ctc_probs_take_nan_like_numpy() {
    // RapidOCR's decoder on these rows gives ('bc', nan).
    let d = Dictionary::from_metadata("a\nb\nc\nd\n");
    let nan = f32::NAN;
    let probs = [
        [0.1, 0.2, nan, 0.3, 0.0, 0.0],
        [nan, nan, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
    ];
    let (text, score) = ctc_decode_probs(probs.as_flattened(), 6, &d);
    assert_eq!(text, "bc");
    assert!(score.is_nan());
}

#[test]
fn page_lines_match_worker() -> Result<()> {
    let f = fixtures()?;
    let case = &f["page_lines"];
    let [height, width]: [usize; 2] = serde_json::from_value(case["image_hw"].clone())?;
    let page = PreparedPage {
        image: Image::zeroed(0, 0),
        ratio_h: case["ratio_h"].as_f64().ok_or("ratio")?,
        ratio_w: case["ratio_w"].as_f64().ok_or("ratio")?,
        pad_top: 0,
    };
    let boxes: Vec<Quad> = case["boxes"]
        .as_array()
        .ok_or("boxes")?
        .iter()
        .map(quad)
        .collect();
    let reads: Vec<(String, f64)> = serde_json::from_value(case["reads"].clone())?;
    let lines = page_lines(&page, width, height, &boxes, reads, 0.3);
    let expected = &case["expected"];
    let quads: Vec<Quad> = expected["boxes"]
        .as_array()
        .ok_or("boxes")?
        .iter()
        .map(quad)
        .collect();
    let texts: Vec<String> = serde_json::from_value(expected["txts"].clone())?;
    let scores: Vec<f64> = serde_json::from_value(expected["scores"].clone())?;
    assert_eq!(lines.iter().map(|l| l.quad).collect::<Vec<_>>(), quads);
    assert_eq!(
        lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>(),
        texts
    );
    assert_eq!(lines.iter().map(|l| l.score).collect::<Vec<_>>(), scores);
    Ok(())
}

#[test]
fn padding_and_score_filter_follow_rapidocr() {
    // RapidOCR's apply_vertical_padding with width_height_ratio -1 and
    // min_height 30, measured: (height, width) -> top band.
    for (h, w, pad) in [(100, 100, 0), (20, 100, 20), (10, 200, 25)] {
        let (_, top) = pad_vertically(&Image::zeroed(w, h), -1.0, 30);
        assert_eq!(top, pad, "{w}x{h}");
    }
    // `score < text_score` is false for NaN, so RapidOCR keeps the line.
    let page = PreparedPage {
        image: Image::zeroed(0, 0),
        ratio_h: 1.0,
        ratio_w: 1.0,
        pad_top: 0,
    };
    let quad = [[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [0.0, 9.0]];
    let reads = vec![("ab".to_owned(), f64::NAN), ("cd".to_owned(), 0.1)];
    let lines = page_lines(&page, 10, 10, &[quad, quad], reads, 0.3);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].text, "ab");
}

#[test]
fn empty_pages_have_no_size() {
    // RapidOCR divides by the empty side and raises.
    let options = PageOptions::default();
    for (w, h) in [(100, 0), (0, 100), (0, 0)] {
        let page = Image::zeroed(w, h);
        assert!(prepare_page(&page, &options).is_none(), "{w}x{h}");
        assert_eq!(det_size(w, h, Limit::Min(32)), None, "{w}x{h}");
        assert_eq!(det_size(w, h, Limit::Max), None, "{w}x{h}");
    }
}

#[test]
fn thin_strips_stay_bounded() {
    let options = PageOptions::default();
    // Padded to 122880x30720, past 4096 squared: no page.
    assert!(prepare_page(&Image::zeroed(4096, 1), &options).is_none());
    // A real line keeps RapidOCR's size: 6144x32, padded to 1536 rows.
    let line = prepare_page(&Image::zeroed(4096, 20), &options).expect("a line page");
    assert_eq!(
        (line.image.width, line.image.height, line.pad_top),
        (6144, 1536, 752)
    );
}
