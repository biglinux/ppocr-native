//! Runs every stage of the crate on the inputs recorded by
//! `tools/dump_reference.py` and writes what it computed, for
//! `tools/check_parity.py` to compare.
//!
//! Each stage starts from the reference's own input to that stage, so a
//! difference points at one stage; the `chain` entries run detection end to
//! end from the recorded probability map instead.
//!
//! Usage:
//!     parity_dump REFERENCE_DIR IMAGE_DIR DICTIONARY_TXT OUT_DIR [PAGE_ID...]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ppocr_native::{
    boxes_from_prob, contour_candidate, contour_mean, crop_text, ctc_decode, ctc_decode_probs,
    det_input, det_size, find_contours, fit_page, mini_box, order_and_clip, pad_vertically,
    page_lines, prepare_page, rec_input, sort_boxes, unclip, Candidate, DbOptions, Dictionary,
    Image, Limit, PageOptions, Quad,
};
use serde_json::{json, Value};

#[path = "../tests/npy/mod.rs"]
mod npy;

use npy::{Npy, Result};

/// IEEE half precision to `f32`, exactly.
fn f16s(array: &Npy) -> Result<Vec<f32>> {
    if array.descr != "<f2" {
        return Err(format!("expected <f2, got {}", array.descr).into());
    }
    Ok(array
        .data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half_to_f32(u16::from_le_bytes(*b)))
        .collect())
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    match exp {
        // Zero and subnormals: frac * 2^-24, exact in f32.
        0 => f32::from_bits(sign | 0x3380_0000) * frac as f32,
        31 => f32::from_bits(sign | 0x7f80_0000 | (frac << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13)),
    }
}

fn write_npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> Result<()> {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    let shape = if dims.len() == 1 {
        format!("({},)", dims[0])
    } else {
        format!("({})", dims.join(", "))
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut file = fs::File::create(path)?;
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&u16::try_from(header.len())?.to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    file.write_all(data)?;
    Ok(())
}

fn write_image(path: &Path, image: &Image) -> Result<()> {
    write_npy(path, "|u1", &[image.height, image.width, 3], &image.data)
}

fn write_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_npy(path, "<f4", shape, &bytes)
}

/// A binary PPM, as RGB, turned into BGR.
fn read_ppm(path: &Path) -> Result<Image> {
    let bytes = fs::read(path)?;
    let mut fields = Vec::new();
    let mut at = 0;
    while fields.len() < 4 {
        while bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        if bytes[at] == b'#' {
            while bytes[at] != b'\n' {
                at += 1;
            }
            continue;
        }
        let begin = at;
        while !bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        fields.push(std::str::from_utf8(&bytes[begin..at])?.to_owned());
    }
    if fields[0] != "P6" || fields[3] != "255" {
        return Err(format!("{}: not an 8-bit P6", path.display()).into());
    }
    let (w, h): (usize, usize) = (fields[1].parse()?, fields[2].parse()?);
    let mut data = bytes[at + 1..at + 1 + w * h * 3].to_vec();
    for px in data.as_chunks_mut::<3>().0 {
        px.swap(0, 2);
    }
    Ok(Image::new(w, h, data))
}

fn as_quad(v: &Value) -> Result<Quad> {
    let mut quad = [[0f32; 2]; 4];
    for (i, p) in quad.iter_mut().enumerate() {
        for (j, c) in p.iter_mut().enumerate() {
            *c = v[i][j].as_f64().ok_or("box coordinate")? as f32;
        }
    }
    Ok(quad)
}

fn as_points(v: &Value) -> Result<Vec<[i32; 2]>> {
    v.as_array()
        .ok_or("point list")?
        .iter()
        .map(|p| {
            Ok([
                p[0].as_i64().ok_or("x")? as i32,
                p[1].as_i64().ok_or("y")? as i32,
            ])
        })
        .collect()
}

fn hw(v: &Value) -> Result<(usize, usize)> {
    Ok((
        v[0].as_u64().ok_or("h")? as usize,
        v[1].as_u64().ok_or("w")? as usize,
    ))
}

fn candidate_json(c: &Candidate) -> Value {
    match c {
        Candidate::Small => json!({"status": "small"}),
        Candidate::LowScore(s) => json!({"status": "low_score", "score": s}),
        Candidate::SmallAfterUnclip(s) => json!({"status": "small_after_unclip", "score": s}),
        Candidate::Kept(s, b) => json!({"status": "kept", "score": s, "box": b}),
    }
}

fn page(ref_dir: &Path, image_dir: &Path, dictionary: &Dictionary, out: &Path) -> Result<()> {
    let id = ref_dir
        .file_name()
        .ok_or("page dir")?
        .to_string_lossy()
        .into_owned();
    let p: Value = serde_json::from_str(&fs::read_to_string(ref_dir.join("page.json"))?)?;
    fs::create_dir_all(out.join("rec"))?;
    let mut report = serde_json::Map::new();

    // Stage 1: page fitting, from the original image.
    let original = read_ppm(&image_dir.join(format!("{id}.ppm")))?;
    let options = PageOptions::default();
    let prepared = prepare_page(&original, &options).ok_or("page too thin")?;
    let fitted = fit_page(&original, options.min_side, options.max_side).ok_or("page too thin")?;
    if (fitted.0.width, fitted.0.height) != (original.width, original.height) {
        write_image(&out.join("preprocessed.npy"), &fitted.0)?;
    }
    // RapidOCR records the page before padding; the padding shows in the
    // detector input.
    report.insert(
        "preprocess".into(),
        json!({"output_hw": [fitted.0.height, fitted.0.width],
               "ratio_h": fitted.1, "ratio_w": fitted.2, "pad_top": prepared.pad_top}),
    );

    // Stage 3: detector input, from the reference's fitted page with the
    // crate's padding; the reference does not record the padded page.
    let det = &p["det"];
    let fitted_ref = match p["preprocess"]["file"].as_str() {
        Some(f) => Npy::read(&ref_dir.join(f))?.image()?,
        None => original.clone(),
    };
    let (stage1, _) = pad_vertically(&fitted_ref, options.width_height_ratio, options.min_height);
    let (w, h) = det_size(stage1.width, stage1.height, Limit::Max).ok_or("det size")?;
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    write_image(
        &out.join("det_resized.npy"),
        &ppocr_native::resize_linear(&stage1, w, h),
    )?;
    write_f32(
        &out.join("det_input.npy"),
        &[1, 3, h, w],
        &det_input(&stage1, w, h, mean, std),
    )?;

    // Stage 5: boxes, from the reference probability map.
    let prob_npy = Npy::read(&ref_dir.join(det["prob_file"].as_str().ok_or("prob")?))?;
    let (map_h, map_w) = (prob_npy.shape[0], prob_npy.shape[1]);
    let prob = prob_npy.f32s()?;
    let (dest_h, dest_w) = hw(&det["dest_hw"])?;
    let db = DbOptions::default();
    let mask: Vec<u8> = prob.iter().map(|&v| u8::from(v > db.thresh)).collect();
    write_npy(&out.join("det_bitmap.npy"), "|u1", &[map_h, map_w], &mask)?;
    let own_contours = find_contours(&mask, map_w, map_h);
    let own: Vec<Value> = own_contours
        .iter()
        .map(|c| {
            json!({"points": c,
                   "candidate": candidate_json(&contour_candidate(&prob, map_w, map_h, c, dest_w, dest_h, &db))})
        })
        .collect();

    let flat =
        Npy::read(&ref_dir.join(det["contours_file"].as_str().ok_or("contours")?))?.i32s()?;
    let mut bounds = vec![0usize];
    for len in det["contour_lengths"].as_array().ok_or("lengths")? {
        bounds.push(bounds.last().copied().unwrap_or(0) + len.as_u64().ok_or("length")? as usize);
    }
    let mut isolated = Vec::new();
    for c in det["contours"].as_array().ok_or("contours")? {
        let i = c["index"].as_u64().ok_or("index")? as usize;
        let contour: Vec<[i32; 2]> = flat[bounds[i] * 2..bounds[i + 1] * 2]
            .as_chunks::<2>()
            .0
            .to_vec();
        let (quad, side) = mini_box(&contour);
        let mut entry = json!({
            "mini_box": quad, "mini_sside": side,
            "score": contour_mean(&prob, map_w, map_h, &contour),
            "candidate": candidate_json(&contour_candidate(&prob, map_w, map_h, &contour, dest_w, dest_h, &db)),
        });
        if c.get("unclipped").is_some() {
            let ref_quad = as_quad(&c["mini_box"])?;
            entry["unclipped"] = json!(unclip(&ref_quad, db.unclip_ratio));
            let (grown, side) = mini_box(&as_points(&c["unclipped"])?);
            entry["unclip_mini_box"] = json!(grown);
            entry["unclip_mini_sside"] = json!(side);
        }
        isolated.push(entry);
    }

    // Stages 6 and 7: filtering from the reference's scaled boxes, sorting
    // from its kept boxes, and the whole chain from the probability map.
    let mut filtered = Vec::new();
    for f in det["filter"].as_array().ok_or("filter")? {
        let b = as_quad(&f["input_box"])?.map(|[x, y]| [x as i32, y as i32]);
        filtered.push(json!(order_and_clip(&b, dest_w, dest_h)));
    }
    let mut kept: Vec<Quad> = Vec::new();
    for f in det["filter"].as_array().ok_or("filter")? {
        if f["kept"].as_bool() == Some(true) {
            kept.push(as_quad(&f["clipped"])?);
        }
    }
    sort_boxes(&mut kept);
    let started = Instant::now();
    let chain_boxes = boxes_from_prob(&prob, map_w, map_h, dest_w, dest_h, &db);
    let boxes_time = started.elapsed();
    report.insert(
        "det".into(),
        json!({"resized_hw": [h, w], "contours": own, "isolated": isolated, "filtered": filtered,
               "sorted": kept, "chain_boxes": chain_boxes}),
    );

    // Stage 8: crops from the reference's sorted boxes.
    for c in p["crops"].as_array().ok_or("crops")? {
        let i = c["index"].as_u64().ok_or("index")? as usize;
        let crop = crop_text(&stage1, &as_quad(&c["points"])?);
        write_image(&out.join(format!("rec/{i:04}_crop.npy")), &crop)?;
    }
    let started = Instant::now();
    let chain_inputs: Vec<(Vec<f32>, usize)> = chain_boxes
        .iter()
        .map(|quad| rec_input(&crop_text(&stage1, quad)))
        .collect();
    let crops_time = started.elapsed();
    eprintln!(
        "{id}: {} boxes in {:.1} ms, their crops and recogniser inputs in {:.1} ms",
        chain_boxes.len(),
        boxes_time.as_secs_f64() * 1e3,
        crops_time.as_secs_f64() * 1e3
    );
    for (i, (input, width)) in chain_inputs.iter().enumerate() {
        let path = out.join(format!("rec/c{i:04}_input.npy"));
        write_f32(&path, &[1, 3, 48, *width], input)?;
    }

    // Stages 9 and 10: recogniser input from the reference's crops, decoding
    // from its recorded outputs.
    let mut reads = Vec::new();
    for r in p["rec"].as_array().ok_or("rec")? {
        let i = r["crop"].as_u64().ok_or("crop")? as usize;
        let crop = Npy::read(&ref_dir.join(format!("rec/{i:04}_crop.npy")))?.image()?;
        let (input, width) = rec_input(&crop);
        write_f32(
            &out.join(format!("rec/{i:04}_input.npy")),
            &[1, 3, 48, width],
            &input,
        )?;
        let best: Vec<u32> = Npy::read(&ref_dir.join(r["argmax_file"].as_str().ok_or("argmax")?))?
            .i32s()?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        let prob =
            Npy::read(&ref_dir.join(r["maxprob_file"].as_str().ok_or("maxprob")?))?.f32s()?;
        let (text, score) = ctc_decode(&best, &prob, dictionary);
        let mut entry = json!({"crop": i, "input_width": width, "text": text, "score": score});
        if let Some(f) = r.get("probs_file").and_then(Value::as_str) {
            let probs = f16s(&Npy::read(&ref_dir.join(f))?)?;
            let (text, score) = ctc_decode_probs(&probs, dictionary.len(), dictionary);
            entry["probs_text"] = json!(text);
            entry["probs_score"] = json!(score);
        }
        reads.push((i, entry));
    }
    reads.sort_by_key(|(i, _)| *i);

    // Stage 11: the final lines, from the reference's boxes and readings.
    let ref_boxes = det["boxes"]
        .as_array()
        .ok_or("boxes")?
        .iter()
        .map(as_quad)
        .collect::<Result<Vec<_>>>()?;
    let ref_reads = p["rec"]
        .as_array()
        .ok_or("rec")?
        .iter()
        .map(|r| {
            Ok((
                r["crop"].as_u64().ok_or("crop")?,
                r["text"].as_str().ok_or("text")?.to_owned(),
                r["score"].as_f64().ok_or("score")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut ordered: Vec<(String, f64)> = vec![(String::new(), 0.0); ref_boxes.len()];
    for (i, text, score) in ref_reads {
        ordered[i as usize] = (text, score);
    }
    let lines = page_lines(
        &prepared,
        original.width,
        original.height,
        &ref_boxes,
        ordered,
        0.3,
    );
    report.insert(
        "rec".into(),
        Value::Array(reads.into_iter().map(|(_, e)| e).collect()),
    );
    report.insert(
        "output".into(),
        json!({"boxes": lines.iter().map(|l| l.quad).collect::<Vec<_>>(),
               "txts": lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>(),
               "scores": lines.iter().map(|l| l.score).collect::<Vec<_>>()}),
    );
    fs::write(
        out.join("rust.json"),
        serde_json::to_string(&Value::Object(report))?,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err(
            "usage: parity_dump REFERENCE_DIR IMAGE_DIR DICTIONARY_TXT OUT_DIR [PAGE_ID...]".into(),
        );
    }
    let (reference, images, out) = (
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
        PathBuf::from(&args[4]),
    );
    let dictionary = Dictionary::from_metadata(&fs::read_to_string(&args[3])?);
    let index: Value = serde_json::from_str(&fs::read_to_string(reference.join("index.json"))?)?;
    let pages: Vec<String> = if args.len() > 5 {
        args[5..].to_vec()
    } else {
        index["pages"]
            .as_array()
            .ok_or("pages")?
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect()
    };
    for id in pages {
        page(&reference.join(&id), &images, &dictionary, &out.join(&id))
            .map_err(|e| format!("{id}: {e}"))?;
    }
    Ok(())
}
