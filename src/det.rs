//! Everything around the DB text detector: the page sizes it is fed, its input
//! tensor, and the boxes read back from its probability map.

use crate::contour::{contour_mean, find_contours};
use crate::geometry::{mini_box, unclip, Quad};
use crate::image::{resize_linear, Image};

/// Python's `int(round(x / 32) * 32)`: halves go to the even multiple, so 2000
/// becomes 1984.
fn snap32(x: usize) -> usize {
    (x as f64 / 32.0).round_ties_even() as usize * 32
}

/// RapidOCR's resize rule: scale both sides by `ratio`, truncate, snap to 32.
/// `None` when a side would vanish. An empty side is checked first: ratios
/// computed from it are infinite, where RapidOCR divides by zero.
fn scaled_size(width: usize, height: usize, ratio: f64) -> Option<(usize, usize)> {
    if width == 0 || height == 0 {
        return None;
    }
    let w = snap32((width as f64 * ratio) as usize);
    let h = snap32((height as f64 * ratio) as usize);
    (w > 0 && h > 0).then_some((w, h))
}

/// A page made ready for detection, and how to map its boxes back.
#[derive(Clone, Debug)]
pub struct PreparedPage {
    pub image: Image,
    /// Original height over prepared height, before padding.
    pub ratio_h: f64,
    /// Original width over prepared width.
    pub ratio_w: f64,
    /// Black rows added above (and as many below) the resized page.
    pub pad_top: usize,
}

/// RapidOCR's page preparation with `use_preprocess_img` on: a page whose
/// long side exceeds `max_side` is scaled down to it, then one whose short
/// side is under `min_side` is scaled up, both snapped to multiples of 32.
/// `None` when that would leave a side of zero; RapidOCR raises there.
pub fn fit_page(image: &Image, min_side: usize, max_side: usize) -> Option<(Image, f64, f64)> {
    let (w, h) = (image.width, image.height);
    let mut out = None;
    if w.max(h) > max_side {
        let ratio = max_side as f64 / if h > w { h } else { w } as f64;
        let (rw, rh) = scaled_size(w, h, ratio)?;
        out = Some(resize_linear(image, rw, rh));
    }
    let current = out.as_ref().unwrap_or(image);
    let (cw, ch) = (current.width, current.height);
    if cw.min(ch) < min_side {
        // RapidOCR scales from the already reduced page but still reports the
        // ratio against it, not against the original.
        let ratio = min_side as f64 / if ch < cw { ch } else { cw } as f64;
        let (rw, rh) = scaled_size(cw, ch, ratio)?;
        let resized = resize_linear(current, rw, rh);
        return Some((resized, ch as f64 / rh as f64, cw as f64 / rw as f64));
    }
    match out {
        Some(resized) => {
            let ratio_h = h as f64 / resized.height as f64;
            let ratio_w = w as f64 / resized.width as f64;
            Some((resized, ratio_h, ratio_w))
        }
        None => Some((image.clone(), 1.0, 1.0)),
    }
}

/// RapidOCR's vertical padding: a page at most `min_height` tall, or wider
/// than `width_height_ratio` times its height, gets black bands above and
/// below so the detector sees a usable aspect. A ratio of -1 turns the width
/// rule off, as in RapidOCR. Returns the padded page and the band height.
///
/// # Panics
///
/// When the bands would not fit in memory, as with a `width_height_ratio`
/// of 0 or close to it.
pub fn pad_vertically(image: &Image, width_height_ratio: f64, min_height: usize) -> (Image, usize) {
    let (w, h) = (image.width, image.height);
    let pad = band_height(w, h, width_height_ratio, min_height);
    if pad == 0 {
        return (image.clone(), 0);
    }
    let band = pad.checked_mul(w * 3).expect("padding band size");
    let mut data = vec![0; band];
    data.extend_from_slice(&image.data);
    data.resize(data.len() + band, 0);
    (Image::new(w, h + 2 * pad, data), pad)
}

/// The band [`pad_vertically`] adds above and below a `w` x `h` page.
fn band_height(w: usize, h: usize, width_height_ratio: f64, min_height: usize) -> usize {
    let too_wide = width_height_ratio != -1.0 && w as f64 / h as f64 > width_height_ratio;
    if h > min_height && !too_wide {
        return 0;
    }
    let target = ((w as f64 / width_height_ratio) as usize)
        .max(min_height)
        .saturating_mul(2);
    target.abs_diff(h) / 2
}

/// Both preparation steps of a page that goes through detection. `None` also
/// for a strip so thin that its padded page would exceed `max_side` squared:
/// a 4096x1 strip scales up to 122880x32 and pads to 30720 rows, gigabytes
/// for a few kilobytes of input.
pub fn prepare_page(image: &Image, options: &PageOptions) -> Option<PreparedPage> {
    let (fitted, ratio_h, ratio_w) = fit_page(image, options.min_side, options.max_side)?;
    let pad = band_height(
        fitted.width,
        fitted.height,
        options.width_height_ratio,
        options.min_height,
    );
    let padded_height = pad.saturating_mul(2).saturating_add(fitted.height);
    if fitted.width.saturating_mul(padded_height)
        > options.max_side.saturating_mul(options.max_side)
    {
        return None;
    }
    let (image, pad_top) = pad_vertically(&fitted, options.width_height_ratio, options.min_height);
    Some(PreparedPage {
        image,
        ratio_h,
        ratio_w,
        pad_top,
    })
}

/// RapidOCR's `Global` page limits.
#[derive(Clone, Copy, Debug)]
pub struct PageOptions {
    pub min_side: usize,
    pub max_side: usize,
    pub width_height_ratio: f64,
    pub min_height: usize,
}

impl Default for PageOptions {
    /// bigocrpdf's values: pages up to 4096 pixels, RapidOCR's other defaults.
    fn default() -> Self {
        Self {
            min_side: 30,
            max_side: 4096,
            width_height_ratio: 8.0,
            min_height: 30,
        }
    }
}

/// How the detector input size is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// The long side goes down to 960, 1500 or 2000, picked from the page's
    /// own long side; smaller pages keep their size.
    Max,
    /// The short side goes up to this length; larger pages keep their size.
    Min(usize),
}

/// The detector input size (width, height) for a page, both multiples of 32.
/// `None` when a side would vanish; RapidOCR then reports no text.
pub fn det_size(width: usize, height: usize, limit: Limit) -> Option<(usize, usize)> {
    let ratio = match limit {
        Limit::Max => {
            let long = width.max(height);
            let side = if long < 960 {
                960
            } else if long < 1500 {
                1500
            } else {
                2000
            };
            if long > side {
                side as f64 / long as f64
            } else {
                1.0
            }
        }
        Limit::Min(side) => {
            let short = width.min(height);
            if short < side {
                side as f64 / short as f64
            } else {
                1.0
            }
        }
    };
    scaled_size(width, height, ratio)
}

/// The detector input: `image` resized to `width` x `height`, scaled to
/// [0, 1] and normalized per channel in B, G, R order, as a 1x3xHxW tensor.
///
/// The arithmetic follows numpy's promotion exactly: the scaling is `f32`,
/// the normalization `f64`, the result rounded once to `f32`. Doing it all
/// in `f32` changes the last bit of many values.
pub fn det_input(
    image: &Image,
    width: usize,
    height: usize,
    mean: [f64; 3],
    std: [f64; 3],
) -> Vec<f32> {
    let resized = resize_linear(image, width, height);
    let plane = width * height;
    let mut out = vec![0f32; 3 * plane];
    let scale = (1.0f64 / 255.0) as f32;
    for (i, px) in resized.data.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            let x = f32::from(px[c]) * scale;
            out[c * plane + i] = ((f64::from(x) - mean[c]) / std[c]) as f32;
        }
    }
    out
}

/// The DB postprocess settings.
#[derive(Clone, Copy, Debug)]
pub struct DbOptions {
    /// Probability above which a pixel is text, compared in `f32` as numpy
    /// compares an `f32` map with a Python float.
    pub thresh: f32,
    /// Mean probability a region needs to be kept.
    pub box_thresh: f64,
    /// How far a box grows: area times this over perimeter.
    pub unclip_ratio: f64,
    /// Regions considered, in OpenCV's contour order.
    pub max_candidates: usize,
}

impl Default for DbOptions {
    /// bigocrpdf's values.
    fn default() -> Self {
        Self {
            thresh: 0.3,
            box_thresh: 0.5,
            unclip_ratio: 1.5,
            max_candidates: 1000,
        }
    }
}

/// Shorter side, in map pixels, under which a region is noise.
const MIN_SIZE: f32 = 3.0;

/// What became of one contour of the probability map.
#[derive(Clone, Debug, PartialEq)]
pub enum Candidate {
    /// Its minimum-area rectangle is thinner than 3 pixels.
    Small,
    /// Mean probability inside the contour under `box_thresh`.
    LowScore(f64),
    /// Thinner than 5 pixels once grown.
    SmallAfterUnclip(f64),
    /// Kept, with its score and its grown box in page pixels, corners in
    /// [`mini_box`] order.
    Kept(f64, [[i32; 2]; 4]),
}

/// One contour through RapidOCR's slow-score DB postprocess: minimum-area
/// rectangle, mean probability inside the contour, growth by the unclip
/// distance, and scaling from the `map_w` x `map_h` map to a
/// `dest_w` x `dest_h` page.
///
/// # Panics
///
/// When `prob` is empty or not `map_w` x `map_h`.
pub fn contour_candidate(
    prob: &[f32],
    map_w: usize,
    map_h: usize,
    contour: &[[i32; 2]],
    dest_w: usize,
    dest_h: usize,
    options: &DbOptions,
) -> Candidate {
    let (quad, side) = mini_box(contour);
    if side < MIN_SIZE {
        return Candidate::Small;
    }
    let score = contour_mean(prob, map_w, map_h, contour);
    if options.box_thresh > score {
        return Candidate::LowScore(score);
    }
    let (grown, side) = mini_box(&unclip(&quad, options.unclip_ratio));
    if side < MIN_SIZE + 2.0 {
        return Candidate::SmallAfterUnclip(score);
    }
    // numpy divides and multiplies the f32 box by the sizes in f32.
    let scale = |v: f32, map: usize, dest: usize| -> i32 {
        let dest = dest as f32;
        (v / map as f32 * dest).round_ties_even().clamp(0.0, dest) as i32
    };
    Candidate::Kept(
        score,
        grown.map(|[x, y]| [scale(x, map_w, dest_w), scale(y, map_h, dest_h)]),
    )
}

/// RapidOCR's `filter_det_res` for one box: corners reordered as top-left,
/// top-right, bottom-right, bottom-left, clipped to the page, and dropped
/// when either side is 3 pixels or shorter.
pub fn order_and_clip(corners: &[[i32; 2]; 4], width: usize, height: usize) -> Option<Quad> {
    let mut by_x = *corners;
    by_x.sort_by_key(|p| p[0]);
    let [mut a, mut b, mut c, mut d] = by_x;
    if b[1] < a[1] {
        (a, b) = (b, a);
    }
    if d[1] < c[1] {
        (c, d) = (d, c);
    }
    let clip = |p: [i32; 2]| -> [f32; 2] {
        [
            p[0].max(0).min(width as i32 - 1) as f32,
            p[1].max(0).min(height as i32 - 1) as f32,
        ]
    };
    let quad = [clip(a), clip(c), clip(d), clip(b)];
    let side = |p: [f32; 2], q: [f32; 2]| distance(p, q) as i32;
    (side(quad[0], quad[1]) > 3 && side(quad[0], quad[3]) > 3).then_some(quad)
}

/// `np.linalg.norm` of the difference of two `f32` points, in `f32`.
pub(crate) fn distance(p: [f32; 2], q: [f32; 2]) -> f32 {
    let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
    (dx * dx + dy * dy).sqrt()
}

/// RapidOCR's reading order: top-left corners sorted by y, a new line
/// wherever y jumps by 10 or more, left to right within a line. Both sorts
/// are stable.
pub fn sort_boxes(boxes: &mut [Quad]) {
    boxes.sort_by(|a, b| a[0][1].total_cmp(&b[0][1]));
    let mut line = 0u32;
    let mut keyed: Vec<(u32, f32, Quad)> = Vec::with_capacity(boxes.len());
    for (i, quad) in boxes.iter().enumerate() {
        if i > 0 && quad[0][1] - boxes[i - 1][0][1] >= 10.0 {
            line += 1;
        }
        keyed.push((line, quad[0][0], *quad));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    for (slot, (_, _, quad)) in boxes.iter_mut().zip(keyed) {
        *slot = quad;
    }
}

/// The text boxes of a detector probability map (`map_w` x `map_h`, values
/// in row order), in page pixels of a `dest_w` x `dest_h` page and in reading
/// order.
///
/// # Panics
///
/// When `prob` is not `map_w` x `map_h`.
pub fn boxes_from_prob(
    prob: &[f32],
    map_w: usize,
    map_h: usize,
    dest_w: usize,
    dest_h: usize,
    options: &DbOptions,
) -> Vec<Quad> {
    let mask: Vec<u8> = prob.iter().map(|&p| u8::from(p > options.thresh)).collect();
    let contours = find_contours(&mask, map_w, map_h);
    let mut boxes: Vec<Quad> = contours
        .iter()
        .take(options.max_candidates)
        .filter_map(|contour| {
            match contour_candidate(prob, map_w, map_h, contour, dest_w, dest_h, options) {
                Candidate::Kept(_, corners) => order_and_clip(&corners, dest_w, dest_h),
                Candidate::Small | Candidate::LowScore(_) | Candidate::SmallAfterUnclip(_) => None,
            }
        })
        .collect();
    sort_boxes(&mut boxes);
    boxes
}
