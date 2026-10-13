//! PP-OCRv6 text detection and recognition in plain Rust, reproducing
//! RapidOCR.
//!
//! PP-OCR reads a page in two neural steps: a DB detector marks where text
//! is, and a CTC recogniser reads each line. [`engine`] runs both networks.
//! Everything between and around them is plain image and geometry work, and
//! it decides as much of the result as the networks do: how the page is
//! resized, which pixels become a box, how a box is straightened. The rest
//! of the crate does that work, so pages come out as RapidOCR reads them.
//!
//! Each step matches what RapidOCR 3.9 computes with OpenCV 5, numpy 2,
//! pyclipper and shapely. Where the arithmetic can be matched it is matched
//! bit for bit; [`min_area_rect`] and [`crop_text`] say where it is not.
//!
//! # Using it
//!
//! ```no_run
//! use ppocr_native::{
//!     boxes_from_prob, crop_text, ctc_decode, det_input, det_size, page_lines, prepare_page,
//!     rec_input, DbOptions, Dictionary, Image, Limit, Line, PageOptions,
//! };
//!
//! /// `detect` runs the detector on a 1x3xHxW tensor and returns the HxW map;
//! /// `recognise` runs the recogniser on a 1x3x48xW tensor and returns the best
//! /// class and its probability at each step.
//! fn read_page(
//!     page: &Image,
//!     dictionary: &Dictionary,
//!     detect: impl Fn(&[f32], usize, usize) -> Vec<f32>,
//!     recognise: impl Fn(&[f32], usize) -> (Vec<u32>, Vec<f32>),
//! ) -> Option<Vec<Line>> {
//!     let prepared = prepare_page(page, &PageOptions::default())?;
//!     let image = &prepared.image;
//!     let (w, h) = det_size(image.width, image.height, Limit::Max)?;
//!     let tensor = det_input(image, w, h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
//!     let prob = detect(&tensor, w, h);
//!     let boxes = boxes_from_prob(&prob, w, h, image.width, image.height, &DbOptions::default());
//!     let reads = boxes
//!         .iter()
//!         .map(|quad| {
//!             let (input, width) = rec_input(&crop_text(image, quad));
//!             let (best, prob) = recognise(&input, width);
//!             ctc_decode(&best, &prob, dictionary)
//!         })
//!         .collect();
//!     Some(page_lines(&prepared, page.width, page.height, &boxes, reads, 0.3))
//! }
//! ```

use std::cmp::Ordering;

mod contour;
mod crop;
mod det;
#[allow(unsafe_code)] // SIMD kernels and scratch views; see each SAFETY comment
pub mod engine;
mod geometry;
mod image;
mod rec;

pub use contour::{contour_mean, find_contours};
pub use crop::{crop_text, VERTICAL_ASPECT_RATIO};
pub use det::{
    boxes_from_prob, contour_candidate, det_input, det_size, fit_page, order_and_clip,
    pad_vertically, prepare_page, sort_boxes, Candidate, DbOptions, Limit, PageOptions,
    PreparedPage,
};
pub use geometry::{min_area_rect, mini_box, unclip, Quad, RotatedRect};
pub use image::{resize_linear, rotate_ccw, Image};
pub use rec::{
    ctc_decode, ctc_decode_probs, is_blank, rec_input, Dictionary, REC_HEIGHT, REC_MIN_WIDTH,
};

/// A line of text on the original page.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// Corners top-left, top-right, bottom-right, bottom-left, in pixels of
    /// the page as given.
    pub quad: Quad,
    pub text: String,
    pub score: f64,
}

/// RapidOCR's final assembly: boxes found on `page` are mapped back to the
/// `width` x `height` original, and lines whose text is blank or scores
/// under `text_score` are dropped. A NaN never counts as under, so such a
/// line stays, as in RapidOCR.
///
/// `boxes` and `reads` are parallel, in reading order. The mapping is done
/// in `f32`, as numpy does it, and clamps to the page size itself, not to
/// its last pixel.
///
/// # Panics
///
/// When `boxes` and `reads` differ in length.
pub fn page_lines(
    page: &PreparedPage,
    width: usize,
    height: usize,
    boxes: &[Quad],
    reads: Vec<(String, f64)>,
    text_score: f64,
) -> Vec<Line> {
    assert_eq!(boxes.len(), reads.len(), "one reading per box");
    boxes
        .iter()
        .zip(reads)
        .filter(|(_, (text, score))| {
            !is_blank(text) && score.partial_cmp(&text_score) != Some(Ordering::Less)
        })
        .map(|(quad, (text, score))| Line {
            quad: to_page(page, width, height, quad),
            text,
            score,
        })
        .collect()
}

/// A box found on `page` on the `width` x `height` original, as
/// [`page_lines`] maps it: in `f32`, clamped to the page size itself.
pub fn to_page(page: &PreparedPage, width: usize, height: usize, quad: &Quad) -> Quad {
    let (ratio_w, ratio_h) = (page.ratio_w as f32, page.ratio_h as f32);
    let (max_x, max_y) = (width as f32, height as f32);
    quad.map(|[x, y]| {
        let x = (x * ratio_w).max(0.0).min(max_x);
        let y = ((y - page.pad_top as f32) * ratio_h).max(0.0).min(max_y);
        [x, y]
    })
}
