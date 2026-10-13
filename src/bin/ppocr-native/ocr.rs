//! Pages through both networks the way RapidOCR reads them, recognising a
//! page's lines on several threads, then reading weak vertical lines again.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::thread;

use ppocr_native::engine::{Kind, Model};
use ppocr_native::{
    boxes_from_prob, crop_text, ctc_decode, det_input, det_size, fit_page, is_blank, page_lines,
    prepare_page, rec_input, to_page, DbOptions, Dictionary, Image, Limit, Line, PageOptions,
    PreparedPage, Quad, REC_HEIGHT, VERTICAL_ASPECT_RATIO,
};

pub const DET_PLAN: &str = "PP-OCRv6_det_small.plan";
pub const REC_PLAN: &str = "PP-OCRv6_rec_small_argmax.plan";
pub const CHARACTERS: &str = "PP-OCRv6_rec_small.characters.txt";

/// The ImageNet normalization PaddleOCR feeds its detector, applied to B, G, R
/// in that order as RapidOCR applies it. RapidOCR's own 0.5 halves the
/// contrast the detector sees, and on low-contrast scans it then drops line
/// ends.
const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
const STD: [f64; 3] = [0.229, 0.224, 0.225];

/// Vertical lines, which `crop_text` turns a quarter: such a line scoring
/// under `CONFIDENT_SCORE` is read again turned half a turn, and the new
/// reading kept when it scores at least `MIN_SCORE_GAIN` more.
const CONFIDENT_SCORE: f64 = 0.90;
const MIN_SCORE_GAIN: f64 = 0.05;

pub struct Options {
    pub page: PageOptions,
    pub limit: Limit,
    pub db: DbOptions,
    pub text_score: f64,
}

/// The read of one page: its lines on the page as given, and how many of
/// them the vertical pass replaced.
pub struct Page {
    pub lines: Vec<Line>,
    pub reoriented: usize,
}

pub struct Reader {
    det: Model,
    /// One recogniser per thread, sharing their weights; none when the
    /// reader only detects.
    rec: Vec<Model>,
    dictionary: Dictionary,
    pub options: Options,
}

impl Reader {
    /// Load the plans in `dir`, checking that they fit together: a
    /// three-channel detector, and an argmax recogniser 48 pixels high whose
    /// classes are the dictionary's.
    pub fn load(dir: &Path, threads: usize, options: Options) -> Result<Self, String> {
        let load = |name: &str| {
            let path = dir.join(name);
            Model::load(&path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let det = load(DET_PLAN)?;
        if det.input_shape().2 != 3 || det.output_count() != 1 {
            return Err(format!("{DET_PLAN}: not a text detector"));
        }
        let mut rec = Vec::new();
        let mut dictionary = Dictionary::from_metadata("");
        if threads > 0 {
            let path = dir.join(CHARACTERS);
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            dictionary = Dictionary::from_metadata(&text);
            let first = load(REC_PLAN)?;
            let classes = first.ops().iter().find(|op| op.kind == Kind::Ctcargmax);
            if first.input_shape().0 != REC_HEIGHT
                || first.input_shape().2 != 3
                || classes.is_none_or(|op| op.n2 != dictionary.len())
            {
                return Err(format!(
                    "{REC_PLAN} is not an argmax recogniser for the {} classes of {CHARACTERS}",
                    dictionary.len()
                ));
            }
            for _ in 1..threads {
                rec.push(first.try_clone().map_err(|e| e.to_string())?);
            }
            rec.insert(0, first);
        }
        Ok(Self {
            det,
            rec,
            dictionary,
            options,
        })
    }

    /// RapidOCR's detection: the page fitted within `page`, the detector run
    /// at its input size, the boxes in reading order on the fitted page.
    fn detect(
        &mut self,
        image: &Image,
        page: &PageOptions,
    ) -> Result<Option<(PreparedPage, Vec<Quad>)>, String> {
        let Some(prepared) = prepare_page(image, page) else {
            return Ok(None);
        };
        let fitted = &prepared.image;
        // Scaling a thin page's short side up to 960 can stretch its long
        // side a hundredfold. Past the page limit squared it is detected at
        // the size `Limit::Max` gives instead.
        let size = |limit| det_size(fitted.width, fitted.height, limit);
        let max_side = self.options.page.max_side;
        let Some((w, h)) = size(self.options.limit)
            .filter(|&(w, h)| w.saturating_mul(h) <= max_side.saturating_mul(max_side))
            .or_else(|| size(Limit::Max))
        else {
            return Ok(None);
        };
        let chw = det_input(fitted, w, h, MEAN, STD);
        self.det.set_input_size(h, w).map_err(|e| e.to_string())?;
        let plane = w * h;
        for (i, v) in self.det.input_mut().iter_mut().enumerate() {
            *v = chw[(i % 3) * plane + i / 3];
        }
        self.det.run();
        let boxes = boxes_from_prob(
            self.det.output(0),
            w,
            h,
            fitted.width,
            fitted.height,
            &self.options.db,
        );
        Ok(Some((prepared, boxes)))
    }

    /// The text lines of a page, vertical ones read again where that reads better.
    pub fn read(&mut self, image: &Image) -> Result<Page, String> {
        if self.rec.is_empty() {
            return Err("this worker only detects".into());
        }
        let page = self.options.page;
        let Some((prepared, boxes)) = self.detect(image, &page)? else {
            return Ok(Page {
                lines: Vec::new(),
                reoriented: 0,
            });
        };
        let reads = recognise_all(&mut self.rec, &self.dictionary, &prepared.image, &boxes)?;
        let mut lines = page_lines(
            &prepared,
            image.width,
            image.height,
            &boxes,
            reads,
            self.options.text_score,
        );
        let reoriented = self.reorient(image, &mut lines)?;
        Ok(Page { lines, reoriented })
    }

    /// The vertical pass over the finished lines: each candidate is
    /// cropped again from the original page, turned half a turn and read
    /// alone, as RapidOCR reads an image without detection.
    fn reorient(&mut self, image: &Image, lines: &mut [Line]) -> Result<usize, String> {
        let mut replaced = 0;
        for line in lines.iter_mut() {
            if !rereads(line) {
                continue;
            }
            let crop = rotate_half(&crop_text(image, &line.quad));
            let page = &self.options.page;
            let Some((fitted, _, _)) = fit_page(&crop, page.min_side, page.max_side) else {
                continue;
            };
            let (text, score) = recognise(&mut self.rec[0], &self.dictionary, &fitted)?;
            if reads_better(line, &text, score) {
                line.text = text;
                line.score = score;
                replaced += 1;
            }
        }
        Ok(replaced)
    }

    /// The detector's boxes alone on the page as given, from a page fitted
    /// within `max_side`: what RapidOCR returns with recognition off.
    pub fn boxes(&mut self, image: &Image, max_side: usize) -> Result<Vec<Quad>, String> {
        let page = PageOptions {
            max_side,
            ..self.options.page
        };
        let Some((prepared, boxes)) = self.detect(image, &page)? else {
            return Ok(Vec::new());
        };
        Ok(boxes
            .iter()
            .map(|quad| to_page(&prepared, image.width, image.height, quad))
            .collect())
    }

    /// The probability map of an image already at the detector's input size,
    /// normalized as bigocrpdf's dewarp expects: in `f32`, the image's
    /// channels in R, G, B order.
    pub fn probability_map(&mut self, image: &Image) -> Result<&[f32], String> {
        self.det
            .set_input_size(image.height, image.width)
            .map_err(|e| e.to_string())?;
        let input = self.det.input_mut();
        for (px, out) in image
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .zip(input.as_chunks_mut::<3>().0)
        {
            for c in 0..3 {
                out[c] = (f32::from(px[2 - c]) / 255.0 - MEAN[c] as f32) / STD[c] as f32;
            }
        }
        self.det.run();
        Ok(self.det.output(0))
    }
}

/// Read the line in each of `boxes` on `image`, on as many threads as there
/// are models. The widest lines go first, so the last to finish is a short
/// one.
fn recognise_all(
    models: &mut [Model],
    dictionary: &Dictionary,
    image: &Image,
    boxes: &[Quad],
) -> Result<Vec<(String, f64)>, String> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| width(&boxes[b]).total_cmp(&width(&boxes[a])));
    let next = AtomicUsize::new(0);
    let reads = Mutex::new(vec![(String::new(), 0.0); boxes.len()]);
    let work = |model: &mut Model| -> Result<(), String> {
        loop {
            let Some(&i) = order.get(next.fetch_add(1, Ordering::Relaxed)) else {
                return Ok(());
            };
            let read = recognise(model, dictionary, &crop_text(image, &boxes[i]))?;
            reads.lock().unwrap_or_else(PoisonError::into_inner)[i] = read;
        }
    };
    let work = &work;
    let busy = models.len().min(boxes.len());
    match models.split_first_mut() {
        Some((first, rest)) if busy > 1 => thread::scope(|scope| {
            let helpers: Vec<_> = rest[..busy - 1]
                .iter_mut()
                .map(|model| scope.spawn(move || work(model)))
                .collect();
            let mut result = work(first);
            for helper in helpers {
                let done = helper
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                result = result.and(done);
            }
            result
        })?,
        Some((first, _)) => work(first)?,
        None => {}
    }
    Ok(reads.into_inner().unwrap_or_else(PoisonError::into_inner))
}

/// One line through the recogniser and CTC decoding.
fn recognise(
    model: &mut Model,
    dictionary: &Dictionary,
    crop: &Image,
) -> Result<(String, f64), String> {
    if crop.width == 0 || crop.height == 0 {
        return Ok((String::new(), 0.0));
    }
    let (chw, width) = rec_input(crop);
    model
        .set_input_size(REC_HEIGHT, width)
        .map_err(|e| e.to_string())?;
    let plane = REC_HEIGHT * width;
    for (i, v) in model.input_mut().iter_mut().enumerate() {
        *v = chw[(i % 3) * plane + i / 3];
    }
    model.run();
    let (best, prob): (Vec<u32>, Vec<f32>) = model
        .output(0)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[class, p]| (class as u32, p))
        .unzip();
    Ok(ctc_decode(&best, &prob, dictionary))
}

/// The longer horizontal side of a box, which its recogniser input grows with.
fn width(quad: &Quad) -> f32 {
    let side = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
    side(quad[0], quad[1]).max(side(quad[2], quad[3]))
}

/// A line the vertical pass reads again: its box's axis-aligned extent, in
/// `f64`, at least `VERTICAL_ASPECT_RATIO` times as tall as wide, and its
/// score under `CONFIDENT_SCORE`.
fn rereads(line: &Line) -> bool {
    let (mut min, mut max) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for point in &line.quad {
        for axis in 0..2 {
            min[axis] = min[axis].min(f64::from(point[axis]));
            max[axis] = max[axis].max(f64::from(point[axis]));
        }
    }
    let (width, height) = (max[0] - min[0], max[1] - min[1]);
    width > 0.0 && height >= width * VERTICAL_ASPECT_RATIO && line.score < CONFIDENT_SCORE
}

/// Whether the half-turn reading `text` scoring `score` replaces `line`'s.
fn reads_better(line: &Line, text: &str, score: f64) -> bool {
    !is_blank(text) && score >= line.score + MIN_SCORE_GAIN
}

/// `cv2.rotate(image, cv2.ROTATE_180)`.
fn rotate_half(image: &Image) -> Image {
    let mut data = image.data.clone();
    data.as_chunks_mut::<3>().0.reverse();
    Image::new(image.width, image.height, data)
}

#[cfg(test)]
mod tests {
    use super::{reads_better, rereads, rotate_half};
    use ppocr_native::{Image, Line};

    fn line(width: f32, height: f32, score: f64) -> Line {
        Line {
            quad: [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            text: "abc".into(),
            score,
        }
    }

    #[test]
    fn vertical_pass_choices() {
        for (width, height, score, expected) in [
            (40.0, 10.0, 0.5, false),
            (10.0, 40.0, 0.5, true),
            (10.0, 15.0, 0.5, true),
            (10.0, 14.9, 0.5, false),
            (0.0, 40.0, 0.5, false),
            (10.0, 40.0, 0.9, false),
        ] {
            let line = line(width, height, score);
            assert_eq!(rereads(&line), expected, "{width}x{height} at {score}");
        }
        let weak = line(10.0, 40.0, 0.5);
        for (text, score, expected) in [
            ("abd", 0.55, true),
            ("abd", 0.54, false),
            (" \x1c", 0.99, false),
        ] {
            assert_eq!(
                reads_better(&weak, text, score),
                expected,
                "{text:?} at {score}"
            );
        }
    }

    #[test]
    fn half_turn_reverses_pixels() {
        let image = Image::new(2, 2, (0..12).collect());
        let turned = rotate_half(&image);
        assert_eq!(turned.data, [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]);
    }
}
