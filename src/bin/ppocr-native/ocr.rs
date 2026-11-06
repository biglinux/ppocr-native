//! Pages through both networks the way RapidOCR reads them, recognising a
//! page's lines on several threads, then reading weak vertical lines again.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex, PoisonError};
use std::thread;

use ppocr_native::engine::{GpuModel, Kind, Model};
use ppocr_native::{
    boxes_from_prob, crop_text, ctc_decode, det_input_hwc, det_size, fit_page, is_blank,
    page_lines, prepare_page, rec_input, to_page, DbOptions, Dictionary, Image, Limit, Line,
    PageOptions, PreparedPage, Quad, REC_HEIGHT, VERTICAL_ASPECT_RATIO,
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
/// under `CONFIDENT_SCORE` is read again turned half a turn, and the second
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

/// The detector on the CPU and, with `--gpu`, on a GPU, which runs every
/// page it accepts; a page it refuses (too large for its buffers, say) runs
/// on the CPU.
struct Detector {
    cpu: Model,
    gpu: Option<GpuModel>,
    /// Whether the GPU has refused a page yet, which is said once.
    refused: bool,
}

impl Detector {
    /// The probability map of a `height` x `width` input that `fill` writes
    /// into the detector's input.
    fn run(
        &mut self,
        height: usize,
        width: usize,
        fill: impl Fn(&mut [f32]),
    ) -> Result<&[f32], String> {
        let on_gpu = match &mut self.gpu {
            Some(gpu) => match gpu.set_input_size(height, width).and_then(|()| {
                fill(gpu.input_mut());
                gpu.run()
            }) {
                Ok(()) => true,
                Err(e) => {
                    if !self.refused {
                        eprintln!("ppocr-native: the GPU refused a page ({e}); detecting such pages on the CPU");
                        self.refused = true;
                    }
                    false
                }
            },
            None => false,
        };
        if let (true, Some(gpu)) = (on_gpu, &self.gpu) {
            return Ok(gpu.output(0));
        }
        self.cpu
            .set_input_size(height, width)
            .map_err(|e| e.to_string())?;
        fill(self.cpu.input_mut());
        self.cpu.run();
        Ok(self.cpu.output(0))
    }
}

pub struct Reader {
    det: Detector,
    det_threads: usize,
    /// One recogniser per thread, sharing their weights; none when the
    /// reader only detects or the GPU reads every line.
    rec: Vec<Model>,
    /// The recogniser on the GPU, and the share of a page's line width it
    /// reads.
    gpu_rec: Option<(GpuModel, f64)>,
    /// Threads that crop the lines the GPU recogniser reads.
    crop_threads: usize,
    dictionary: Dictionary,
    pub options: Options,
}

impl Reader {
    /// Load the plans in `dir`, checking that they fit together: a
    /// three-channel detector, run on `det_threads` threads, and
    /// `recognisers` argmax recognisers 48 pixels high whose classes are the
    /// dictionary's.
    pub fn load(
        dir: &Path,
        recognisers: usize,
        det_threads: usize,
        gpu: bool,
        options: Options,
    ) -> Result<Self, String> {
        let load = |name: &str| {
            let path = dir.join(name);
            Model::load(&path).map_err(|e| format!("{}: {e}", path.display()))
        };
        let mut det = load(DET_PLAN)?;
        if det.input_shape().2 != 3 || det.output_count() != 1 {
            return Err(format!("{DET_PLAN}: not a text detector"));
        }
        det.set_threads(det_threads)
            .map_err(|e| format!("{}: {e}", dir.join(DET_PLAN).display()))?;
        // With `gpu`, both networks run on the GPU when it takes both, else
        // both on the CPU. An integrated GPU reads `GPU_SHARE_INTEGRATED` of
        // each page's line width, the cores the rest; a discrete one reads
        // every line.
        let on_gpu = |name: &str| GpuModel::load(&dir.join(name));
        let gpu = if !gpu {
            None
        } else {
            let both = on_gpu(DET_PLAN).and_then(|det| {
                let rec = (recognisers > 0).then(|| on_gpu(REC_PLAN)).transpose()?;
                Ok((det, rec))
            });
            match both {
                Ok(both) => Some(both),
                Err(e) => {
                    eprintln!("ppocr-native: no GPU ({e}); running on the CPU");
                    None
                }
            }
        };
        let gpu_share = match &gpu {
            Some((det, _)) if det.integrated() => GPU_SHARE_INTEGRATED,
            Some(_) => 1.0,
            None => 0.0,
        };
        let mut rec = Vec::new();
        let mut dictionary = Dictionary::from_metadata("");
        if recognisers > 0 {
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
            if gpu_share < 1.0 {
                for _ in 1..recognisers {
                    rec.push(first.try_clone().map_err(|e| e.to_string())?);
                }
                rec.insert(0, first);
            }
        }
        let (gpu_det, gpu_rec) = match gpu {
            Some((gpu_det, gpu_rec)) => {
                eprintln!("ppocr-native: running on {}", gpu_det.device());
                (Some(gpu_det), gpu_rec.map(|m| (m, gpu_share)))
            }
            None => (None, None),
        };
        Ok(Self {
            det: Detector {
                cpu: det,
                gpu: gpu_det,
                refused: false,
            },
            det_threads: det_threads.max(1),
            rec,
            gpu_rec,
            crop_threads: recognisers.max(1),
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
        let threads = self.det_threads;
        let prob = self.det.run(h, w, |input| {
            det_input_hwc(fitted, w, h, MEAN, STD, threads, input);
        })?;
        let boxes = boxes_from_prob(prob, w, h, fitted.width, fitted.height, &self.options.db);
        Ok(Some((prepared, boxes)))
    }

    /// The text lines of a page, vertical ones read again where that reads better.
    pub fn read(&mut self, image: &Image) -> Result<Page, String> {
        if self.rec.is_empty() && self.gpu_rec.is_none() {
            return Err("this worker only detects".into());
        }
        let page = self.options.page;
        let Some((prepared, boxes)) = self.detect(image, &page)? else {
            return Ok(Page {
                lines: Vec::new(),
                reoriented: 0,
            });
        };
        let reads = recognise_all(
            &mut self.rec,
            self.gpu_rec.as_mut(),
            self.crop_threads,
            &self.dictionary,
            &prepared.image,
            &boxes,
        )?;
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
            let (text, score) = match (self.rec.first_mut(), self.gpu_rec.as_mut()) {
                (Some(model), _) => recognise(model, &self.dictionary, &fitted)?,
                (None, Some((model, _))) => match rec_hwc(&fitted) {
                    Some(input) => read_on_gpu(model, &self.dictionary, &[&input])?.swap_remove(0),
                    None => (String::new(), 0.0),
                },
                (None, None) => unreachable!("`read` refuses a worker that only detects"),
            };
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
        self.det.run(image.height, image.width, |input| {
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
        })
    }
}

/// Read the line in each of `boxes` on `image`, on as many threads as there
/// are CPU models and, when there is one, on the GPU, which takes its share
/// of the line width: the widest line, then each next while its lines are
/// under that share of those so far. The widest lines go first, so the last
/// to finish is a short one, and which device reads a line depends on the
/// boxes alone.
fn recognise_all(
    models: &mut [Model],
    gpu: Option<&mut (GpuModel, f64)>,
    threads: usize,
    dictionary: &Dictionary,
    image: &Image,
    boxes: &[Quad],
) -> Result<Vec<(String, f64)>, String> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| width(&boxes[b]).total_cmp(&width(&boxes[a])));
    let Some((gpu, share)) = gpu else {
        return recognise_on_cpu(models, dictionary, image, boxes, &order);
    };
    if models.is_empty() {
        return recognise_on_gpu(gpu, threads, dictionary, image, boxes, &order);
    }
    let (mut on_gpu, mut on_cpu) = (Vec::new(), Vec::new());
    let (mut total, mut taken) = (0.0, 0.0);
    for &i in &order {
        let w = f64::from(width(&boxes[i]));
        total += w;
        if taken < *share * total {
            taken += w;
            on_gpu.push(i);
        } else {
            on_cpu.push(i);
        }
    }
    let (gpu_reads, mut reads) = thread::scope(|scope| {
        let gpu = scope.spawn(|| recognise_on_gpu(gpu, 1, dictionary, image, boxes, &on_gpu));
        let cpu = recognise_on_cpu(models, dictionary, image, boxes, &on_cpu);
        let gpu = gpu
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        Ok::<_, String>((gpu?, cpu?))
    })?;
    for i in on_gpu {
        reads[i] = gpu_reads[i].clone();
    }
    Ok(reads)
}

/// Read the lines of `boxes` in `order` on the CPU models, one thread each.
fn recognise_on_cpu(
    models: &mut [Model],
    dictionary: &Dictionary,
    image: &Image,
    boxes: &[Quad],
    order: &[usize],
) -> Result<Vec<(String, f64)>, String> {
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
    let busy = models.len().min(order.len());
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

/// Lines the GPU recogniser reads in one run, at most: enough to keep it
/// busy between the steps of narrow lines.
const GPU_LINES: usize = 16;

/// The share of a page's line width an integrated GPU reads, the cores
/// beside it the rest, since the two share the memory and the power budget
/// (docs/development.md has the measurements).
const GPU_SHARE_INTEGRATED: f64 = 0.4;

/// Read the lines of `boxes` in `order` on the GPU, up to `GPU_LINES` at a
/// time, each a lane of its own: the same reads as one line at a time.
/// `threads` threads crop the lines in order while the GPU reads the runs
/// already cropped.
fn recognise_on_gpu(
    model: &mut GpuModel,
    threads: usize,
    dictionary: &Dictionary,
    image: &Image,
    boxes: &[Quad],
    order: &[usize],
) -> Result<Vec<(String, f64)>, String> {
    let next = AtomicUsize::new(0);
    let (sent, received) = mpsc::channel();
    thread::scope(|scope| {
        // Dropped on return, which stops the croppers early after an error.
        let received = received;
        for _ in 0..threads.min(order.len()) {
            let (sent, next) = (sent.clone(), &next);
            scope.spawn(move || loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(&i) = order.get(k) else {
                    return;
                };
                let input = rec_hwc(&crop_text(image, &boxes[i]));
                if sent.send((k, i, input)).is_err() {
                    return;
                }
            });
        }
        drop(sent);
        let mut cropped = vec![None; order.len()];
        let mut reads = vec![(String::new(), 0.0); boxes.len()];
        let batch = GPU_LINES.min(model.max_lanes());
        for start in (0..order.len()).step_by(batch) {
            let run = start..(start + batch).min(order.len());
            while cropped[run.clone()].iter().any(Option::is_none) {
                // Every cropper gone with lines left: one panicked, and the
                // scope raises it.
                let Ok((k, i, input)) = received.recv() else {
                    return Ok(reads);
                };
                cropped[k] = Some((i, input));
            }
            let lines: Vec<(usize, (Vec<f32>, usize))> = cropped[run]
                .iter_mut()
                .filter_map(|line| {
                    let (i, input) = line.take()?;
                    Some((i, input?))
                })
                .collect();
            let tensors: Vec<_> = lines.iter().map(|(_, input)| input).collect();
            for ((i, _), read) in lines.iter().zip(read_on_gpu(model, dictionary, &tensors)?) {
                reads[*i] = read;
            }
        }
        Ok(reads)
    })
}

/// Read the recogniser inputs `lines` (as [`rec_hwc`] gives them) on the GPU
/// in one run, each a lane of its own.
fn read_on_gpu(
    model: &mut GpuModel,
    dictionary: &Dictionary,
    lines: &[&(Vec<f32>, usize)],
) -> Result<Vec<(String, f64)>, String> {
    if lines.is_empty() {
        return Ok(Vec::new());
    }
    let sizes: Vec<_> = lines.iter().map(|&&(_, w)| (REC_HEIGHT, w)).collect();
    model.set_sizes(&sizes)?;
    for (lane, (hwc, _)) in lines.iter().enumerate() {
        model.lane_input_mut(lane).copy_from_slice(hwc);
    }
    model.run()?;
    Ok((0..lines.len())
        .map(|lane| decode(model.lane_output(lane, 0), dictionary))
        .collect())
}

/// `rec_input`'s tensor for `crop`, HWC, and its width; none for an empty
/// crop, which reads as no text.
fn rec_hwc(crop: &Image) -> Option<(Vec<f32>, usize)> {
    if crop.width == 0 || crop.height == 0 {
        return None;
    }
    let (chw, width) = rec_input(crop);
    let mut hwc = vec![0.0; chw.len()];
    to_hwc(&mut hwc, &chw, REC_HEIGHT * width);
    Some((hwc, width))
}

/// A CHW tensor of `plane` pixels into `dst`, HWC.
fn to_hwc(dst: &mut [f32], chw: &[f32], plane: usize) {
    for (i, v) in dst.iter_mut().enumerate() {
        *v = chw[(i % 3) * plane + i / 3];
    }
}

/// The text of the recogniser's output: each step's best class and its
/// probability, through CTC decoding.
fn decode(output: &[f32], dictionary: &Dictionary) -> (String, f64) {
    let (best, prob): (Vec<u32>, Vec<f32>) = output
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[class, p]| (class as u32, p))
        .unzip();
    ctc_decode(&best, &prob, dictionary)
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
    to_hwc(model.input_mut(), &chw, REC_HEIGHT * width);
    model.run();
    Ok(decode(model.output(0), dictionary))
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
