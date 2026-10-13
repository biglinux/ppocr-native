//! The recogniser's input tensor, and reading text back from its output.

use crate::image::{resize_linear, Image};

/// Input height of the PP-OCR recogniser.
pub const REC_HEIGHT: usize = 48;
/// Narrowest input; shorter lines are padded to it.
pub const REC_MIN_WIDTH: usize = 320;

/// The recogniser input for one crop, with a batch of one: a 1x3x48xW
/// tensor and its width W.
///
/// The crop is resized to height 48 at its own aspect ratio (rounded up),
/// scaled to [-1, 1], and padded with zeros on the right up to 320.
///
/// # Panics
///
/// When the crop is empty.
pub fn rec_input(crop: &Image) -> (Vec<f32>, usize) {
    let (h, w) = (crop.height as f64, crop.width as f64);
    let height = REC_HEIGHT as f64;
    let max_ratio = (REC_MIN_WIDTH as f64 / height).max(w / h);
    let width = (height * max_ratio) as usize;
    // The tensor width truncates and the resized width rounds up, so a wide
    // crop can come out one pixel over; it is cut back to the tensor.
    let fitted = (height * (w / h)).ceil();
    let resized_w = if fitted > width as f64 {
        width
    } else {
        fitted as usize
    };
    let resized = resize_linear(crop, resized_w, REC_HEIGHT);
    let plane = REC_HEIGHT * width;
    let mut out = vec![0f32; 3 * plane];
    for (row, pixels) in resized.data.chunks_exact(resized_w * 3).enumerate() {
        for (x, px) in pixels.as_chunks::<3>().0.iter().enumerate() {
            for (c, &v) in px.iter().enumerate() {
                out[c * plane + row * width + x] = (f32::from(v) / 255.0 - 0.5) / 0.5;
            }
        }
    }
    (out, width)
}

/// The recogniser's classes: CTC blank, the model's characters, and a space.
#[derive(Clone, Debug)]
pub struct Dictionary {
    classes: Vec<String>,
}

impl Dictionary {
    /// Builds the classes from the model's `character` metadata, split into
    /// lines as Python's `str.splitlines` does.
    pub fn from_metadata(text: &str) -> Self {
        let mut classes = vec!["blank".to_owned()];
        let mut rest = text;
        while !rest.is_empty() {
            match rest.char_indices().find(|&(_, c)| is_line_break(c)) {
                Some((at, c)) => {
                    classes.push(rest[..at].to_owned());
                    let skip = if rest[at..].starts_with("\r\n") {
                        2
                    } else {
                        c.len_utf8()
                    };
                    rest = &rest[at + skip..];
                }
                None => {
                    classes.push(rest.to_owned());
                    rest = "";
                }
            }
        }
        classes.push(" ".to_owned());
        Self { classes }
    }

    /// Number of classes, blank and space included.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Whether there are no classes; never, since blank and space are added.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// The text of class `index`.
    pub fn class(&self, index: usize) -> &str {
        &self.classes[index]
    }
}

fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// numpy's float64 `add.reduce`: a plain loop below 8 values, eight running
/// sums up to 128, halves split at a multiple of 8 above. Recognition scores
/// are means over these sums, so the grouping decides the last bit.
fn pairwise_sum(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 8 {
        return v.iter().fold(0.0, |s, &x| s + x);
    }
    if n <= 128 {
        let mut r: [f64; 8] = v[..8].try_into().expect("eight values");
        let full = n - n % 8;
        for chunk in v[8..full].as_chunks::<8>().0 {
            for (acc, &x) in r.iter_mut().zip(chunk) {
                *acc += x;
            }
        }
        let mut s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &v[full..] {
            s += x;
        }
        return s;
    }
    let half = n / 2 - (n / 2) % 8;
    pairwise_sum(&v[..half]) + pairwise_sum(&v[half..])
}

/// RapidOCR's CTC greedy decode from the best class and its probability at
/// each timestep: repeats collapse, blanks drop, and the score is the mean
/// of the kept probabilities, each rounded to five decimals first and the
/// mean rounded again. A line with nothing kept scores 0.
///
/// # Panics
///
/// When `best` and `prob` differ in length, or a class is outside
/// `dictionary`, as when the model and the dictionary do not belong together.
pub fn ctc_decode(best: &[u32], prob: &[f32], dictionary: &Dictionary) -> (String, f64) {
    assert_eq!(best.len(), prob.len(), "one probability per timestep");
    let mut text = String::new();
    let mut kept = Vec::new();
    for (t, (&class, &p)) in best.iter().zip(prob).enumerate() {
        if class == 0 || (t > 0 && class == best[t - 1]) {
            continue;
        }
        text.push_str(dictionary.class(class as usize));
        // An f32 times 1e5 is exact in f64, so this is Python's round(p, 5).
        kept.push((f64::from(p) * 1e5).round_ties_even() / 1e5);
    }
    if kept.is_empty() {
        return (text, 0.0);
    }
    let mean = pairwise_sum(&kept) / kept.len() as f64;
    (text, (mean * 1e5).round_ties_even() / 1e5)
}

/// [`ctc_decode`] from the full softmax output, `classes` values per
/// timestep; ties go to the lowest class and a NaN beats every number, as
/// numpy's `argmax` and `max` do.
///
/// # Panics
///
/// When `classes` is 0, or a timestep's best class is outside `dictionary`.
pub fn ctc_decode_probs(probs: &[f32], classes: usize, dictionary: &Dictionary) -> (String, f64) {
    let (best, prob): (Vec<u32>, Vec<f32>) = probs
        .chunks_exact(classes)
        .map(|row| {
            let mut top = 0;
            for (i, &p) in row.iter().enumerate() {
                if p > row[top] || (p.is_nan() && !row[top].is_nan()) {
                    top = i;
                }
            }
            (top as u32, row[top])
        })
        .unzip();
    ctc_decode(&best, &prob, dictionary)
}

/// Python's `str.strip()` would leave nothing: RapidOCR drops such lines.
pub fn is_blank(text: &str) -> bool {
    text.chars()
        .all(|c| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}
