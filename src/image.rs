//! The decoded page and the two pixel operations every stage shares: OpenCV's
//! bilinear resize on 8-bit pixels, and a quarter turn.

/// An 8-bit BGR image: rows top to bottom, the three channels of a pixel
/// next to each other, as `cv2.imread` returns it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Image {
    /// Wraps `data`, which must hold `width * height * 3` bytes.
    ///
    /// # Panics
    ///
    /// When the length does not match the size.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height * 3, "BGR buffer size");
        Self {
            width,
            height,
            data,
        }
    }

    /// A black image.
    pub fn zeroed(width: usize, height: usize) -> Self {
        Self::new(width, height, vec![0; width * height * 3])
    }
}

/// One output coordinate of the bilinear resize: two source indices and
/// their weights in 1/2048 units.
struct Tap {
    near: usize,
    far: usize,
    near_weight: i32,
    far_weight: i32,
}

/// OpenCV places output pixel `d` at source `(d + 0.5) * scale - 0.5`, rounds
/// that to `f32`, and rounds each of the two weights to 1/2048 on its own, so
/// they may not add up to 2048. Taps past either edge read the edge pixel.
fn taps(src: usize, dst: usize) -> Vec<Tap> {
    let scale = 1.0 / (dst as f64 / src as f64);
    let last = src as i64 - 1;
    (0..dst)
        .map(|d| {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let floor = f.floor();
            let a = f - floor;
            let s = floor as i64;
            Tap {
                near: s.clamp(0, last) as usize,
                far: (s + 1).clamp(0, last) as usize,
                near_weight: ((1.0 - a) * 2048.0).round_ties_even() as i32,
                far_weight: (a * 2048.0).round_ties_even() as i32,
            }
        })
        .collect()
}

/// `cv2.resize(image, (width, height))` with its default `INTER_LINEAR`,
/// bit for bit.
///
/// The horizontal pass keeps the full 1/2048 products; the vertical pass
/// drops four bits of each row before weighting and sixteen after, then
/// rounds the sum of the two. That staging is OpenCV's, and summing first
/// changes some pixels of a downscaled page.
/// OpenCV switches exact halvings to `INTER_AREA`, which produces the same
/// values as this formula.
///
/// # Panics
///
/// When either size of `image` or of the target is zero.
pub fn resize_linear(image: &Image, width: usize, height: usize) -> Image {
    assert!(
        image.width > 0 && image.height > 0 && width > 0 && height > 0,
        "resize to or from an empty image"
    );
    let xs = taps(image.width, width);
    let ys = taps(image.height, height);
    let stride = image.width * 3;
    let horizontal = |row: usize, out: &mut [i32]| {
        let src = &image.data[row * stride..(row + 1) * stride];
        for (tap, px) in xs.iter().zip(out.as_chunks_mut::<3>().0) {
            for c in 0..3 {
                px[c] = i32::from(src[tap.near * 3 + c]) * tap.near_weight
                    + i32::from(src[tap.far * 3 + c]) * tap.far_weight;
            }
        }
    };
    // The two most recent horizontal rows, by source row; output rows walk
    // down the source, so each source row is filtered once.
    let mut rows = [
        (usize::MAX, vec![0i32; width * 3]),
        (usize::MAX, vec![0i32; width * 3]),
    ];
    let mut out = Vec::with_capacity(width * height * 3);
    for tap in &ys {
        for want in [tap.near, tap.far] {
            if rows[0].0 != want && rows[1].0 != want {
                let slot = usize::from(rows[0].0 == tap.near || rows[0].0 == tap.far);
                horizontal(want, &mut rows[slot].1);
                rows[slot].0 = want;
            }
        }
        let pick = |want: usize| {
            if rows[0].0 == want {
                &rows[0].1
            } else {
                &rows[1].1
            }
        };
        let (top, bottom) = (pick(tap.near), pick(tap.far));
        for (&a, &b) in top.iter().zip(bottom) {
            let v = (((a >> 4) * tap.near_weight) >> 16) + (((b >> 4) * tap.far_weight) >> 16);
            out.push(((v + 2) >> 2).clamp(0, 255) as u8);
        }
    }
    Image::new(width, height, out)
}

/// `np.rot90`: a quarter turn counter-clockwise.
pub fn rotate_ccw(image: &Image) -> Image {
    let (w, h) = (image.width, image.height);
    let mut out = Vec::with_capacity(w * h * 3);
    for col in (0..w).rev() {
        for row in 0..h {
            let at = (row * w + col) * 3;
            out.extend_from_slice(&image.data[at..at + 3]);
        }
    }
    Image::new(h, w, out)
}
