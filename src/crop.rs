//! Straightening a text box out of the page for the recogniser.

use crate::det::distance;
use crate::geometry::Quad;
use crate::image::{rotate_ccw, Image};

/// The projective map sending the corners of the `width` x `height`
/// rectangle (top-left, top-right, bottom-right, bottom-left) onto `quad`,
/// as a row-major 3x3 matrix with the last entry 1.
fn rectangle_to_quad(width: f64, height: f64, quad: &Quad) -> [f64; 9] {
    let from = [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]];
    // Eight equations u = (a x + b y + c) / (g x + h y + 1), v likewise, in
    // the unknowns a..h; solved by Gaussian elimination with partial pivoting.
    let mut m = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let [x, y] = from[i];
        let (u, v) = (f64::from(quad[i][0]), f64::from(quad[i][1]));
        m[i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u];
        m[i + 4] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v];
    }
    for col in 0..8 {
        let pivot = (col..8)
            .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
            .unwrap_or(col);
        m.swap(col, pivot);
        let p = m[col][col];
        if p == 0.0 {
            continue;
        }
        let pivot_row = m[col];
        for (r, row) in m.iter_mut().enumerate() {
            let f = row[col] / p;
            if r != col && f != 0.0 {
                for (v, &q) in row.iter_mut().zip(&pivot_row).skip(col) {
                    *v -= f * q;
                }
            }
        }
    }
    let mut out = [0.0; 9];
    for i in 0..8 {
        out[i] = if m[i][i] == 0.0 {
            0.0
        } else {
            m[i][8] / m[i][i]
        };
    }
    out[8] = 1.0;
    out
}

/// Keys' cubic convolution weights with a = -0.75, OpenCV's `INTER_CUBIC`,
/// for a sample `t` in [0, 1) past the second of four taps. The factoring is
/// OpenCV's `interpolateCubic` (imgwarp.cpp), on which the `f32` rounding
/// depends; its licence is in NOTICE.
fn cubic_weights(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let w0 = ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
    let w1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    let w2 = ((A + 2.0) * (1.0 - t) - (A + 3.0)) * (1.0 - t) * (1.0 - t) + 1.0;
    [w0, w1, w2, 1.0 - w0 - w1 - w2]
}

/// How many times taller than wide a crop is when `crop_text` turns it a
/// quarter counter-clockwise, as RapidOCR does; text set bottom to top then
/// reads upside down.
pub const VERTICAL_ASPECT_RATIO: f64 = 1.5;

/// RapidOCR's `get_rotate_crop_image`: the quad (top-left, top-right,
/// bottom-right, bottom-left) mapped onto an upright rectangle as wide as its
/// longer horizontal side and as tall as its longer vertical side, sampled
/// bicubically with the edge pixels repeated outside `image`, then turned a
/// quarter counter-clockwise when it is at least `VERTICAL_ASPECT_RATIO` times
/// taller than wide.
///
/// OpenCV 5 rounds some values one level differently: those whose bicubic
/// sum lies within a few thousandths of a half, where its float32 evaluation
/// order decides.
///
/// # Panics
///
/// When `image` is empty, or the quad is too large for its crop to fit in
/// memory.
pub fn crop_text(image: &Image, quad: &Quad) -> Image {
    let width = distance(quad[0], quad[1]).max(distance(quad[2], quad[3])) as usize;
    let height = distance(quad[0], quad[3]).max(distance(quad[1], quad[2])) as usize;
    let m = rectangle_to_quad(width as f64, height as f64, quad);
    let (last_x, last_y) = (image.width as i64 - 1, image.height as i64 - 1);
    let stride = image.width * 3;
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (xf, yf) = (x as f64, y as f64);
            let w = m[6] * xf + m[7] * yf + m[8];
            let w = if w == 0.0 { 0.0 } else { 1.0 / w };
            let sx = ((m[0] * xf + m[1] * yf + m[2]) * w) as f32;
            let sy = ((m[3] * xf + m[4] * yf + m[5]) * w) as f32;
            let (fx, fy) = (sx.floor(), sy.floor());
            let (wx, wy) = (cubic_weights(sx - fx), cubic_weights(sy - fy));
            let (ix, iy) = (fx as i64 - 1, fy as i64 - 1);
            let cols = [0, 1, 2, 3].map(|i| (ix + i).clamp(0, last_x) as usize * 3);
            // Each row is summed first, then the rows: OpenCV's order.
            let mut acc = [0f32; 3];
            for (j, wy) in (0..4).zip(wy) {
                let row = (iy + j).clamp(0, last_y) as usize * stride;
                let row = &image.data[row..row + stride];
                let mut line = [0f32; 3];
                for (&col, wx) in cols.iter().zip(wx) {
                    let px = &row[col..col + 3];
                    line[0] += f32::from(px[0]) * wx;
                    line[1] += f32::from(px[1]) * wx;
                    line[2] += f32::from(px[2]) * wx;
                }
                for (a, l) in acc.iter_mut().zip(line) {
                    *a += l * wy;
                }
            }
            out.extend(acc.map(|v| v.round_ties_even().clamp(0.0, 255.0) as u8));
        }
    }
    let crop = Image::new(width, height, out);
    if height as f64 / width as f64 >= VERTICAL_ASPECT_RATIO {
        rotate_ccw(&crop)
    } else {
        crop
    }
}
