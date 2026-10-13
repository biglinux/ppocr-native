//! Region borders of a binary mask, and the mean of a map inside one.
//!
//! The borders follow Suzuki and Abe's border following ("Topological
//! structural analysis of digitized binary images by border following",
//! 1985), which is what `cv2.findContours` implements: outer borders and hole
//! borders alike, in `RETR_LIST` mode, compressed to their corners as
//! `CHAIN_APPROX_SIMPLE` does.

/// Neighbour steps in counter-clockwise order on screen (y down), starting
/// east.
const STEPS: [(isize, isize); 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const EAST: usize = 0;
const WEST: usize = 4;

/// Pixel states during the scan. Any nonzero state is foreground.
const BACKGROUND: u8 = 0;
const UNSEEN: u8 = 1;
/// On a border, east neighbour not known to be background.
const MARKED: u8 = 2;
/// On a border with background to its east; such a pixel cannot start a hole
/// border.
const MARKED_EAST: u8 = 3;

/// Borders of the nonzero pixels of `mask` (`width` x `height`, row order),
/// as `cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)`
/// returns them: points `[x, y]` at the corners of each border, outer
/// borders counter-clockwise on screen and hole borders clockwise, each
/// starting at its first corner from where the raster scan met it, the
/// borders found last listed first. Pixels on the image edge count as
/// border pixels.
///
/// # Panics
///
/// When `mask` is not `width` x `height`.
pub fn find_contours(mask: &[u8], width: usize, height: usize) -> Vec<Vec<[i32; 2]>> {
    assert_eq!(mask.len(), width * height, "mask size");
    // A frame of background removes every bounds check.
    let stride = width + 2;
    let mut state = vec![BACKGROUND; stride * (height + 2)];
    for (row, src) in mask.chunks_exact(width.max(1)).enumerate().take(height) {
        let dst = &mut state[(row + 1) * stride + 1..(row + 1) * stride + 1 + width];
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = if s != 0 { UNSEEN } else { BACKGROUND };
        }
    }
    let offsets = STEPS.map(|(dx, dy)| dx + dy * stride as isize);
    let step = |at: usize, dir: usize| at.wrapping_add_signed(offsets[dir]);
    let point = |at: usize| [(at % stride) as i32 - 1, (at / stride) as i32 - 1];

    let mut contours = Vec::new();
    let mut chain: Vec<(usize, usize)> = Vec::new();
    for y in 1..=height {
        for x in 1..=width {
            let start = y * stride + x;
            let here = state[start];
            let from = if here == UNSEEN && state[start - 1] == BACKGROUND {
                WEST
            } else if (here == UNSEEN || here == MARKED) && state[start + 1] == BACKGROUND {
                EAST
            } else {
                continue;
            };

            // The border's last pixel: the first foreground neighbour clockwise
            // from the background pixel that revealed the border.
            let Some(last_dir) = (0..8)
                .map(|k| (from + 8 - k) % 8)
                .find(|&d| state[step(start, d)] != BACKGROUND)
            else {
                state[start] = MARKED_EAST;
                contours.push(vec![point(start)]);
                continue;
            };
            let last = step(start, last_dir);

            // Walk counter-clockwise; `chain` holds each pixel and the step
            // that leaves it.
            chain.clear();
            let (mut current, mut back) = (start, last_dir);
            loop {
                let mut east_background = false;
                let mut next_dir = back;
                for k in 1..=8 {
                    let d = (back + k) % 8;
                    if state[step(current, d)] != BACKGROUND {
                        next_dir = d;
                        break;
                    }
                    if d == EAST {
                        east_background = true;
                    }
                }
                if east_background {
                    state[current] = MARKED_EAST;
                } else if state[current] == UNSEEN {
                    state[current] = MARKED;
                }
                chain.push((current, next_dir));
                let next = step(current, next_dir);
                if next == start && current == last {
                    break;
                }
                back = (next_dir + 4) % 8;
                current = next;
            }

            // Keep the pixels where the step direction changes, beginning with
            // the first such pixel at or after the start.
            let turns = |i: usize| chain[i].1 != chain[(i + chain.len() - 1) % chain.len()].1;
            let first = (0..chain.len()).find(|&i| turns(i)).unwrap_or(0);
            let corners = (0..chain.len())
                .map(|i| (first + i) % chain.len())
                .filter(|&i| turns(i))
                .map(|i| point(chain[i].0))
                .collect();
            contours.push(corners);
        }
    }
    contours.reverse();
    contours
}

/// The mean of `map` (`width` x `height`, row order) over the pixels inside
/// or on `contour`, as `cv2.fillPoly` marks them, in `f64`.
///
/// `contour` comes from [`find_contours`], so its edges run along rows,
/// columns or diagonals and meet the pixel grid exactly. A pixel is inside
/// by the even-odd rule along its row.
///
/// # Panics
///
/// When `map` is empty or not `width` x `height`.
pub fn contour_mean(map: &[f32], width: usize, height: usize, contour: &[[i32; 2]]) -> f64 {
    assert_eq!(map.len(), width * height, "map size");
    let clamp_x = |v: i32| v.clamp(0, width as i32 - 1);
    let clamp_y = |v: i32| v.clamp(0, height as i32 - 1);
    let x0 = clamp_x(contour.iter().map(|p| p[0]).min().unwrap_or(0));
    let x1 = clamp_x(contour.iter().map(|p| p[0]).max().unwrap_or(0));
    let y0 = clamp_y(contour.iter().map(|p| p[1]).min().unwrap_or(0));
    let y1 = clamp_y(contour.iter().map(|p| p[1]).max().unwrap_or(0));
    let w = (x1 - x0 + 1) as usize;
    let h = (y1 - y0 + 1) as usize;
    let mut inside = vec![false; w * h];
    let mut mark = |x: i32, y: i32| {
        if (x0..=x1).contains(&x) && (y0..=y1).contains(&y) {
            inside[(y - y0) as usize * w + (x - x0) as usize] = true;
        }
    };

    let edges = || contour.iter().zip(contour.iter().cycle().skip(1));
    for (a, b) in edges() {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let n = dx.abs().max(dy.abs());
        for i in 0..=n {
            mark(a[0] + dx.signum() * i, a[1] + dy.signum() * i);
        }
    }
    let mut crossings = Vec::new();
    for y in y0..=y1 {
        crossings.clear();
        for (a, b) in edges() {
            let (lo, hi) = if a[1] < b[1] { (a, b) } else { (b, a) };
            if lo[1] <= y && y < hi[1] {
                crossings.push(lo[0] + (y - lo[1]) * (hi[0] - lo[0]) / (hi[1] - lo[1]));
            }
        }
        crossings.sort_unstable();
        for &[left, right] in crossings.as_chunks::<2>().0 {
            for x in left..=right {
                mark(x, y);
            }
        }
    }

    let (mut sum, mut count) = (0.0f64, 0usize);
    for row in 0..h {
        let line = &map[(y0 as usize + row) * width + x0 as usize..][..w];
        for (&v, &hit) in line.iter().zip(&inside[row * w..(row + 1) * w]) {
            if hit {
                sum += f64::from(v);
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}
