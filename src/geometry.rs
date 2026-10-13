//! Box geometry on integer contours: minimum-area rectangles in OpenCV's
//! representation, and the round-joined polygon offset that grows a box.
//!
//! The hull's cyclic shift in `hull_order` and the rotating calipers in
//! `calipers` are adapted from OpenCV's `convexHull` and `minAreaRect`
//! (convhull.cpp, rotcalipers.cpp); the corner offsets in `unclip` from
//! Clipper 6.4.2's `ClipperOffset`. Their licences are in NOTICE.

/// Four corners, `[x, y]` each.
pub type Quad = [[f32; 2]; 4];

/// A rectangle in `cv2.RotatedRect` form.
///
/// `angle` is in degrees within [-90, 0): it is the direction of the side
/// called `width`, measured with y pointing down. An upright rectangle is
/// therefore -90 degrees with `width` its vertical extent, as OpenCV 5
/// reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotatedRect {
    pub center: [f32; 2],
    pub width: f32,
    pub height: f32,
    pub angle: f32,
}

impl RotatedRect {
    /// `cv2.boxPoints`, bit for bit: everything in `f32` except the sine and
    /// cosine, evaluated left to right.
    pub fn points(&self) -> Quad {
        let rad = f64::from(self.angle) * std::f64::consts::PI / 180.0;
        let a = rad.sin() as f32 * 0.5;
        let b = rad.cos() as f32 * 0.5;
        let [cx, cy] = self.center;
        let (w, h) = (self.width, self.height);
        [
            [cx - a * h - b * w, cy + b * h - a * w],
            [cx + a * h - b * w, cy - b * h - a * w],
            [cx + a * h + b * w, cy - b * h + a * w],
            [cx - a * h + b * w, cy + b * h + a * w],
        ]
    }
}

fn cross(o: [i32; 2], a: [i32; 2], b: [i32; 2]) -> i64 {
    let d = |p: [i32; 2], q: [i32; 2]| {
        [
            i64::from(q[0]) - i64::from(p[0]),
            i64::from(q[1]) - i64::from(p[1]),
        ]
    };
    let (u, v) = (d(o, a), d(o, b));
    u[0] * v[1] - u[1] * v[0]
}

/// `cv2.convexHull(points)` as indices into `points`, in the order OpenCV
/// returns them; [`min_area_rect`] depends on that order to the last bit.
///
/// The hull runs counter-clockwise in y-up terms without collinear points,
/// starting from the largest point by (x, y). OpenCV then rotates a hull of
/// three points or more so the indices read as one ascending or descending
/// run whenever they can, which makes a convex polygon given in order keep
/// its own first point. Of repeated points the first listed is kept; OpenCV
/// may keep a later copy.
fn hull_order(points: &[[i32; 2]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by_key(|&i| points[i]);
    order.dedup_by_key(|i| points[*i]);
    if order.len() < 3 {
        // Largest first, as the hull below starts.
        order.reverse();
        return order;
    }
    // One side of Andrew's monotone chain, without its last point.
    fn chain(points: &[[i32; 2]], seq: impl Iterator<Item = usize>) -> Vec<usize> {
        let mut h: Vec<usize> = Vec::new();
        for i in seq {
            while h.len() >= 2
                && cross(points[h[h.len() - 2]], points[h[h.len() - 1]], points[i]) <= 0
            {
                h.pop();
            }
            h.push(i);
        }
        h.pop();
        h
    }
    let lower = chain(points, order.iter().copied());
    let mut hull = chain(points, order.iter().rev().copied());
    hull.extend(lower);

    // OpenCV's search for that run, over three points or more: count
    // ascents, give up once the indices both rise and fall more than once,
    // then rotate only if the whole cycle is monotonic from the extreme found.
    let n = hull.len();
    if n < 3 {
        return hull;
    }
    let (mut lowest, mut highest, mut ascents) = (0, 0, 0);
    for i in 1..n {
        ascents += usize::from(hull[i - 1] < hull[i]);
        if ascents > 1 && ascents + 1 < i {
            break;
        }
        if hull[i] < hull[lowest] {
            lowest = i;
        }
        if hull[i] > hull[highest] {
            highest = i;
        }
    }
    let gap = lowest.abs_diff(highest);
    if (gap == 1 || gap == n - 1) && (ascents <= 1 || ascents + 2 >= n) {
        let ascending = (highest + 1) % n == lowest;
        let first = if ascending { lowest } else { highest };
        let rotated: Vec<usize> = (0..n).map(|k| hull[(first + k) % n]).collect();
        if first > 0 && rotated.windows(2).all(|w| (w[0] < w[1]) == ascending) {
            return rotated;
        }
    }
    hull
}

/// OpenCV's rotating calipers over a hull in its order, in `f32` as OpenCV
/// computes them: a corner of the minimum-area rectangle and its two sides.
fn calipers(p: &[[f32; 2]]) -> ([f32; 2], [f32; 2], [f32; 2]) {
    let n = p.len();
    let mut edge = Vec::with_capacity(n);
    let mut inv_len = Vec::with_capacity(n);
    let (mut left, mut bottom, mut right, mut top) = (0, 0, 0, 0);
    for i in 0..n {
        let (q, r) = (p[i], p[(i + 1) % n]);
        if q[0] < p[left][0] {
            left = i;
        }
        if q[0] > p[right][0] {
            right = i;
        }
        if q[1] > p[top][1] {
            top = i;
        }
        if q[1] < p[bottom][1] {
            bottom = i;
        }
        let (dx, dy) = (f64::from(r[0] - q[0]), f64::from(r[1] - q[1]));
        edge.push([dx as f32, dy as f32]);
        inv_len.push((1.0 / (dx * dx + dy * dy).sqrt()) as f32);
    }

    // Callipers bottom, right, top, left. Each step turns the one whose next
    // hull edge, rotated into the bottom calliper's frame, is rightmost by
    // OpenCV's cross-product test; the edges are integers, so ties are exact.
    let mut seq = [bottom, right, top, left];
    let mut min_area = f32::MAX;
    let mut best = (0, 0f32, 0f32, 0f32, 0f32, 0);
    for _ in 0..n {
        let e = seq.map(|i| edge[i]);
        let turned = [
            e[0],
            [e[1][1], -e[1][0]],
            [-e[2][0], -e[2][1]],
            [-e[3][1], e[3][0]],
        ];
        let mut main = 0;
        for i in 1..4 {
            let (v, w) = (turned[i], turned[main]);
            if v[1] * w[0] + -v[0] * w[1] < 0.0 {
                main = i;
            }
        }
        let k = seq[main];
        let (lx, ly) = (edge[k][0] * inv_len[k], edge[k][1] * inv_len[k]);
        let (a, b) = match main {
            0 => (lx, ly),
            1 => (ly, -lx),
            2 => (-lx, -ly),
            _ => (-ly, lx),
        };
        seq[main] = (seq[main] + 1) % n;
        let width = (p[seq[1]][0] - p[seq[3]][0]) * a + (p[seq[1]][1] - p[seq[3]][1]) * b;
        let height = -(p[seq[2]][0] - p[seq[0]][0]) * b + (p[seq[2]][1] - p[seq[0]][1]) * a;
        let area = width * height;
        if area <= min_area {
            min_area = area;
            best = (seq[3], a, width, b, height, seq[0]);
        }
    }

    let (l, a1, width, b1, height, bt) = best;
    let (a2, b2) = (-b1, a1);
    let c1 = a1 * p[l][0] + p[l][1] * b1;
    let c2 = a2 * p[bt][0] + p[bt][1] * b2;
    let inv_det = 1.0 / (a1 * b2 - a2 * b1);
    let corner = [(c1 * b2 - c2 * b1) * inv_det, (a1 * c2 - a2 * c1) * inv_det];
    (corner, [a1 * width, b1 * width], [a2 * height, b2 * height])
}

/// `cv2.minAreaRect` on integer points, as OpenCV 5 computes it.
///
/// The arithmetic is OpenCV's: rotating calipers in `f32` over the hull in
/// the order `cv2.convexHull` returns it, the angle taken in `f64` from the
/// side the calipers end on.
///
/// When the points repeat one, OpenCV's hull can start at another copy of
/// it; the calipers then meet the rectangles in another order, and of two
/// with exactly equal area the other one wins, which can move the corners by
/// 1 to 2 pixels.
pub fn min_area_rect(points: &[[i32; 2]]) -> RotatedRect {
    let hull: Vec<[f32; 2]> = hull_order(points)
        .into_iter()
        .map(|i| [points[i][0] as f32, points[i][1] as f32])
        .collect();
    let length = |dx: f64, dy: f64| (dx * dx + dy * dy).sqrt() as f32;
    // An upright rectangle, a vertical segment and a single point keep -90.
    let mut angle = -std::f64::consts::FRAC_PI_2;
    let (center, width, height) = match hull[..] {
        [] => ([0.0; 2], 0.0, 0.0),
        [p] => (p, 0.0, 0.0),
        [p, q] => {
            let center = [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5];
            let (dx, dy) = (f64::from(p[0] - q[0]), f64::from(p[1] - q[1]));
            let size = length(dx, dy);
            if dx == 0.0 {
                (center, size, 0.0)
            } else if dy < 0.0 {
                angle = dy.atan2(dx);
                (center, size, 0.0)
            } else {
                if dy > 0.0 {
                    angle = -dx.atan2(dy);
                }
                (center, 0.0, size)
            }
        }
        _ => {
            let (corner, u, v) = calipers(&hull);
            let center = [
                corner[0] + (u[0] + v[0]) * 0.5,
                corner[1] + (u[1] + v[1]) * 0.5,
            ];
            let (u_len, v_len) = (
                length(f64::from(u[0]), f64::from(u[1])),
                length(f64::from(v[0]), f64::from(v[1])),
            );
            if u[0] == 0.0 && u[1] > 0.0 {
                (center, u_len, v_len)
            } else {
                angle = -f64::from(u[0]).atan2(f64::from(u[1]));
                (center, v_len, u_len)
            }
        }
    };
    RotatedRect {
        center,
        width,
        height,
        angle: (angle * 180.0 / std::f64::consts::PI) as f32,
    }
}

/// RapidOCR's `get_mini_boxes`: the minimum-area rectangle's corners ordered
/// top-left, top-right, bottom-right, bottom-left (by its own tie rules), and
/// its shorter side.
pub fn mini_box(points: &[[i32; 2]]) -> (Quad, f32) {
    let rect = min_area_rect(points);
    let mut p = rect.points();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let (tl, bl) = if p[1][1] > p[0][1] {
        (p[0], p[1])
    } else {
        (p[1], p[0])
    };
    let (tr, br) = if p[3][1] > p[2][1] {
        (p[2], p[3])
    } else {
        (p[3], p[2])
    };
    ([tl, tr, br, bl], rect.width.min(rect.height))
}

/// Shapely's area and perimeter of a quad, as GEOS computes them: the
/// shoelace relative to the first x, and the sum of the edge lengths.
fn area_and_perimeter(quad: &Quad) -> (f64, f64) {
    let p: Vec<[f64; 2]> = quad
        .iter()
        .map(|q| [f64::from(q[0]), f64::from(q[1])])
        .collect();
    let x0 = p[0][0];
    let mut sum = 0.0;
    for i in 1..4 {
        sum += (p[i][0] - x0) * (p[i - 1][1] - p[(i + 1) % 4][1]);
    }
    let mut perimeter = 0.0;
    for i in 0..4 {
        let (a, b) = (p[i], p[(i + 1) % 4]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    ((sum / 2.0).abs(), perimeter)
}

/// Clipper's rounding to its integer grid: half away from zero, computed as
/// it does by adding a half and truncating.
fn clipper_round(v: f64) -> i32 {
    if v < 0.0 {
        (v - 0.5) as i32
    } else {
        (v + 0.5) as i32
    }
}

/// RapidOCR's unclip: grow `quad` by its area times `ratio` over its
/// perimeter, with rounded corners, as `pyclipper.PyclipperOffset` does with
/// `JT_ROUND`, `ET_CLOSEDPOLYGON` and its default arc tolerance of 0.25.
///
/// pyclipper truncates the corners to integers before offsetting. The arcs
/// are flattened so each chord stays within the arc tolerance of the true
/// arc. Clipper then unions the ring with itself; for this convex input that
/// only drops repeated and collinear points and picks the first point, which
/// is reproduced without running a general union.
pub fn unclip(quad: &Quad, ratio: f64) -> Vec<[i32; 2]> {
    const ARC_TOLERANCE: f64 = 0.25;
    let (area, perimeter) = area_and_perimeter(quad);
    let delta = area * ratio / perimeter;
    let mut ring: Vec<[f64; 2]> = Vec::with_capacity(4);
    for q in quad {
        let p = [f64::from(q[0].trunc()), f64::from(q[1].trunc())];
        if ring.last() != Some(&p) {
            ring.push(p);
        }
    }
    while ring.len() > 1 && ring.first() == ring.last() {
        ring.pop();
    }
    if ring.len() < 3 {
        return Vec::new();
    }
    if delta.abs() < 1e-20 {
        return clipper_cleanup(ring.iter().map(|p| [p[0] as i32, p[1] as i32]).collect());
    }
    let signed_area: f64 = (0..ring.len())
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    if signed_area < 0.0 {
        ring.reverse();
    }

    let tolerance = ARC_TOLERANCE.min(delta.abs() * 0.25);
    let steps = (std::f64::consts::PI / (1.0 - tolerance / delta.abs()).acos())
        .min(delta.abs() * std::f64::consts::PI);
    let (step_sin, step_cos) = (std::f64::consts::TAU / steps).sin_cos();
    let steps_per_rad = steps / std::f64::consts::TAU;

    let n = ring.len();
    let normals: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let (a, b) = (ring[i], ring[(i + 1) % n]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let inv = 1.0 / (dx * dx + dy * dy).sqrt();
            [dy * inv, -dx * inv]
        })
        .collect();
    let at = |p: [f64; 2], n: [f64; 2]| {
        [
            clipper_round(p[0] + n[0] * delta),
            clipper_round(p[1] + n[1] * delta),
        ]
    };

    let mut out = Vec::new();
    let mut k = n - 1;
    for j in 0..n {
        let (p, nk, nj) = (ring[j], normals[k], normals[j]);
        let mut sin_a = nk[0] * nj[1] - nj[0] * nk[1];
        let cos_a = nk[0] * nj[0] + nk[1] * nj[1];
        if (sin_a * delta).abs() < 1.0 {
            // Nearly straight: one point, and the next corner still measures
            // from this edge's predecessor.
            if cos_a > 0.0 {
                out.push(at(p, nk));
                continue;
            }
        } else {
            sin_a = sin_a.clamp(-1.0, 1.0);
        }
        if sin_a * delta < 0.0 {
            out.push(at(p, nk));
            out.push([p[0] as i32, p[1] as i32]);
            out.push(at(p, nj));
        } else {
            let turns = clipper_round(steps_per_rad * sin_a.atan2(cos_a).abs()).max(1);
            let mut v = nk;
            for _ in 0..turns {
                out.push(at(p, v));
                v = [
                    v[0] * step_cos - step_sin * v[1],
                    v[0] * step_sin + v[1] * step_cos,
                ];
            }
            out.push(at(p, nj));
        }
        k = j;
    }
    clipper_cleanup(out)
}

/// The ring as Clipper's union hands it back: repeated and collinear points
/// removed, starting after the last point of the topmost run. Only the order
/// matters downstream, through [`hull_order`].
fn clipper_cleanup(mut ring: Vec<[i32; 2]>) -> Vec<[i32; 2]> {
    while ring.len() >= 3 {
        let n = ring.len();
        let redundant = (0..n).find(|&i| {
            let (prev, here, next) = (ring[(i + n - 1) % n], ring[i], ring[(i + 1) % n]);
            here == next || cross(prev, here, next) == 0
        });
        match redundant {
            Some(i) => {
                ring.remove(i);
            }
            None => break,
        }
    }
    let n = ring.len();
    let Some(top) = ring.iter().map(|p| p[1]).min() else {
        return ring;
    };
    let last = (0..n)
        .find(|&i| ring[i][1] == top && ring[(i + 1) % n][1] != top)
        .unwrap_or(n - 1);
    ring.rotate_left((last + 1) % n.max(1));
    ring
}
