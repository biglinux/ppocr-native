//! The exported networks give the same bits on any number of threads.
//!
//! It needs the exported plans, which the repository does not keep: run it
//! with `--include-ignored` and `PPOCR_NATIVE_PLANS` naming them.

use std::path::PathBuf;

use ppocr_native::engine::Model;

fn plan(name: &str) -> PathBuf {
    let dir = std::env::var_os("PPOCR_NATIVE_PLANS").expect("PPOCR_NATIVE_PLANS is not set");
    PathBuf::from(dir).join(name)
}

/// The output of `model` at `h` by `w` on `threads` threads, as bits, from
/// an input of smooth stripes that every layer turns into busy maps.
fn bits(model: &mut Model, h: usize, w: usize, threads: usize) -> Vec<Vec<u32>> {
    model.set_threads(threads).unwrap();
    model.set_input_size(h, w).unwrap();
    for (i, v) in model.input_mut().iter_mut().enumerate() {
        *v = ((i % 977) as f32 * 0.37).sin() * 2.0;
    }
    model.run();
    (0..model.output_count())
        .map(|o| model.output(o).iter().map(|v| v.to_bits()).collect())
        .collect()
}

#[test]
#[ignore = "needs PPOCR_NATIVE_PLANS"]
fn networks_match_on_any_threads() {
    let cases = [
        ("PP-OCRv6_det_small.plan", 960, 736),
        ("PP-OCRv6_det_small.plan", 1984, 1408),
        ("PP-OCRv6_det_small.plan", 96, 2016),
        ("PP-OCRv6_rec_small_argmax.plan", 48, 640),
    ];
    for (name, h, w) in cases {
        let mut model = Model::load(&plan(name)).unwrap();
        let one = bits(&mut model, h, w, 1);
        for threads in [4, 7] {
            assert!(
                bits(&mut model, h, w, threads) == one,
                "{name} at {h}x{w} on {threads} threads"
            );
        }
    }
}
