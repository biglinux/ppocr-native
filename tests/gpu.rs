//! The networks on a GPU read as they do on the CPU: the same classes, and
//! values within a thousandth, one image at a time or several together.
//!
//! It needs the exported plans and a Vulkan GPU: run it with
//! `--include-ignored` and `PPOCR_NATIVE_PLANS` naming the plans. A machine
//! without a GPU skips it with `--skip gpu_`.

use std::path::PathBuf;

use ppocr_native::engine::{GpuModel, Model};

fn plan(name: &str) -> PathBuf {
    let dir = std::env::var_os("PPOCR_NATIVE_PLANS").expect("PPOCR_NATIVE_PLANS is not set");
    PathBuf::from(dir).join(name)
}

/// Smooth stripes, which every layer turns into busy maps.
fn stripes(input: &mut [f32], phase: usize) {
    for (i, v) in input.iter_mut().enumerate() {
        *v = (((i + phase) % 977) as f32 * 0.37).sin() * 2.0;
    }
}

/// The largest difference between two outputs of the same length, all of
/// whose values must be finite.
fn largest_gap(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite(), "{x} against {y}");
            (x - y).abs()
        })
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs PPOCR_NATIVE_PLANS and a Vulkan GPU"]
fn gpu_detector_matches_cpu() {
    let path = plan("PP-OCRv6_det_small.plan");
    let mut cpu = Model::load(&path).unwrap();
    let mut gpu = GpuModel::load(&path).unwrap();
    for (h, w) in [(960, 736), (96, 2016)] {
        cpu.set_input_size(h, w).unwrap();
        gpu.set_input_size(h, w).unwrap();
        stripes(cpu.input_mut(), 0);
        stripes(gpu.input_mut(), 0);
        cpu.run();
        gpu.run().unwrap();
        let gap = largest_gap(cpu.output(0), gpu.output(0));
        assert!(gap < 1e-3, "probabilities {gap} apart at {h}x{w}");
    }
}

#[test]
#[ignore = "needs PPOCR_NATIVE_PLANS and a Vulkan GPU"]
fn gpu_recogniser_lanes_match_cpu() {
    let path = plan("PP-OCRv6_rec_small_argmax.plan");
    let mut cpu = Model::load(&path).unwrap();
    let mut gpu = GpuModel::load(&path).unwrap();
    let widths = [640, 96, 1280];
    gpu.set_sizes(&widths.map(|w| (48, w))).unwrap();
    for lane in 0..widths.len() {
        stripes(gpu.lane_input_mut(lane), lane);
    }
    gpu.run().unwrap();
    for (lane, &w) in widths.iter().enumerate() {
        cpu.set_input_size(48, w).unwrap();
        stripes(cpu.input_mut(), lane);
        cpu.run();
        let (c, g) = (cpu.output(0), gpu.lane_output(lane, 0));
        let classes = |o: &[f32]| o.iter().step_by(2).copied().collect::<Vec<f32>>();
        assert!(classes(c) == classes(g), "classes differ at width {w}");
        let probabilities = |o: &[f32]| o.iter().skip(1).step_by(2).copied().collect::<Vec<f32>>();
        let gap = largest_gap(&probabilities(c), &probabilities(g));
        assert!(gap < 1e-3, "probabilities {gap} apart at width {w}");
    }
    // A lane alone reads as it does beside others.
    let together = gpu.lane_output(1, 0).to_vec();
    gpu.set_input_size(48, 96).unwrap();
    stripes(gpu.input_mut(), 1);
    gpu.run().unwrap();
    assert!(gpu.output(0) == together.as_slice());
}
