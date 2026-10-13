//! Run or time a PP-OCRv6 plan on RapidOCR's own input tensors.
//!
//! Each input is the raw `[1, 3, h, w]` float32 tensor RapidOCR feeds the
//! model (CHW, already normalized): a text line `FILE`, as high as the
//! plan's input, its width following from the file size, or a page
//! `FILE:HxW`. The model is set to the tensor's size and the tensor
//! transposed to HWC outside the timed region.
//!
//! Usage (`cargo build --release --example plan`, then
//! `target/release/examples/plan`):
//!   plan PLAN run OUT.f32 INPUT...   each output, in order, to OUT
//!                                    (`-` for standard output)
//!   plan PLAN bench INPUT RUNS       CPU ms per run, JSON
//!   plan PLAN profile INPUT RUNS     time by kind of step
//!
//! `PPOCR_NATIVE_TIER=0..3` caps the kernel tier.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use ppocr_native::engine::{simd_tier, Kind, Model, Op};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: plan PLAN run OUT INPUT... | bench INPUT RUNS | profile INPUT RUNS");
        std::process::exit(2);
    }
    let started = Instant::now();
    let mut model = Model::load(Path::new(&args[0])).unwrap_or_else(|e| fail(&e.to_string()));
    let load_ms = started.elapsed().as_secs_f64() * 1e3;
    match args[1].as_str() {
        "run" => {
            let mut out: Box<dyn Write> = if args[2] == "-" {
                Box::new(std::io::BufWriter::new(std::io::stdout().lock()))
            } else {
                let file = std::fs::File::create(&args[2]).unwrap_or_else(|e| fail(&e.to_string()));
                Box::new(std::io::BufWriter::new(file))
            };
            for input in &args[3..] {
                let (h, w, hwc) = read_input(&model, input);
                set_size(&mut model, h, w);
                model.input_mut().copy_from_slice(&hwc);
                model.run();
                let bytes: Vec<u8> = model
                    .output(0)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                out.write_all(&bytes)
                    .unwrap_or_else(|e| fail(&e.to_string()));
            }
            out.flush().unwrap_or_else(|e| fail(&e.to_string()));
        }
        "bench" | "profile" => {
            let (h, w, hwc) = read_input(&model, &args[2]);
            set_size(&mut model, h, w);
            let runs: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(20);
            for _ in 0..3 {
                model.input_mut().copy_from_slice(&hwc);
                model.run();
            }
            if args[1] == "profile" {
                profile(&mut model, &hwc, runs);
                return;
            }
            let mut cpu = Vec::with_capacity(runs);
            for _ in 0..runs {
                model.input_mut().copy_from_slice(&hwc);
                let t0 = cpu_seconds();
                model.run();
                cpu.push((cpu_seconds() - t0) * 1e3);
            }
            cpu.sort_by(f64::total_cmp);
            let macs: u64 = model.ops().iter().map(macs).sum();
            println!(
                "{{\"tier\":{},\"height\":{h},\"width\":{w},\"runs\":{runs},\"load_ms\":{load_ms:.3},\"best_ms\":{:.4},\"median_ms\":{:.4},\"p90_ms\":{:.4},\"gmac\":{:.4},\"scratch_mib\":{:.1},\"pss_mib\":{:.1},\"peak_rss_mib\":{:.1},\"samples_ms\":{:?}}}",
                simd_tier(),
                cpu[0],
                cpu[runs / 2],
                cpu[runs * 9 / 10],
                macs as f64 / 1e9,
                model.scratch_bytes() as f64 / 1048576.0,
                pss_mib(),
                peak_rss_mib(),
                cpu
            );
        }
        _ => fail("expected run, bench or profile"),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("plan: {message}");
    std::process::exit(1);
}

fn set_size(model: &mut Model, h: usize, w: usize) {
    model
        .set_input_size(h, w)
        .unwrap_or_else(|e| fail(&format!("{h}x{w}: {e}")));
}

/// `FILE` or `FILE:HxW`, CHW `[c][h][w]` on disk, as its height, width and
/// HWC values. A bare file is as high as the model's input.
fn read_input(model: &Model, input: &str) -> (usize, usize, Vec<f32>) {
    let size = input.rsplit_once(':').and_then(|(path, size)| {
        let (h, w) = size.split_once('x')?;
        Some((path, h.parse::<usize>().ok()?, w.parse::<usize>().ok()?))
    });
    let (path, height) = match size {
        Some((path, h, _)) => (path, h),
        None => (input, model.input_shape().0),
    };
    let raw = std::fs::read(path).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    let values: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let (h, c) = (height, model.input_shape().2);
    let w = size.map_or(values.len() / (h * c).max(1), |(_, _, w)| w);
    if w * h * c != values.len() || w == 0 {
        fail(&format!("{path}: not a [1, {c}, {h}, {w}] tensor"));
    }
    let mut hwc = vec![0.0; h * w * c];
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                hwc[(y * w + x) * c + ch] = values[(ch * h + y) * w + x];
            }
        }
    }
    (h, w, hwc)
}

fn macs(op: &Op) -> u64 {
    let spatial = (op.oh * op.ow) as u64;
    match op.kind {
        Kind::Conv => (op.kh * op.kw * (op.ic + op.n2) * op.oc) as u64 * spatial,
        Kind::Dwconv => (op.kh * op.kw * op.oc) as u64 * spatial,
        // Scores and weighted values: two t-by-t products per head dimension.
        Kind::Attention => 2 * (op.iw * op.iw * op.oc) as u64,
        Kind::Ctchead => (op.iw * op.ic * op.oc) as u64,
        Kind::Ctcargmax => (op.iw * op.ic * op.n2) as u64,
        Kind::Convtranspose => (op.kh * op.kw * op.ic * op.oc) as u64 * (op.ih * op.iw) as u64,
        _ => 0,
    }
}

fn label(op: &Op) -> String {
    let fused = format!("{:?}", op.fused).to_lowercase();
    let fused = if fused == "none" {
        String::new()
    } else {
        format!("+{fused}")
    };
    match op.kind {
        Kind::Conv if op.kh * op.kw == 1 && op.oh * op.ow == 1 => "squeeze-excite 1x1".into(),
        Kind::Conv if op.kh * op.kw == 1 && op.oh == 1 => format!("neck 1x1{fused}"),
        Kind::Conv if op.kh * op.kw == 1 => {
            format!("pointwise{fused} {}->{} x{}", op.ic, op.oc, op.oh * op.ow)
        }
        Kind::Conv => format!(
            "conv {}x{} s{} {}->{} {}x{}",
            op.kh,
            op.kw,
            op.sh,
            op.ic + op.n2,
            op.oc,
            op.oh,
            op.ow
        ),
        Kind::Dwconv => format!(
            "depthwise {}x{} s{}x{} c{} {}x{}",
            op.kh, op.kw, op.sh, op.sw, op.oc, op.oh, op.ow
        ),
        Kind::Convtranspose => format!(
            "transposed {}x{} {}->{}{fused} {}x{}",
            op.kh, op.kw, op.ic, op.oc, op.oh, op.ow
        ),
        other => format!("{other:?}").to_lowercase(),
    }
}

fn profile(model: &mut Model, hwc: &[f32], runs: usize) {
    let mut spent: BTreeMap<String, (f64, u64)> = BTreeMap::new();
    let mut times = Vec::new();
    for _ in 0..runs {
        model.input_mut().copy_from_slice(hwc);
        model.run_timed(&mut times, || thread_cpu_seconds() * 1e3);
        // A step that runs several operations together counts as one kind.
        for (ops, ms) in &times {
            let ops = &model.ops()[ops.clone()];
            let name: Vec<String> = ops.iter().map(label).collect();
            let row = spent.entry(name.join(" + ")).or_default();
            row.0 += ms;
            row.1 += ops.iter().map(macs).sum::<u64>();
        }
    }
    let total: f64 = spent.values().map(|r| r.0).sum();
    let mut rows: Vec<_> = spent.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    println!(
        "per run {:.3} ms of thread CPU (clock read around every step)",
        total / runs as f64
    );
    for (name, (ms, macs)) in rows {
        let per = ms / runs as f64;
        let rate = if macs > 0 {
            format!(
                "{:6.1} GMAC/s",
                macs as f64 / runs as f64 / (per * 1e-3) / 1e9
            )
        } else {
            String::new()
        };
        println!(
            "{per:9.3} ms {:5.1}%  {rate:>14}  {name}",
            100.0 * ms / total
        );
    }
}

fn pss_mib() -> f64 {
    let text = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    text.lines()
        .filter(|l| l.starts_with("Pss:"))
        .filter_map(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        .sum::<f64>()
        / 1024.0
}

fn peak_rss_mib() -> f64 {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    text.lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        .unwrap_or(0.0)
        / 1024.0
}

fn cpu_seconds() -> f64 {
    clock(2) // CLOCK_PROCESS_CPUTIME_ID
}

fn thread_cpu_seconds() -> f64 {
    clock(3) // CLOCK_THREAD_CPUTIME_ID
}

#[allow(unsafe_code)] // clock_gettime, which std does not wrap for CPU time
fn clock(id: i32) -> f64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
    }
    let mut ts = Timespec { sec: 0, nsec: 0 };
    // SAFETY: `ts` is writable with the Linux 64-bit timespec layout.
    let rc = unsafe { clock_gettime(id, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime failed");
    ts.sec as f64 + ts.nsec as f64 * 1e-9
}
