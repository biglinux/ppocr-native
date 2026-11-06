# Development

```bash
cargo test
```

compares every step with fixtures written by `tools/make_fixtures.py` from
OpenCV, pyclipper and RapidOCR themselves, 1.6 MB in all. With exported
plans, the ignored tests in `tests/worker.rs` drive the worker through each
request, check that it reads RapidOCR's text, and check that a thin strip
under `--full-resolution` is answered without the oversized input; those
in `tests/gpu.rs` check that a GPU reads as the CPU does, and a machine
without one skips them with `--skip gpu_`:

```bash
PPOCR_NATIVE_PLANS=plans cargo test --release -- --include-ignored
for tier in 0 1 2; do
    PPOCR_NATIVE_TIER=$tier PPOCR_NATIVE_PLANS=plans cargo test --release -- --include-ignored
done
tools/check_isa.py target/release/ppocr-native
```

The loop runs the same tests on each older tier's kernels, and
`check_isa.py` confirms from the disassembly (`objdump`, in `binutils`)
that each tier's kernels use only that tier's instructions. The package
build runs all three. Lints: `cargo fmt --check`, `cargo clippy
--all-targets -- -D warnings`, and `ruff check tools` with `ruff format
--check tools`.

On aarch64 the same tests pass on tier 0. Without the hardware, cross-build
them and run them under qemu-user (`aarch64-linux-gnu-gcc`,
`qemu-user-binfmt`; the binfmt handler lets `tests/worker.rs` start the
worker):

```bash
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
QEMU_LD_PREFIX=/usr/aarch64-linux-gnu PPOCR_NATIVE_PLANS=plans \
cargo test --release --target aarch64-unknown-linux-gnu -- --include-ignored --skip gpu_
```

## Parity with RapidOCR

`make_fixtures.py`, `dump_reference.py`, `check_parity.py` and
`check_ocr_parity.py` check the crate against the libraries it reproduces,
so they need what the crate does not: RapidOCR with ONNX Runtime and the
PP-OCRv6 small models (`python-rapidocr`, which brings OpenCV, pyclipper,
shapely and the models, plus `python-onnxruntime-cpu`). Neither the build
nor the worker uses them. The full comparison runs on a reference set
recorded from RapidOCR:

```bash
tools/dump_reference.py MANIFEST IMAGE_DIR REF [--models DIR]
cargo run --release --example parity_dump -- REF IMAGE_DIR characters.txt OUT
tools/check_parity.py REF OUT --e2e PP-OCRv6_rec_small.onnx
```

Each step starts from the reference's own input to it, so a difference
points at one step; `--e2e` also reads the crate's own crops with ONNX
Runtime and compares the final lines. The crops are the one step that is
not bit-identical: OpenCV 5 rounds a bicubic sum within a few thousandths of
a half by its float32 evaluation order, which moves 3 in 100 000 values on
the 30 reference pages, 14 in 100 000 on 19 others and 1 to 2% on random
noise.

The networks are checked against ONNX Runtime on RapidOCR's own input
tensors, at each tier:

```bash
PPOCR_NATIVE_TIER=2 tools/check_ocr_parity.py MODEL.onnx PLAN INPUT_DIR
```

It passes when every line reads the same text with at least 99.9% of time
steps agreeing, and every page's probability map stays within 1e-3 with at
most 1e-4 of its pixels binarized differently.

## Timing the engine

The `plan` example runs or times a plan on RapidOCR's input tensors (CHW
float32 files, a line `FILE` or a page `FILE:HxW`), and
`tools/bench_ocr_abba.py` times two builds against each other on one CPU:

```bash
cargo build --release --example plan
target/release/examples/plan PLAN run OUT.f32 INPUT...   # outputs, in order
target/release/examples/plan PLAN bench INPUT RUNS       # CPU ms per run, JSON
target/release/examples/plan PLAN profile INPUT RUNS     # time by kind of step
tools/bench_ocr_abba.py --cpu 4 --input LINE.f32 --output bench.json \
    --native old=old/plan:PLAN --native new=target/release/examples/plan:PLAN
```

The kernel choices in `src/engine/kernels/` were timed on an i3-2375M
(Sandy Bridge, AVX, 256 KB L2) and an i5-13400. The pointwise tile's copy
for exactly 16 outputs is 16% faster for 8->16 channels on the first and
12% on the second. On the i3-2375M, the AVX tile of two pixels by 32
outputs is 2 to 14% faster than four by 16 while the weights fit in L2,
and 8 to 11% slower past it (112->672, 192->1152, 1152->192); the copy that
spreads a pixel tile over L1 banks loses with two 16-column panels and pays
from four; opaque pointers made per tile were 10% slower than per-call
offsets at 16 input channels. Blocking the dense convolution's taps to keep
their weights in L1 moved layers by 2% either way.

## The GPU kernels

Measured on 10 pages at 300 dpi, on a Radeon Vega 7 beside a Ryzen 5 5600H
and a GeForce GTX 1660 Ti beside a Core i5-13400:

- An integrated GPU reads 40% of a page's line width: on the Vega 7, every
  line on the cores took 14.1 s; a GPU share of 0.4 took 11.3 s and 0.5 took
  12.5 to 13.5 s.
- The matrix kernel takes eight values of the reduction a step; 16 and 32 are
  slower on both GPUs, and 4 is slower on the Vega.
- Shared tiles read as `vec4` are faster on the GTX 1660 Ti and slower on the
  Vega (detector 268 to 274 ms against 311 to 315 ms), so AMD reads them one
  value at a time.
- Slower on the Vega 7, and left out: half-float tensors (15% faster, but 49
  of 440 boxes moved), weights read through the scalar cache (detector twice
  as long), two 1x1 convolutions passing their product through shared memory
  (267 to 270 ms against 308 to 312 ms), register tiles of 8x4 or 8x8, a max
  pool computed inside the convolution that reads it, and keys and values of
  the attention in shared memory (the page 10% slower; 1% faster on the GTX
  1660 Ti, within noise).
