# ppocr-native

PP-OCRv6 text detection and recognition in plain Rust, giving the same
results as RapidOCR with the settings bigocrpdf uses. It is bigocrpdf's OCR
engine: a worker that reads page images and answers in JSON lines, and a
library with every step in between.

- **No dependencies.** The library and the worker use only `std`: no ONNX
  Runtime, OpenCV, numpy, pyclipper or shapely.
- **Its own inference engine.** Both networks run on the CPU from a plan
  exported once from RapidOCR's ONNX files, with kernels chosen for the CPU
  when the process starts. One plan serves every page and line size.
- **The same arithmetic.** Each image step follows what RapidOCR 3.9
  computes with OpenCV 5, numpy 2, pyclipper 1.4 and shapely 2, down to
  OpenCV's fixed-point resize, its float32 rotating calipers, Clipper's arc
  steps and numpy's pairwise summation.

Build with `cargo build --release --locked`; check a change with
`cargo test`, and with exported plans `PPOCR_NATIVE_PLANS=plans cargo test
--release -- --include-ignored` (see [Development](#development)).

---

## Install

On BigLinux, or Arch Linux with the BigLinux repositories:

```bash
sudo pacman -S ppocr-native
```

The package installs the worker as `/usr/bin/ppocr-native` and the plans and
character list in `/usr/share/ppocr-native`.

## Build from source

The worker needs Rust 1.93 or later. The plans it runs are exported from
the PP-OCRv6 small ONNX models RapidOCR distributes, which on BigLinux come
in `rapidocr-models-v6-small` (installed in `/usr/share/rapidocr/models`).
The export needs Python with onnx and numpy (`python-onnx`, `python-numpy`):

```bash
cargo build --release --locked --bin ppocr-native

models=/usr/share/rapidocr/models
mkdir -p plans
tools/export_ocr.py $models/PP-OCRv6_det_small.onnx plans/PP-OCRv6_det_small.plan
tools/export_ocr.py $models/PP-OCRv6_rec_small.onnx plans/PP-OCRv6_rec_small_argmax.plan --argmax
tools/export_dictionary.py $models/PP-OCRv6_rec_small.onnx plans/PP-OCRv6_rec_small.characters.txt

PPOCR_NATIVE_PLANS=plans target/release/ppocr-native page.ppm
```

The repository keeps no plans or model files: they are 31 MB of binaries
under the models' licence, and exporting them from the installed models
keeps them matched to those weights. `pkgbuild/PKGBUILD` does the same
steps.

`--argmax` makes the recogniser output each time step's best class and its
probability, which is what the worker reads, instead of the softmax over all
18 710 classes. `src/engine/plan.rs` describes the plan format and checks
every plan before anything runs.

## The worker

```bash
ppocr-native [OPTIONS] IMAGE...      # one JSON line per image
ppocr-native --persistent [OPTIONS]  # requests on standard input
```

Images are binary 8-bit PPM or PGM, as `pdftoppm` writes them. A page's
answer is `{"boxes": [...], "txts": [...], "scores": [...]}`, with
`"vertical_reoriented": N` when N vertical lines read better turned half a
turn, `{"boxes": null}` for a page without text, or `{"error": "..."}`.

A persistent worker first answers `{"ready": true, "tier": N}` with the
kernel tier it runs, or `{"fatal": "..."}` and exits. Then each request line
gets one answer line, until standard input closes:

| Request | Answer |
|---|---|
| `PATH` | the page's text, as above |
| `detect<TAB>PATH<TAB>MAX_SIDE` | `{"boxes": [...]}`, the detector's boxes alone on the page fitted within `MAX_SIDE` pixels |
| `probmap<TAB>PATH<TAB>OUT` | `{"height": H, "width": W}`, after writing to `OUT` the detector's probability map of an image already at its input size, as little-endian `f32` rows |

bigocrpdf uses the last two for dewarping, with `--detect-only`, which
loads the detector alone. Plans are read from `PPOCR_NATIVE_PLANS`, by
default `/usr/share/ppocr-native`, and checked before the ready line.
`src/bin/ppocr-native/main.rs` lists every option.

`--threads N` sets how many recognisers read a page's lines side by side,
by default the CPUs the process may use; detection runs on one thread. On
30 DharmaOCR pages on a Ryzen 7 5700U (8 cores, 16 threads, 35 W), one
worker took 215 s on one thread, 95 s on four, 89 s on eight and 97 s on
sixteen, with peak RSS rising from 461 MiB on one thread to 1639 MiB on
sixteen; RapidOCR on OpenVINO with 16 threads took 137 s at 2451 MiB.
bigocrpdf runs one worker with a recogniser per physical core. A model keeps
the tensors of a run in one block, each holding memory only from the step
that writes it to the last that reads it; the detector alone peaks at
27 MiB for a 480x448 input, 184 MiB at 1984x1344 and 1087 MiB at the
4096x4096 limit.

## CPU tiers

x86_64 builds carry four sets of kernels and pick one when the process
starts: AVX2 with FMA (tier 3), AVX (2), SSE4.1 (1) or plain Rust (0).
`PPOCR_NATIVE_TIER=0..3` caps the tier, to run an older machine's code on a
newer one. Results are bit-identical across runs and machines within a
tier, not across tiers. Other architectures, aarch64 included, run tier 0.

## What it covers

PP-OCR reads a page with two networks: a DB detector that marks where text
is, and a CTC recogniser that reads each line. Around them, the page is
resized, the detector's probability map becomes boxes, each box is
straightened into a crop, each crop becomes a recogniser input, and the
recogniser's output becomes text and a score. Each of those steps is a
public function, checked against RapidOCR on 30 pages:

| Step | Function | Matches RapidOCR on 30 pages |
|---|---|---|
| Page limits (4096 px, 32-pixel multiples) | `prepare_page`, `fit_page` | bit for bit |
| Detector input size and tensor | `det_size`, `det_input` | bit for bit |
| Contours, rectangles, scores, unclip, filter, order | `boxes_from_prob` | 1301 of 1310 boxes identical, 1305 with IoU ≥ 0.99 |
| Crop a box upright | `crop_text` | 1184 of 1310 crops identical, the rest one level off on 0.003 % of pixels |
| Recogniser input | `rec_input` | bit for bit |
| CTC decoding, text and score | `ctc_decode`, `ctc_decode_probs` | bit for bit |
| Lines on the original page | `page_lines` | bit for bit |

The nine boxes that differ, five of them below an IoU of 0.99, come from
near-ties inside OpenCV's `minAreaRect` that it settles the other way (see
`min_area_rect`); they move a box edge by a pixel or two. The crop
differences are values whose bicubic sum sits next to a half. Outside the
reference set the rates are alike: 863 of 866 boxes identical on 19 pages,
and crop values one level off on 0.007 to 0.026 % of them depending on the
content, never by more than one level. Run end to end with the recogniser
on ONNX Runtime, 1290 of the 1295 lines read the same text; the other five
are the lines whose box moved.

Not covered: the detector's `fast` score mode (bigocrpdf uses `slow`),
dilation, the text-line classifier and batches larger than one. Two kinds
of thin page are answered differently from RapidOCR, to keep memory bounded:
`prepare_page` refuses a page whose padded detector input would exceed
4096² pixels, and the worker answers it as having no text; with
`--full-resolution`, a page whose short side scaled up to 960 would pass
that size is detected at the size used without the option. Re-reading
vertical lines is the worker's, not the library's.

## Using the library

```rust,no_run
use ppocr_native::{
    boxes_from_prob, crop_text, ctc_decode, det_input, det_size, page_lines, prepare_page,
    rec_input, DbOptions, Dictionary, Image, Limit, Line, PageOptions,
};

/// `detect` runs the detector on a 1x3xHxW tensor and returns the HxW map;
/// `recognise` runs the recogniser on a 1x3x48xW tensor and returns the best
/// class and its probability at each step.
fn read_page(
    page: &Image,
    dictionary: &Dictionary,
    detect: impl Fn(&[f32], usize, usize) -> Vec<f32>,
    recognise: impl Fn(&[f32], usize) -> (Vec<u32>, Vec<f32>),
) -> Option<Vec<Line>> {
    let prepared = prepare_page(page, &PageOptions::default())?;
    let image = &prepared.image;
    let (w, h) = det_size(image.width, image.height, Limit::Max)?;
    let tensor = det_input(image, w, h, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
    let prob = detect(&tensor, w, h);
    let boxes = boxes_from_prob(&prob, w, h, image.width, image.height, &DbOptions::default());
    let reads = boxes
        .iter()
        .map(|quad| {
            let (input, width) = rec_input(&crop_text(image, quad));
            let (best, prob) = recognise(&input, width);
            ctc_decode(&best, &prob, dictionary)
        })
        .collect();
    Some(page_lines(&prepared, page.width, page.height, &boxes, reads, 0.3))
}
```

The dictionary is the recogniser's `character` metadata, written by
`tools/export_dictionary.py` and read by `Dictionary::from_metadata`.

`engine::Model` runs a plan; the engine reads HWC tensors, so RapidOCR's CHW
ones are transposed on the way in:

```rust,no_run
use ppocr_native::engine::{Error, Model};

/// The `recognise` step above on a model loaded once with
/// `Model::load(Path::new("plans/PP-OCRv6_rec_small_argmax.plan"))`.
fn recognise(rec: &mut Model, chw: &[f32], width: usize) -> Result<(Vec<u32>, Vec<f32>), Error> {
    rec.set_input_size(48, width)?;
    let plane = 48 * width;
    for (i, v) in rec.input_mut().iter_mut().enumerate() {
        *v = chw[(i % 3) * plane + i / 3];
    }
    rec.run();
    Ok(rec.output(0).chunks_exact(2).map(|s| (s[0] as u32, s[1])).unzip())
}
```

## Where things live

| Path | Holds |
|---|---|
| `src/*.rs` | the image and geometry steps of the table above |
| `src/engine/mod.rs` | `Model`: memory, running the steps |
| `src/engine/plan.rs` | the plan format, its validation and size rules |
| `src/engine/schedule.rs` | which operations run together, and where each tensor lives |
| `src/engine/conv.rs`, `layers.rs` | whole layers: convolution and pooling, then attention, CTC head, layer norm, upsampling, activations |
| `src/engine/kernels/` | the tiered SIMD kernels, one file per family |
| `src/bin/ppocr-native/` | the worker: protocol, page reading, PPM decoding |
| `tools/export_ocr.py`, `export_dictionary.py` | ONNX to plan and character list |
| `tools/`, `examples/` | parity checks, ISA check and benchmarks, below |

## Development

```bash
cargo test
```

compares every step with fixtures written by `tools/make_fixtures.py` from
OpenCV, pyclipper and RapidOCR themselves, 1.6 MB in all. With exported
plans, the ignored tests in `tests/worker.rs` drive the worker through each
request, check that it reads RapidOCR's text, and check that a thin strip
under `--full-resolution` is answered without the oversized input:

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
cargo test --release --target aarch64-unknown-linux-gnu -- --include-ignored
```

### Parity with RapidOCR

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
the 30 reference pages, 14 in 100 000 on 19 others and 1 to 2 % on random
noise.

The networks are checked against ONNX Runtime on RapidOCR's own input
tensors, at each tier:

```bash
PPOCR_NATIVE_TIER=2 tools/check_ocr_parity.py MODEL.onnx PLAN INPUT_DIR
```

It passes when every line reads the same text with at least 99.9 % of time
steps agreeing, and every page's probability map stays within 1e-3 with at
most 1e-4 of its pixels binarized differently.

### Timing the engine

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
for exactly 16 outputs is 16 % faster for 8->16 channels on the first and
12 % on the second. On the i3-2375M, the AVX tile of two pixels by 32
outputs is 2 to 14 % faster than four by 16 while the weights fit in L2,
and 8 to 11 % slower past it (112->672, 192->1152, 1152->192); the copy that
spreads a pixel tile over L1 banks loses with two 16-column panels and pays
from four; opaque pointers made per tile were 10 % slower than per-call
offsets at 16 input channels. Blocking the dense convolution's taps to keep
their weights in L1 moved layers by 2 % either way.

## License

By Bruno Gonçalves ([BigLinux](https://www.biglinux.com.br)), licensed MIT
OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)). [NOTICE](NOTICE) names the projects whose
behaviour this reproduces, and carries the licences of the OpenCV and
Clipper code that parts of `src/geometry.rs` and `src/crop.rs` are adapted
from. The plans the package installs are exported from PaddleOCR's models
and keep their Apache-2.0 licence.
