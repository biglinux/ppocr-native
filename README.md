# ppocr-native

PP-OCRv6 text detection and recognition in plain Rust: the same text as
RapidOCR, without Python, ONNX Runtime or OpenCV. It is bigocrpdf's OCR
engine: a worker that reads page images and answers in JSON lines, and a
library with every step in between.

- No crates beyond Rust's standard library; `--gpu` opens the system's
  `libvulkan.so.1` at run time.
- Both networks run on our own engine, on the CPU or, with `--gpu`, on a
  GPU with a Vulkan 1.2 driver (a software one such as llvmpipe doesn't
  count).
- Every image step reproduces RapidOCR's arithmetic, down to OpenCV's
  fixed-point resize and numpy's pairwise sums.

## How fast is it?

Ten pages of a printer manual rendered at 300 dpi, 440 lines of text, read
by the worker and by RapidOCR 3.9.2 configured the same way (PP-OCRv6 small,
6 threads, median of 2 runs). RapidOCR runs on ONNX Runtime 1.31.1 or
OpenVINO 2026.4.1, under Python 3.14.

| | Laptop | Desktop |
|---|---|---|
| CPU | AMD Ryzen 5 5600H (6 cores, 12 threads) | Intel Core i5-13400 (6 performance + 4 efficient cores) |
| RAM | 40 GB DDR4-3200 | 32 GB DDR5-5200 |
| GPU | AMD Radeon Vega 7 (integrated) | NVIDIA GeForce GTX 1660 Ti, 6 GB |
| Drivers | Mesa 26.2 (RADV) | NVIDIA 615.78 |
| OS | BigLinux with BigDesktop, Linux 7.2 | BigLinux with BigDesktop, Linux 7.2 |

### Laptop (Ryzen 5 5600H, Radeon Vega 7)

| 10 pages | Time | CPU time | Memory |
|---|---:|---:|---:|
| ppocr-native, `--gpu` | **9.8 s** | 41 s | **388 MiB** |
| ppocr-native, CPU | 13.0 s | 70 s | 506 MiB |
| RapidOCR + OpenVINO | 21.8 s | 111 s | 2002 MiB |
| RapidOCR + ONNX Runtime | 22.4 s | 129 s | 854 MiB |

1.7 times faster than RapidOCR on the CPU alone, 2.2 times with the
integrated GPU. OpenVINO matches ONNX Runtime's time and uses 2 GB for the
privilege.

### Desktop (Core i5-13400, GeForce GTX 1660 Ti)

Pinned to one thread per performance core; RapidOCR was not measured on this
machine.

| 10 pages | Time | CPU time | Memory |
|---|---:|---:|---:|
| ppocr-native, `--gpu` | **5.0 s** | **5 s** | 440 MiB |
| ppocr-native, CPU | 9.0 s | 48 s | 513 MiB |

### Older machines

| 10 pages | ppocr-native, CPU | ppocr-native, `--gpu` |
|---|---:|---:|
| Ryzen 7 5700U (8 threads), Radeon Vega 8 | 19.6 s | 14.7 s |
| Core i3-2375M (Sandy Bridge, 4 threads, no Vulkan GPU) | 166 s | falls back to the CPU |

### Install size and start-up

| | ppocr-native | RapidOCR + ONNX Runtime | RapidOCR + OpenVINO |
|---|---:|---:|---:|
| Libraries | 1.4 MB, GPU support included | 370 MB of Python packages | 550 MB |
| Models | 31 MB | 31 MB | 31 MB |
| Ready to read, laptop | 0.07 s (0.13 s with `--gpu`) | 0.47 s | 0.67 s |

The Python figures leave out Python itself.

### Are the results the same?

On the CPU, yes, within each instruction tier:

- Every page step was checked against RapidOCR on 30 reference pages: page
  limits, detector input, recogniser input, CTC decoding and line placement
  match bit for bit. 1301 of 1310 boxes are identical; the other nine come
  from near-ties inside OpenCV's `minAreaRect` and move an edge by a pixel
  or two. 1184 of 1310 crops are identical, the rest one level off on 0.003%
  of their pixels.
- On the 10 pages above, all 440 lines read the same text as RapidOCR with a
  recogniser batch of 1. Its default batch of 6 pads each line to the
  widest in the batch, which changed 5 lines and made it slower, not faster
  (37 s on the laptop).
- Any thread count gives the same bytes, and so does any machine running the
  same tier (AVX2+FMA, AVX, SSE4.1 or plain Rust); different tiers round a
  few sums differently. Other architectures, aarch64 included, run plain
  Rust.

With `--gpu`, sums run in a different order: the detector's probabilities
move by up to 4e-5 and scores by up to 1e-5. On the 10 pages, the boxes and
the text were the same as on the CPU on all three GPUs above.

## Why is it fast?

### On the CPU

- **Hand-written kernels** in AVX2+FMA, AVX, SSE4.1 and plain Rust, picked
  at start-up (`PPOCR_NATIVE_TIER=N` forces an older one). The pointwise
  tile does six pixels by sixteen channels, about 58 billion multiply-adds a
  second on one Zen 3 core.
- **No im2col.** Convolutions read their windows in place; the neck's
  upsampling and channel concatenation are read by the convolution after
  them instead of being written out.
- **Fused layers.** Depthwise convolutions feed the pointwise ones a group of
  rows at a time, squeeze-and-excite gates, activations and residual sums
  are applied while the output is still in cache.
- **Memory that comes and goes.** Each tensor holds memory only from the
  step that writes it to the last that reads it.
- **Deterministic threads.** Layers are split by rows, never inside a sum,
  so 1 thread and 16 give the same bytes. The input is resized and
  normalized on every thread too.
- **No subnormal weights.** A tenth of the detector's weights are subnormal
  floats, and recent Intel cores take a microcode assist for each product
  with one, which makes the detector five times slower on an i5-13400. They
  load as zero, and no sum notices.

### On the GPU (Vulkan)

- **Plain Vulkan compute shaders**, with `libvulkan.so.1` opened at run time.
  No SDK, no CUDA, no ROCm; without a GPU the worker says so and uses the
  CPU.
- **32-bit floats.** Half floats ran 15% faster on the Vega 7 and moved 49 of
  the 440 boxes, so they stayed out.
- **Fused layers here too.** Convolutions take in the squeeze-and-excite
  product before them, the residual sum after them and the neck's
  upsampling and concatenation; a 7x7 depthwise kernel keeps its 49 weights
  in registers.
- **Different reads for different GPUs.** AMD GPUs read the shared tiles one
  float at a time, which is faster on the Vega 7; the others read four at a
  time, which is faster on the GTX 1660 Ti.
- **CPU and GPU together.** A discrete GPU reads every line, sixteen at a
  time, while the CPU crops the next ones. An integrated GPU shares the
  memory and the power budget with the cores, so it reads 40% of each
  page's line width and the cores read the rest.

## Install

On BigLinux, or Arch Linux with the BigLinux repositories:

```bash
sudo pacman -S ppocr-native
```

The package installs the worker as `/usr/bin/ppocr-native` and the plans and
character list in `/usr/share/ppocr-native`.

## Build from source

You need Rust 1.93 or newer and `glslc` (from shaderc) for the shaders. The
plans are exported from the PP-OCRv6 small ONNX models RapidOCR distributes
(`rapidocr-models-v6-small` on BigLinux), with Python, onnx and numpy:

```bash
cargo build --release --locked --bin ppocr-native

models=/usr/share/rapidocr/models
mkdir -p plans
tools/export_ocr.py $models/PP-OCRv6_det_small.onnx plans/PP-OCRv6_det_small.plan
tools/export_ocr.py $models/PP-OCRv6_rec_small.onnx plans/PP-OCRv6_rec_small_argmax.plan --argmax
tools/export_dictionary.py $models/PP-OCRv6_rec_small.onnx plans/PP-OCRv6_rec_small.characters.txt

PPOCR_NATIVE_PLANS=plans target/release/ppocr-native page.ppm
```

The repository keeps no model files: they are 31 MB under the models'
licence, and exporting them from the installed models keeps them matched to
those weights. `--argmax` makes the recogniser output each time step's best
class and its probability instead of the softmax over all 18 710 classes.
`src/engine/plan.rs` describes the plan format and checks every plan before
anything runs. `pkgbuild/PKGBUILD` does all of the above.

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

bigocrpdf uses the last two for dewarping, with `--detect-only`, which loads
the detector alone. Plans are read from `PPOCR_NATIVE_PLANS`, by default
`/usr/share/ppocr-native`, and checked before the ready line. `--threads N`
sets how many threads the detector shares each page between and how many
lines are read side by side, by default the CPUs the process may use.
`--gpu` runs both networks on a Vulkan GPU. `src/bin/ppocr-native/main.rs`
lists every option.

Memory follows the page: the detector alone peaks at 27 MiB for a 480x448
input, 184 MiB at 1984x1344 and 1087 MiB at the 4096x4096 limit.

## Using the library

Each step of the pipeline is a public function:

| Step | Function |
|---|---|
| Page limits (4096 px, 32-pixel multiples) | `prepare_page`, `fit_page` |
| Detector input size and tensor | `det_size`, `det_input`, `det_input_hwc` |
| Contours, rectangles, scores, unclip, filter, order | `boxes_from_prob` |
| Crop a box upright | `crop_text` |
| Recogniser input | `rec_input` |
| CTC decoding, text and score | `ctc_decode`, `ctc_decode_probs` |
| Lines on the original page | `page_lines` |

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
`engine::Model` runs a plan on the CPU and `engine::GpuModel` on a GPU; both
read HWC tensors, so RapidOCR's CHW ones are transposed on the way in:

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

Not covered: the detector's `fast` score mode (bigocrpdf uses `slow`),
dilation and the text-line classifier. Two kinds of thin page are answered
differently from RapidOCR, to keep memory bounded: `prepare_page` refuses a
page whose padded detector input would exceed 4096² pixels, and the worker
answers it as having no text; with `--full-resolution`, a page whose short
side scaled up to 960 would pass that size is detected at the size used
without the option. Re-reading vertical lines is the worker's, not the
library's.

## Repository layout

- `src/*.rs`: the image and geometry steps of the table above.
- `src/engine/`: the inference engine. `plan.rs` is the plan format,
  `schedule.rs` decides which operations run together and where each tensor
  lives, `mod.rs` runs them, `conv.rs` and `layers.rs` compute whole layers
  with the SIMD kernels in `kernels/`.
- `src/engine/gpu/`, `shaders/`: `GpuModel`, the same plans on a Vulkan GPU.
- `src/bin/ppocr-native/`: the worker.
- `tools/`: plan and dictionary export, parity checks against RapidOCR, the
  instruction check and benchmarks.

## Testing

```bash
cargo test
PPOCR_NATIVE_PLANS=plans cargo test --release -- --include-ignored
```

The first compares every step with fixtures written from OpenCV, pyclipper
and RapidOCR; the second also drives the worker, and checks that a GPU reads
as the CPU does (`--skip gpu_` on a machine without one). Running the
tests at each older tier, the instruction check, the parity tools and the
benchmarks are in [docs/development.md](docs/development.md).

## License

By Bruno Gonçalves ([BigLinux](https://www.biglinux.com.br)), licensed MIT
OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)). [NOTICE](NOTICE) names the projects whose
behaviour this reproduces, and carries the licences of the OpenCV, Clipper
and XNNPACK code that parts of the crate are adapted from. The plans the
package installs are exported from PaddleOCR's models and keep their
Apache-2.0 licence.
