#!/usr/bin/env python3
"""Compare every stage of the crate with the RapidOCR reference, page by page.

`dump_reference.py` records what RapidOCR computed; `examples/parity_dump.rs`
reruns each stage on the reference's own input to that stage and writes what
the crate computed. This reads both and counts, per stage, how much differs.

The gates are the ones the crate is held to:

- page fitting, detector resize and detector tensor: identical;
- boxes, chained from the probability map: same count per page, and IoU of
  at least 0.99 for at least 99 % of them;
- crops: at least 99 % of pixels within 2 levels;
- recogniser tensors: largest difference at most 1e-3;
- decoded text and score, and the final lines: identical.

Every page of the reference must have been dumped. The final lines are the
worker's before it re-reads vertical lines, which the crate leaves to its
caller.

With --e2e, the recogniser runs on ONNX Runtime over the crate's own crops of
its own boxes, and the final lines are compared with those same worker lines.

Usage:
    cargo run --release --example parity_dump -- REF IMAGES DICT OUT
    check_parity.py REF OUT [--e2e MODEL.onnx]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# RapidOCR imports OpenVINO when it is installed, and OpenVINO reports usage
# to Google Analytics on import unless this module is missing.
sys.modules.setdefault("openvino_telemetry", None)

import numpy as np
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from shapely.geometry import Polygon


def iou(a, b) -> float:
    pa, pb = Polygon(a), Polygon(b)
    union = pa.union(pb).area
    return pa.intersection(pb).area / union if union else float(a == b)


def compare_page(ref: Path, out: Path, t: Counter, decode, e2e) -> None:
    p = json.loads((ref / "page.json").read_text(encoding="utf-8"))
    r = json.loads((out / "rust.json").read_text(encoding="utf-8"))
    det, rdet = p["det"], r["det"]
    t["pages"] += 1

    pre, rpre = p["preprocess"], r["preprocess"]
    t["preprocess: size or ratio differs"] += (
        pre["output_hw"] != rpre["output_hw"]
        or pre["ratio_h"] != rpre["ratio_h"]
        or pre["ratio_w"] != rpre["ratio_w"]
    )
    if "file" in pre:
        t["preprocess: shrunk pages"] += 1
        t["preprocess: shrunk page differs"] += not np.array_equal(
            np.load(ref / pre["file"]), np.load(out / "preprocessed.npy")
        )

    t["det: resized page differs"] += not np.array_equal(
        np.load(ref / det["resized_file"]), np.load(out / "det_resized.npy")
    )
    a, b = np.load(ref / det["input_file"]), np.load(out / "det_input.npy")
    t["det: tensor differs"] += a.shape != b.shape or not np.array_equal(a, b)
    t["det: bitmap differs"] += not np.array_equal(
        np.load(ref / det["bitmap_file"]), np.load(out / "det_bitmap.npy")
    )

    # Contours from the recorded bitmap, in OpenCV's order and point form.
    flat = np.load(ref / det["contours_file"])
    bounds = np.cumsum([0] + det["contour_lengths"])
    ref_contours = [
        flat[bounds[i] : bounds[i + 1]].tolist() for i in range(len(bounds) - 1)
    ]
    own = [c["points"] for c in rdet["contours"]]
    t["contours"] += len(ref_contours)
    t["contours: count differs (pages)"] += len(own) != len(ref_contours)
    t["contours: points differ"] += sum(
        x != y for x, y in zip(ref_contours, own)
    ) + abs(len(own) - len(ref_contours))

    # Each box step on the reference's own input to it.
    for c, mine in zip(det["contours"], rdet["isolated"]):
        t["minAreaRect: box differs"] += (
            np.float32(c["mini_box"]).tolist() != np.float32(mine["mini_box"]).tolist()
        )
        t["minAreaRect: box max |diff|"] = max(
            t["minAreaRect: box max |diff|"],
            float(
                np.abs(np.float64(c["mini_box"]) - np.float64(mine["mini_box"])).max()
            ),
        )
        t["minAreaRect: short side differs"] += c["mini_sside"] != mine["mini_sside"]
        status = mine["candidate"]["status"]
        t["candidate: status differs"] += status != c["status"]
        if "score" in c:
            t["score: max |diff|"] = max(
                t["score: max |diff|"], abs(c["score"] - mine["score"])
            )
        if "unclipped" in c:
            t["unclip"] += 1
            t["unclip: polygon differs"] += c["unclipped"] != mine["unclipped"]
            t["unclip: box differs"] += (
                np.float32(c["unclip_mini_box"]).tolist()
                != np.float32(mine["unclip_mini_box"]).tolist()
            )
            t["unclip: short side differs"] += (
                c["unclip_mini_sside"] != mine["unclip_mini_sside"]
            )
        if c["status"] == "kept" and status == "kept":
            t["scaled box differs (chained per contour)"] += (
                c["scaled_box"] != mine["candidate"]["box"]
            )

    for f, mine in zip(det["filter"], rdet["filtered"]):
        expect = f["clipped"] if f["kept"] else None
        t["filter: differs"] += expect != mine
    t["sort: differs (pages)"] += det["boxes"] != rdet["sorted"]

    ref_boxes, chain = det["boxes"], rdet["chain_boxes"]
    t["boxes"] += len(ref_boxes)
    t["boxes: count differs (pages)"] += len(chain) != len(ref_boxes)
    t["boxes: identical"] += sum(x == y for x, y in zip(ref_boxes, chain))
    t["boxes: IoU >= 0.99"] += sum(iou(x, y) >= 0.99 for x, y in zip(ref_boxes, chain))

    for c in p["crops"]:
        a = np.load(ref / c["file"]).astype(np.int16)
        b = np.load(out / c["file"]).astype(np.int16)
        t["crops"] += 1
        if a.shape != b.shape:
            t["crops: shape differs"] += 1
            continue
        d = np.abs(a - b)
        t["crop pixels"] += d.size
        t["crop pixels: identical"] += int((d == 0).sum())
        t["crop pixels: within 2"] += int((d <= 2).sum())
        t["crop pixels: max |diff|"] = max(t["crop pixels: max |diff|"], int(d.max()))
        t["crops: identical"] += not d.any()

    by_crop = {e["crop"]: e for e in r["rec"]}
    for e in p["rec"]:
        mine = by_crop[e["crop"]]
        a = np.load(ref / e["input_file"])
        b = np.load(out / f"rec/{e['crop']:04d}_input.npy")
        t["rec tensors"] += 1
        if a.shape != b.shape:
            t["rec tensors: shape differs"] += 1
            continue
        t["rec tensors: max |diff|"] = max(
            t["rec tensors: max |diff|"], float(np.abs(a - b).max())
        )
        t["rec tensors: identical"] += np.array_equal(a, b)
        t["ctc: text differs"] += e["text"] != mine["text"]
        t["ctc: score differs"] += e["score"] != mine["score"]
        if "probs_text" in mine:
            # The recorded probabilities are float16, so the score is checked
            # against RapidOCR's decoder on those same values.
            text, score = decode(
                np.load(ref / e["probs_file"]).astype(np.float32)[None]
            )[0][0]
            t["ctc from full probs"] += 1
            t["ctc from full probs: text differs"] += text != mine["probs_text"]
            t["ctc from full probs: score differs"] += score != mine["probs_score"]

    # The worker writes a page without lines as {"boxes": None}.
    if p["worker_before_vertical"]["boxes"] is None:
        p["worker_before_vertical"] = {"boxes": [], "txts": [], "scores": []}
    t["final lines: differ (pages)"] += r["output"] != p["worker_before_vertical"]
    if e2e is not None:
        e2e(ref, out, p, chain, t)


def end_to_end(model: Path, decode, verbose: bool):
    """Recognise the crate's chained crops with ONNX Runtime and assemble lines."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name

    def run(ref: Path, out: Path, p: dict, chain: list, t: Counter) -> None:
        reads = []
        for i in range(len(chain)):
            probs = session.run(None, {name: np.load(out / f"rec/c{i:04d}_input.npy")})[
                0
            ]
            reads.append(decode(probs)[0][0])
        ry, rx = p["preprocess"]["ratio_h"], p["preprocess"]["ratio_w"]
        height, width = p["image_hw"]
        lines = []
        for quad, (text, score) in zip(chain, reads):
            if not text.strip() or score < 0.3:
                continue
            b = np.array(quad, dtype=np.float32)
            b[:, 0] *= rx
            b[:, 1] *= ry
            b = np.where(b < 0, 0, b)
            b[:, 0] = np.minimum(b[:, 0], width)
            b[:, 1] = np.minimum(b[:, 1], height)
            lines.append((b.tolist(), text, score))
        w = p["worker_before_vertical"]
        expect = list(zip(w["boxes"], w["txts"], w["scores"]))
        t["e2e: lines"] += len(expect)
        t["e2e: pages identical"] += lines == expect
        t["e2e: lines identical"] += sum(x == y for x, y in zip(lines, expect))
        t["e2e: line count differs (pages)"] += len(lines) != len(expect)
        t["e2e: boxes identical"] += sum(x[0] == y[0] for x, y in zip(lines, expect))
        t["e2e: texts identical"] += sum(x[1] == y[1] for x, y in zip(lines, expect))
        for x, y in zip(lines, expect):
            if x[1] != y[1] and verbose:
                print(
                    f"{ref.name}: {y[1]!r} -> {x[1]!r}, box {'same' if x[0] == y[0] else 'differs'}"
                )
        t["e2e: score max |diff|"] = max(
            [t["e2e: score max |diff|"]]
            + [abs(x[2] - y[2]) for x, y in zip(lines, expect)]
        )

    return run


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--e2e",
        type=Path,
        metavar="MODEL.onnx",
        help="recogniser to run on the chained crops",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print each end-to-end text that differs",
    )
    args = parser.parse_args()

    index = json.loads((args.reference / "index.json").read_text(encoding="utf-8"))
    characters = json.loads(
        (args.reference / "characters.json").read_text(encoding="utf-8")
    )
    # RapidOCR's own decoder, which adds the blank and the space itself.
    decode = CTCLabelDecode(character=characters[1:-1])
    e2e = end_to_end(args.e2e, decode, args.verbose) if args.e2e else None
    t = Counter()
    for page_id in index["pages"]:
        if (args.out / page_id / "rust.json").exists():
            compare_page(args.reference / page_id, args.out / page_id, t, decode, e2e)
        else:
            t["pages: not dumped"] += 1
    for key, value in t.items():
        print(f"{key}: {value}")

    failures = []
    if t["pages: not dumped"]:
        failures.append("pages not dumped")
    if t["preprocess: size or ratio differs"] or t["preprocess: shrunk page differs"]:
        failures.append("page fitting")
    if (
        t["det: resized page differs"]
        or t["det: tensor differs"]
        or t["det: bitmap differs"]
    ):
        failures.append("detector input")
    if t["boxes: count differs (pages)"] or t["boxes: IoU >= 0.99"] < 0.99 * t["boxes"]:
        failures.append("boxes")
    if (
        t["crops: shape differs"]
        or t["crop pixels: within 2"] < 0.99 * t["crop pixels"]
    ):
        failures.append("crops")
    if t["rec tensors: shape differs"] or t["rec tensors: max |diff|"] > 1e-3:
        failures.append("recogniser input")
    if any(
        t[k]
        for k in (
            "ctc: text differs",
            "ctc: score differs",
            "ctc from full probs: text differs",
            "ctc from full probs: score differs",
        )
    ):
        failures.append("decoding")
    if t["final lines: differ (pages)"]:
        failures.append("final lines")
    print("FAILED: " + ", ".join(failures) if failures else "all gates passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
