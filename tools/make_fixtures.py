#!/usr/bin/env python3
"""Write the small fixtures behind `cargo test`, from RapidOCR and OpenCV themselves.

Every expected value here is computed by the library the crate reproduces:
cv2 for resizing, contours, rectangles and crops, pyclipper for growing boxes,
RapidOCR's own classes for the detector input, the DB postprocess, the
recogniser input and the decoding, and RapidOCR itself, on the PP-OCRv6
small ONNX models with the worker's settings, for the worker's answers. Real
data comes from the reference set `dump_reference.py` records; synthetic
inputs use a fixed seed.

Usage:
    make_fixtures.py REFERENCE_DIR DICTIONARY_TXT OUT_DIR
        [--models /usr/share/rapidocr/models]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

# RapidOCR imports OpenVINO when it is installed, and OpenVINO reports usage
# to Google Analytics on import unless this module is missing.
sys.modules.setdefault("openvino_telemetry", None)

import cv2
import numpy as np
import pyclipper
from dump_reference import BOX_THRESH, TEXT_SCORE, box_list, build_engine
from rapidocr.ch_ppocr_det import TextDetector
from rapidocr.ch_ppocr_det.utils import DBPostProcess, DetPreProcess
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.utils.process_img import get_rotate_crop_image
from shapely.geometry import Polygon

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def contours_of(mask: np.ndarray) -> list:
    found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).tolist() for c in found]


def synthetic_masks(rng: np.random.Generator) -> list[np.ndarray]:
    ring = np.zeros((12, 14), np.uint8)
    ring[1:11, 1:13] = 1
    ring[3:8, 4:10] = 0
    ring[5, 6] = 1  # an island inside the hole
    edges = np.zeros((9, 9), np.uint8)
    edges[0, :] = 1  # touches the top edge
    edges[:, 8] = 1
    edges[4, 4] = 1  # a lone pixel
    edges[6:9, 0:3] = 1
    diamond = np.ones((9, 9), np.uint8)
    diamond[0, :] = diamond[-1, :] = diamond[:, 0] = diamond[:, -1] = 0
    diamond[4, 3:6] = 0
    diamond[3:6, 4] = 0
    noise = (rng.random((40, 60)) < 0.45).astype(np.uint8)
    return [ring, edges, diamond, noise]


def worker_reading(engine, image: np.ndarray) -> dict:
    """RapidOCR's lines of a page, as the worker writes them."""
    result = engine(
        image,
        use_det=True,
        use_rec=True,
        use_cls=False,
        text_score=TEXT_SCORE,
        box_thresh=BOX_THRESH,
    )
    return {
        "boxes": [box_list(b) for b in result.boxes],
        "txts": list(result.txts),
        "scores": [float(s) for s in result.scores],
    }


def half_turn_reading(engine, image: np.ndarray, box: list) -> tuple[str, float]:
    """A line cropped, turned half a turn and read alone, as the worker's vertical pass does."""
    crop = get_rotate_crop_image(image, np.array(box, np.float32))
    result = engine(
        cv2.rotate(crop, cv2.ROTATE_180), use_det=False, use_cls=False, use_rec=True
    )
    return result.txts[0], float(result.scores[0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("dictionary", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--models", type=Path, default=Path("/usr/share/rapidocr/models")
    )
    args = parser.parse_args()
    ref, out = args.reference, args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261011)
    fixtures: dict = {}

    # Bilinear resize, including an exact halving, and the detector tensor.
    image = rng.integers(0, 256, (28, 40, 3), dtype=np.uint8)
    np.save(out / "noise.npy", image)
    fixtures["resize"] = []
    for w, h in [(17, 13), (20, 14), (100, 64), (40, 57), (7, 3)]:
        name = f"resize_{w}x{h}.npy"
        np.save(out / name, cv2.resize(image, (w, h)))
        fixtures["resize"].append({"width": w, "height": h, "file": name})
    tensor = DetPreProcess(960, "max", MEAN, STD)(image)
    np.save(out / "det_input.npy", tensor)
    fixtures["det_input"] = {"file": "det_input.npy", "mean": MEAN, "std": STD}

    # Contours of synthetic masks and of part of a real detector bitmap.
    page = ref / "dharmaocr_test_0000"
    bitmap = np.ascontiguousarray(np.load(page / "det_bitmap.npy")[140:400, 100:420])
    masks = [*synthetic_masks(rng), bitmap]
    fixtures["contours"] = []
    for i, mask in enumerate(masks):
        name = f"mask_{i}.npy"
        np.save(out / name, mask)
        fixtures["contours"].append({"file": name, "contours": contours_of(mask * 255)})

    # The DB postprocess on part of a real probability map, mapped to a
    # slightly larger page as RapidOCR does when sizes snap to 32.
    prob = np.ascontiguousarray(np.load(page / "det_prob.npy")[140:460, 100:420])
    np.save(out / "prob.npy", prob)
    dest_h, dest_w = prob.shape[0] + 3, prob.shape[1] + 5
    post = DBPostProcess(
        thresh=0.3,
        box_thresh=0.5,
        max_candidates=1000,
        unclip_ratio=1.5,
        score_mode="slow",
        use_dilation=False,
    )
    boxes, _ = post(prob[None, None], (dest_h, dest_w))
    fixtures["boxes"] = {
        "file": "prob.npy",
        "dest": [dest_w, dest_h],
        "boxes": TextDetector.sorted_boxes(boxes).tolist(),
    }

    # Minimum-area rectangles and grown boxes on real contours.
    rects, grown = [], []
    for page_dir in sorted(ref.glob("dharmaocr_test_*"))[:6]:
        p = json.loads((page_dir / "page.json").read_text(encoding="utf-8"))
        flat = np.load(page_dir / "det_contours.npy")
        bounds = np.cumsum([0] + p["det"]["contour_lengths"])
        for c in p["det"]["contours"]:
            for pts in (
                flat[bounds[c["index"]] : bounds[c["index"] + 1]],
                np.array(c.get("unclipped", []), np.int32),
            ):
                if len(pts):
                    rect = cv2.minAreaRect(pts)
                    rects.append(
                        {"points": pts.tolist(), "box": cv2.boxPoints(rect).tolist()}
                    )
            if "unclipped" in c:
                quad = np.array(c["mini_box"], np.float32)
                poly = Polygon(quad)
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(quad, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                grown.append(
                    {
                        "quad": quad.tolist(),
                        "ring": offset.Execute(poly.area * 1.5 / poly.length)[0],
                    }
                )
    fixtures["min_area_rect"] = rects
    fixtures["unclip"] = grown[:150]

    # Crops from part of a real page: straight, tilted with fractional
    # corners, and tall enough to be turned.
    crop_page = np.ascontiguousarray(
        np.load(page / "det_resized.npy")[140:260, 110:330]
    )
    np.save(out / "page.npy", crop_page)
    quads = [
        [[6, 9], [200, 9], [200, 31], [6, 31]],
        [[10.5, 40.25], [190.75, 44.5], [189.5, 70.0], [9.25, 66.75]],
        [[30, 20], [52, 20], [52, 110], [30, 110]],
        [[-3, 100], [230, 98], [231, 125], [-2, 127]],
    ]
    fixtures["crops"] = []
    for i, q in enumerate(quads):
        name = f"crop_{i}.npy"
        np.save(out / name, get_rotate_crop_image(crop_page, np.array(q, np.float32)))
        fixtures["crops"].append({"quad": q, "file": name})

    # Recogniser input for a narrow crop (padded) and a wide one (not).
    owner = SimpleNamespace(rec_image_shape=[3, 48, 320])
    fixtures["rec_input"] = []
    for i, (h, w) in enumerate([(20, 60), (10, 71)]):
        crop = np.ascontiguousarray(crop_page[:h, :w])
        ratio = max(320 / 48, w / h)
        np.save(out / f"rec_crop_{i}.npy", crop)
        np.save(
            out / f"rec_input_{i}.npy",
            TextRecognizer.resize_norm_img(owner, crop, ratio)[None],
        )
        fixtures["rec_input"].append(
            {"crop": f"rec_crop_{i}.npy", "input": f"rec_input_{i}.npy"}
        )

    # Decoding: an empty line, a short one, and three whose score changes
    # if the mean is not summed in numpy's pairwise order (the last one long
    # enough for numpy to split the sum).
    fixtures["ctc"] = []
    lines = [(3, 42), (0, 41), (11, 9), (24, 19), (20, 43)]
    for page_id, crop in [(f"dharmaocr_test_{n:04d}", c) for n, c in lines]:
        p = json.loads((ref / page_id / "page.json").read_text(encoding="utf-8"))
        e = next(e for e in p["rec"] if e["crop"] == crop)
        stem = f"ctc_{page_id[-4:]}_{crop:04d}"
        shutil.copy(ref / page_id / e["argmax_file"], out / f"{stem}_argmax.npy")
        shutil.copy(ref / page_id / e["maxprob_file"], out / f"{stem}_maxprob.npy")
        fixtures["ctc"].append({"stem": stem, "text": e["text"], "score": e["score"]})
    shutil.copy(args.dictionary, out / "characters.txt")

    # Final assembly on a page that was shrunk and lost two lines.
    p = json.loads(
        (ref / "dharmaocr_test_0001" / "page.json").read_text(encoding="utf-8")
    )
    by_crop = {e["crop"]: e for e in p["rec"]}
    fixtures["page_lines"] = {
        "image_hw": p["image_hw"],
        "ratio_h": p["preprocess"]["ratio_h"],
        "ratio_w": p["preprocess"]["ratio_w"],
        "boxes": p["det"]["boxes"],
        "reads": [
            [by_crop[i]["text"], by_crop[i]["score"]]
            for i in range(len(p["det"]["boxes"]))
        ],
        "expected": p["worker_before_vertical"],
    }

    # The worker's answers: the crop page as read, and a caption set bottom
    # to top, which reads upside down until the vertical pass turns it.
    engine = build_engine(args.models)
    fixtures["worker"] = {"page": worker_reading(engine, crop_page)}
    line = crop_page[9:31, 6:200]
    caption = np.full((line.shape[1] + 40, line.shape[0] + 40, 3), 255, np.uint8)
    caption[20:-20, 20:-20] = cv2.rotate(line, cv2.ROTATE_90_COUNTERCLOCKWISE)
    np.save(out / "caption.npy", caption)
    first = worker_reading(engine, caption)
    if len(first["boxes"]) != 1:
        raise SystemExit(f"caption: expected one line, got {first}")
    text, score = half_turn_reading(engine, caption, first["boxes"][0])
    if not (first["scores"][0] < 0.9 and score >= first["scores"][0] + 0.05):
        raise SystemExit(f"caption: {first} would not be read again as {text!r}")
    fixtures["worker"]["caption"] = {"file": "caption.npy", "text": text}

    (out / "fixtures.json").write_text(
        json.dumps(fixtures, ensure_ascii=False), encoding="utf-8"
    )
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"{len(list(out.iterdir()))} files, {total / 1e6:.2f} MB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
