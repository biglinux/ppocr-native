#!/usr/bin/env python3
"""Record every intermediate of RapidOCR, page by page: the oracle the crate is checked against.

RapidOCR runs on ONNX Runtime with the settings the ppocr-native worker
defaults to. It is instrumented by wrapping its objects in this process only,
so what is recorded is what the library computed, not a reimplementation of
it. A development tool: it needs RapidOCR, ONNX Runtime, OpenCV, pyclipper,
shapely and the PP-OCRv6 small ONNX models, none of which the crate needs.

Arrays are written as .npy files, which a test can read with a few lines of
std Rust; page.json says what each file holds and carries every box, score and
text. Full recogniser probabilities are 18 710 classes per timestep, so they
are kept only for the first --full-probs pages; every line keeps the argmax,
the max probability and the top five.

Usage:
    dump_reference.py MANIFEST IMAGE_DIR OUT_DIR [--pages 30] [--full-probs 2]
        [--models /usr/share/rapidocr/models]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

# RapidOCR imports OpenVINO when it is installed, and OpenVINO reports usage
# to Google Analytics on import unless this module is missing.
sys.modules.setdefault("openvino_telemetry", None)

import cv2
import numpy as np
import onnxruntime
import pyclipper
import rapidocr
import rapidocr.main
import rapidocr.utils.process_img
import shapely
from omegaconf import OmegaConf
from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.ch_ppocr_det import TextDetector

THREADS = 4
TOP_K = 5
# The worker's defaults: src/det.rs (PageOptions, DbOptions) and the text
# score in src/bin/ppocr-native/main.rs.
BOX_THRESH = 0.5
UNCLIP_RATIO = 1.5
TEXT_SCORE = 0.3
MAX_SIDE_LEN = 4096


class Recorder:
    """Collects one page's intermediates while RapidOCR runs."""

    def __init__(self) -> None:
        self.page_dir: Path | None = None
        self.full_probs = False
        self.reset()

    def reset(self) -> None:
        self.json: dict = {
            "det": {"contours": [], "filter": [], "boxes": []},
            "crops": [],
            "rec": [],
        }
        self.contour_ids: dict[int, int] = {}
        self.current_contour: dict | None = None
        self.crop_ids: dict[int, int] = {}
        self.pending_rec: dict | None = None

    def save(self, name: str, array: np.ndarray) -> str:
        assert self.page_dir is not None
        path = self.page_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ascontiguousarray(array))
        return name


rec = Recorder()


def box_list(box: np.ndarray) -> list:
    return np.asarray(box, dtype=np.float64).reshape(-1, 2).tolist()


class SessionProxy:
    """Stands in for an inference session and records what crosses it."""

    def __init__(self, session, on_run) -> None:
        self._session = session
        self._on_run = on_run

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        output = self._session(tensor)
        self._on_run(tensor, output)
        return output

    def __getattr__(self, name):
        return getattr(self._session, name)


class DecoderProxy:
    def __init__(self, decoder) -> None:
        self._decoder = decoder

    def __call__(self, preds, *args, **kwargs):
        lines, words = self._decoder(preds, *args, **kwargs)
        assert rec.pending_rec is not None and len(lines) == 1
        rec.pending_rec["text"], rec.pending_rec["score"] = lines[0]
        return lines, words

    def __getattr__(self, name):
        return getattr(self._decoder, name)


def instrument(engine) -> None:
    preprocess_img = engine.preprocess_img

    def recorded_preprocess(image):
        out, op_record = preprocess_img(image)
        entry = {
            "input_hw": list(image.shape[:2]),
            "output_hw": list(out.shape[:2]),
            "ratio_h": op_record["preprocess"]["ratio_h"],
            "ratio_w": op_record["preprocess"]["ratio_w"],
        }
        if out.shape != image.shape:
            entry["file"] = rec.save("preprocessed.npy", out)
        rec.json["preprocess"] = entry
        return out, op_record

    engine.preprocess_img = recorded_preprocess

    det = engine.text_det
    get_preprocess = det.get_preprocess

    def recorded_get_preprocess(max_wh):
        op = get_preprocess(max_wh)
        rec.json["det"].update(
            max_wh=int(max_wh),
            limit_side_len=op.limit_side_len,
            limit_type=op.limit_type,
            mean=op.mean.tolist(),
            std=op.std.tolist(),
        )
        resize = op.resize

        def recorded_resize(image):
            out = resize(image)
            # Boxes are scaled back to the image the detector was given.
            rec.json["det"]["dest_hw"] = list(image.shape[:2])
            rec.json["det"]["resized_hw"] = list(out.shape[:2])
            rec.json["det"]["resized_file"] = rec.save("det_resized.npy", out)
            return out

        op.resize = recorded_resize
        return op

    det.get_preprocess = recorded_get_preprocess

    def on_det(tensor, preds):
        post = det.postprocess_op
        rec.json["det"].update(
            input_file=rec.save("det_input.npy", tensor),
            prob_file=rec.save("det_prob.npy", preds[0, 0]),
            bitmap_file=rec.save(
                "det_bitmap.npy", (preds[0, 0] > post.thresh).astype(np.uint8)
            ),
            thresh=post.thresh,
            box_thresh=post.box_thresh,
            unclip_ratio=post.unclip_ratio,
            score_mode=post.score_mode,
            max_candidates=post.max_candidates,
            min_size=post.min_size,
            dilation=post.dilation_kernel is not None,
        )

    det.session = SessionProxy(det.session, on_det)

    post = det.postprocess_op
    get_mini_boxes, box_score_slow, unclip = (
        post.get_mini_boxes,
        post.box_score_slow,
        post.unclip,
    )
    filter_det_res = post.filter_det_res

    def recorded_mini(contour):
        box, sside = get_mini_boxes(contour)
        index = rec.contour_ids.get(id(contour))
        if index is not None:
            rec.current_contour = {
                "index": index,
                "mini_box": box_list(box),
                "mini_sside": float(sside),
                "status": "small" if sside < post.min_size else "pending",
            }
            rec.json["det"]["contours"].append(rec.current_contour)
        else:
            entry = rec.current_contour
            entry["unclip_mini_box"] = box_list(box)
            entry["unclip_mini_sside"] = float(sside)
            if sside < post.min_size + 2:
                entry["status"] = "small_after_unclip"
            else:
                entry["status"] = "kept"
        return box, sside

    def recorded_score(bitmap, contour):
        score = box_score_slow(bitmap, contour)
        entry = rec.current_contour
        entry["score"] = float(score)
        if post.box_thresh > score:
            entry["status"] = "low_score"
        return score

    def recorded_unclip(box):
        expanded = unclip(box)
        entry = rec.current_contour
        entry["unclip_distance"] = float(
            shapely.Polygon(box).area * post.unclip_ratio / shapely.Polygon(box).length
        )
        entry["unclipped"] = np.asarray(expanded).reshape(-1, 2).tolist()
        return expanded

    def recorded_filter(boxes, scores, img_height, img_width):
        out_boxes, out_scores = filter_det_res(boxes, scores, img_height, img_width)
        # boxes_from_bitmap returns one scaled box per kept contour, in order.
        kept_contours = [
            c for c in rec.json["det"]["contours"] if c["status"] == "kept"
        ]
        assert len(kept_contours) == len(boxes)
        for contour, box in zip(kept_contours, boxes):
            contour["scaled_box"] = np.asarray(box).tolist()
        kept = 0
        for box, score in zip(boxes, scores):
            ordered = post.order_points_clockwise(box)
            clipped = post.clip_det_res(ordered.copy(), img_height, img_width)
            rect_w = int(np.linalg.norm(clipped[0] - clipped[1]))
            rect_h = int(np.linalg.norm(clipped[0] - clipped[3]))
            keep = rect_w > 3 and rect_h > 3
            if keep:
                assert np.array_equal(clipped, out_boxes[kept])
                kept += 1
            rec.json["det"]["filter"].append(
                {
                    "input_box": box_list(box),
                    "score": float(score),
                    "ordered": box_list(ordered),
                    "clipped": box_list(clipped),
                    "kept": keep,
                }
            )
        assert kept == len(out_boxes)
        rec.json["det"]["scores_as_returned"] = [float(s) for s in out_scores]
        return out_boxes, out_scores

    post.get_mini_boxes = recorded_mini
    post.box_score_slow = recorded_score
    post.unclip = recorded_unclip
    post.filter_det_res = recorded_filter

    rec_engine = engine.text_rec
    resize_norm_img = rec_engine.resize_norm_img

    def recorded_resize_norm(image, max_wh_ratio):
        out = resize_norm_img(image, max_wh_ratio)
        crop = rec.crop_ids.get(id(image))
        rec.pending_rec = {
            "crop": crop,
            "image_hw": list(image.shape[:2]),
            "max_wh_ratio": float(max_wh_ratio),
            "input_width": int(out.shape[2]),
        }
        return out

    rec_engine.resize_norm_img = recorded_resize_norm

    def on_rec(tensor, preds):
        entry = rec.pending_rec
        prefix = f"rec/{entry['crop']:04d}"
        probs = preds[0]
        top = np.argsort(-probs, axis=1, kind="stable")[:, :TOP_K]
        entry.update(
            order=len(rec.json["rec"]),
            input_file=rec.save(f"{prefix}_input.npy", tensor),
            timesteps=int(probs.shape[0]),
            classes=int(probs.shape[1]),
            argmax_file=rec.save(
                f"{prefix}_argmax.npy", probs.argmax(axis=1).astype(np.int32)
            ),
            maxprob_file=rec.save(
                f"{prefix}_maxprob.npy", probs.max(axis=1).astype(np.float32)
            ),
            topk_index_file=rec.save(f"{prefix}_topk_index.npy", top.astype(np.int32)),
            topk_prob_file=rec.save(
                f"{prefix}_topk_prob.npy", np.take_along_axis(probs, top, axis=1)
            ),
        )
        if rec.full_probs:
            entry["probs_file"] = rec.save(
                f"{prefix}_probs_f16.npy", probs.astype(np.float16)
            )
        rec.json["rec"].append(entry)

    rec_engine.session = SessionProxy(rec_engine.session, on_rec)
    rec_engine.postprocess_op = DecoderProxy(rec_engine.postprocess_op)

    for module in (rapidocr.main, rapidocr.utils.process_img):
        module.get_rotate_crop_image = recorded_crop(module.get_rotate_crop_image)


def recorded_crop(get_rotate_crop_image):
    def crop(image, points):
        points_in = np.array(points, dtype=np.float64)
        out = get_rotate_crop_image(image, points)
        warp_w = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        warp_h = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        entry = {
            "points": points_in.tolist(),
            "warp_hw": [warp_h, warp_w],
            "rotated": bool(out.shape[:2] != (warp_h, warp_w)),
            "hw": list(out.shape[:2]),
        }
        index = len(rec.json["crops"])
        entry["index"] = index
        entry["file"] = rec.save(f"rec/{index:04d}_crop.npy", out)
        rec.json["crops"].append(entry)
        rec.crop_ids[id(out)] = index
        return out

    return crop


def install_contour_capture() -> None:
    find_contours = cv2.findContours

    def recorded(image, mode, method, *args, **kwargs):
        outs = find_contours(image, mode, method, *args, **kwargs)
        contours = outs[-2]
        # RETR_LIST order is the order RapidOCR scores contours in.
        points = [c.reshape(-1, 2) for c in contours]
        rec.json["det"]["contours_file"] = rec.save(
            "det_contours.npy",
            np.concatenate([np.empty((0, 2), np.int32), *points]).astype(np.int32),
        )
        rec.json["det"]["contour_lengths"] = [len(c) for c in points]
        rec.contour_ids = {id(c): i for i, c in enumerate(contours)}
        rec.json["det"]["bitmap_hw"] = list(image.shape[:2])
        rec.json["det"]["contour_mode"] = [int(mode), int(method)]
        return outs

    cv2.findContours = recorded


def install_sort_capture() -> None:
    sorted_boxes = TextDetector.sorted_boxes

    def recorded(dt_boxes):
        out = sorted_boxes(dt_boxes)
        used: set[int] = set()
        order = []
        for box in out:
            index = next(
                i
                for i, b in enumerate(dt_boxes)
                if i not in used and np.array_equal(b, box)
            )
            used.add(index)
            order.append(index)
        rec.json["det"]["sort_order"] = order
        rec.json["det"]["boxes"] = [box_list(b) for b in out]
        return out

    TextDetector.sorted_boxes = staticmethod(recorded)


def build_engine(models: Path) -> RapidOCR:
    """RapidOCR on ONNX Runtime, one image at a time, as the worker reads a page."""
    return RapidOCR(
        params={
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.model_path": str(models / "PP-OCRv6_det_small.onnx"),
            "Det.box_thresh": BOX_THRESH,
            "Det.unclip_ratio": UNCLIP_RATIO,
            "Det.use_dilation": False,
            # PaddleOCR's normalization; RapidOCR ships 0.5 for both.
            "Det.mean": [0.485, 0.456, 0.406],
            "Det.std": [0.229, 0.224, 0.225],
            "Det.score_mode": "slow",
            "Det.limit_type": "max",
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.model_path": str(models / "PP-OCRv6_rec_small.onnx"),
            "Rec.lang_type": LangRec.LATIN,
            "Rec.rec_batch_num": 1,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Global.use_cls": False,
            "Global.text_score": TEXT_SCORE,
            "Global.max_side_len": MAX_SIDE_LEN,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.SMALL,
            "Rec.model_type": ModelType.SMALL,
            "EngineConfig.onnxruntime.intra_op_num_threads": THREADS,
            "EngineConfig.onnxruntime.inter_op_num_threads": 2,
        }
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_page(engine, image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    rec.json.update(
        image=str(image_path),
        image_sha256=sha256(image_path),
        image_hw=list(image.shape[:2]),
    )
    # Every stage is named: RapidOCR keeps whatever the previous call passed.
    result = engine(
        image,
        use_det=True,
        use_rec=True,
        use_cls=False,
        text_score=TEXT_SCORE,
        box_thresh=BOX_THRESH,
    )
    # RapidOCR reports a page without text as None in all three fields.
    rec.json["engine_output"] = {
        "boxes": [box_list(b) for b in ([] if result.boxes is None else result.boxes)],
        "txts": list(result.txts or ()),
        "scores": [float(s) for s in result.scores or ()],
    }
    # The worker's lines before it reads vertical ones again, which the
    # crate leaves to its caller.
    rec.json["worker_before_vertical"] = (
        {"boxes": None}
        if result.boxes is None
        else {
            "boxes": [box_list(b) for b in result.boxes],
            "txts": list(result.txts or ()),
            "scores": [float(s) for s in result.scores or ()],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--pages", type=int, default=30)
    parser.add_argument("--full-probs", type=int, default=2)
    parser.add_argument(
        "--models", type=Path, default=Path("/usr/share/rapidocr/models")
    )
    args = parser.parse_args()

    engine = build_engine(args.models)
    install_contour_capture()
    install_sort_capture()
    instrument(engine)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
    ]
    page_ids = [row["id"] for row in rows[: args.pages]]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    det_model = args.models / "PP-OCRv6_det_small.onnx"
    rec_model = args.models / "PP-OCRv6_rec_small.onnx"
    characters = engine.text_rec.postprocess_op.character
    index = {
        "pages": page_ids,
        "full_probs_pages": page_ids[: args.full_probs],
        "options": {
            "box_thresh": BOX_THRESH,
            "unclip_ratio": UNCLIP_RATIO,
            "text_score": TEXT_SCORE,
            "limit_side_len": MAX_SIDE_LEN,
        },
        "rapidocr_config": OmegaConf.to_container(engine.cfg, resolve=True),
        "models": {
            "det": {"path": str(det_model), "sha256": sha256(det_model)},
            "rec": {"path": str(rec_model), "sha256": sha256(rec_model)},
        },
        "versions": {
            "rapidocr": importlib.metadata.version("rapidocr"),
            "opencv": cv2.__version__,
            "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__,
            "pyclipper": pyclipper.__version__,
            "shapely": shapely.__version__,
        },
        "threads": THREADS,
    }
    (args.out_dir / "characters.json").write_text(
        json.dumps(characters, ensure_ascii=False), encoding="utf-8"
    )
    (args.out_dir / "index.json").write_text(
        json.dumps(index, indent=1, default=str), encoding="utf-8"
    )

    for number, page_id in enumerate(page_ids):
        rec.reset()
        rec.page_dir = args.out_dir / page_id
        rec.full_probs = number < args.full_probs
        run_page(engine, args.image_dir / f"{page_id}.ppm")
        (rec.page_dir / "page.json").write_text(
            json.dumps(rec.json, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(
            f"{page_id}: {len(rec.json['crops'])} crops, "
            f"{len(rec.json['worker_before_vertical'].get('txts') or [])} lines",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
