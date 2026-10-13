#!/usr/bin/env python3
"""Compare a PP-OCRv6 plan with ONNX Runtime on RapidOCR's own input tensors.

A directory holding `lines.jsonl` is recogniser input: every line runs at
its own width in both engines, without padding, and the report gives the
largest probability gap (of the best class, for an argmax plan), the
per-timestep argmax agreement and CTC greedy text equality. One holding
`pages.jsonl` is detector input: every page runs at its own size, and the
report gives the largest probability-map gap and the share of pixels whose
binarization at RapidOCR's threshold 0.3 differs. The engine runs at
whatever tier `PPOCR_NATIVE_TIER` allows.

The gates: a recogniser reads every line to the same text, agrees on at
least 99.9% of time steps and stays within 5e-3 in probability; a detector
stays within 1e-3 in probability, with at most 1e-4 of any page's pixels
binarized differently.

Usage:
    check_ocr_parity.py MODEL.onnx PLAN INPUT_DIR [--native BIN] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

# RapidOCR's DBPostProcess threshold on the probability map.
DET_THRESHOLD = 0.3
ROOT = Path(__file__).resolve().parent.parent
# The kind code of `ctcargmax` and the bytes of an operation record, as
# src/engine/plan.rs reads them.
CTCARGMAX, OP_BYTES = 11, 80


def argmax_plan(path: Path) -> bool:
    """Whether the plan's last operation is the argmax CTC head."""
    with path.open("rb") as plan:
        head = plan.read(36)
        outputs, ops = (
            int.from_bytes(head[28:32], "little"),
            int.from_bytes(head[32:36], "little"),
        )
        plan.seek(36 + 4 * outputs + OP_BYTES * (ops - 1))
        return int.from_bytes(plan.read(4), "little") == CTCARGMAX


def decode(ids: np.ndarray, chars: list[str]) -> str:
    out, last = [], 0
    for i in ids:
        if i != last and i != 0:
            out.append(chars[i])
        last = i
    return "".join(out)


def read_output(stream, count: int, name: str) -> np.ndarray | None:
    data = stream.read(count * 4)
    if len(data) != count * 4:
        print(f"{name}: the engine stopped early", file=sys.stderr)
        return None
    values = np.frombuffer(data, dtype=np.float32)
    if not np.isfinite(values).all():
        print(f"{name}: non-finite output", file=sys.stderr)
        return None
    return values


def check_lines(args, session, rows, native) -> bool:
    argmax_only = argmax_plan(args.plan)
    meta = {p.key: p.value for p in onnx.load(args.model).metadata_props}
    chars = ["blank", *meta["character"].splitlines(), " "]
    worst, agree, steps, same_text, examples = 0.0, 0, 0, 0, []
    for row in rows:
        path = args.inputs / f"line_{row['index']:05d}.f32"
        x = np.fromfile(path, dtype=np.float32).reshape(1, 3, 48, -1)
        want = session.run(None, {"x": x})[0][0]
        t, vocab = want.shape
        width = 2 if argmax_only else vocab
        got = read_output(native.stdout, t * width, path.name)
        if got is None:
            return False
        got = got.reshape(t, width)
        a = want.argmax(1)
        if argmax_only:
            b = got[:, 0].astype(np.int64)
            gap = np.abs(got[:, 1] - want.max(1)).max()
        else:
            b = got.argmax(1)
            gap = np.abs(got - want).max()
        worst = max(worst, float(gap))
        agree += int((a == b).sum())
        steps += t
        ta, tb = decode(a, chars), decode(b, chars)
        same_text += ta == tb
        if ta != tb and len(examples) < 5:
            examples.append((path.name, ta, tb))
    n = len(rows)
    widths = [r["width"] for r in rows]
    print(f"  {n} lines, widths {min(widths)}..{max(widths)}")
    what = "best-class p" if argmax_only else "p"
    print(f"  max |{what} native - ort|   {worst:.3e}")
    print(f"  argmax agreement            {agree}/{steps} = {100 * agree / steps:.4f}%")
    print(f"  identical decoded text      {same_text}/{n} = {100 * same_text / n:.2f}%")
    for name, ta, tb in examples:
        print(f"  differs {name}: ort {ta!r}\n                native {tb!r}")
    ok = worst <= 5e-3 and agree / steps >= 0.999 and same_text == n
    print(
        "  gate (gap <= 5e-3, argmax >= 99.9%, every text):",
        "PASS" if ok else "FAIL",
    )
    return ok


def check_pages(args, session, rows, native) -> bool:
    worst, worst_share, flipped, pixels = 0.0, 0.0, 0, 0
    for row in rows:
        path = args.inputs / f"page_{row['index']:03d}.f32"
        x = np.fromfile(path, dtype=np.float32).reshape(1, 3, row["h"], row["w"])
        # A side off the 32-pixel grid comes back rounded up to it.
        want = session.run(None, {"x": x})[0][0, 0]
        h, w = want.shape
        got = read_output(native.stdout, h * w, path.name)
        if got is None:
            return False
        got = got.reshape(h, w)
        gap = float(np.abs(got - want).max())
        differ = int(((got > DET_THRESHOLD) != (want > DET_THRESHOLD)).sum())
        worst = max(worst, gap)
        worst_share = max(worst_share, differ / (h * w))
        flipped += differ
        pixels += h * w
        print(f"  {row['page']} {h}x{w}: max gap {gap:.3e}, {differ} pixels flip")
    sides = [max(r["h"], r["w"]) for r in rows]
    print(f"  {len(rows)} pages, long side {min(sides)}..{max(sides)}, {pixels} pixels")
    print(f"  max |p native - ort|       {worst:.3e}")
    print(f"  binarization differs       {flipped} pixels = {flipped / pixels:.3e}")
    print(f"  worst page                 {worst_share:.3e}")
    ok = worst <= 1e-3 and worst_share <= 1e-4
    print("  gate (gap <= 1e-3, every page <= 1e-4):", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument(
        "--native", type=Path, default=ROOT / "target/release/examples/plan"
    )
    parser.add_argument("--limit", type=int, help="only the first N inputs")
    args = parser.parse_args()

    pages = (args.inputs / "pages.jsonl").exists()
    listing = args.inputs / ("pages.jsonl" if pages else "lines.jsonl")
    with listing.open(encoding="utf-8") as rows_file:
        rows = [json.loads(line) for line in rows_file][: args.limit]
    if pages:
        files = [
            f"{args.inputs}/page_{r['index']:03d}.f32:{r['h']}x{r['w']}" for r in rows
        ]
    else:
        files = [f"{args.inputs}/line_{r['index']:05d}.f32" for r in rows]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.model), options, providers=["CPUExecutionProvider"]
    )
    tier = os.environ.get("PPOCR_NATIVE_TIER", "detected")
    print(f"{args.plan.name}, tier cap {tier}:")
    command = [str(args.native), str(args.plan), "run", "-", *files]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as native:
        ok = (check_pages if pages else check_lines)(args, session, rows, native)
        if not ok:
            native.kill()
        status = native.wait(timeout=60)
    if ok and status != 0:
        print("the engine failed", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
