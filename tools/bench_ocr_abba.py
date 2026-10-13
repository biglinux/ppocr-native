#!/usr/bin/env python3
"""Interleaved single-CPU comparison of ppocr-native builds on one input.

Each block runs every build once in order and once in reverse (ABBA for two
builds), each run a fresh process pinned to one CPU that times RUNS
inferences after its own warmup. Reports, per build, the median over blocks
of each run's median CPU ms, the spread, and the load average while it ran.

A build is the `plan` example and a plan (`--native NAME=BIN:PLAN`, given
once per build). INPUT is RapidOCR's CHW float32 tensor: a line `FILE`, 48
pixels high, or a page `FILE:HxW`.

Usage:
    bench_ocr_abba.py --cpu 4 --blocks 6 --runs 10 --input LINE.f32 \\
        --native old=old/plan:plans/PP-OCRv6_rec_small_argmax.plan \\
        --native new=target/release/examples/plan:plans/PP-OCRv6_rec_small_argmax.plan \\
        --output RESULT.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def command(build: tuple[str, str], args) -> list[str]:
    binary, plan = build
    return [
        "taskset",
        "-c",
        str(args.cpu),
        binary,
        plan,
        "bench",
        args.input,
        str(args.runs),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--native", action="append", required=True, help="NAME=BIN:PLAN"
    )
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    builds: dict[str, tuple[str, str]] = {}
    for item in args.native:
        name, _, rest = item.partition("=")
        binary, _, plan = rest.partition(":")
        if not (name and binary and plan):
            parser.error(f"--native {item}: expected NAME=BIN:PLAN")
        builds[name] = (binary, plan)
    names = list(builds)
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU outside the affinity mask")
    results: dict[str, list[dict]] = {name: [] for name in names}
    loads = []
    for block in range(args.blocks):
        for name in names + names[::-1]:
            run = subprocess.run(
                command(builds[name], args),
                capture_output=True,
                text=True,
                timeout=1800,
                check=True,
            )
            row = json.loads(
                next(line for line in run.stdout.splitlines() if line.startswith("{"))
            )
            results[name].append(row)
            loads.append(os.getloadavg()[0])
        print(
            f"block {block + 1}/{args.blocks} done, load {loads[-1]:.1f}",
            file=sys.stderr,
        )
    summary = {}
    for name, rows in results.items():
        medians = [r["median_ms"] for r in rows]
        summary[name] = {
            "median_ms": statistics.median(medians),
            "best_ms": min(r["best_ms"] for r in rows),
            "min_median_ms": min(medians),
            "max_median_ms": max(medians),
            "load_ms": statistics.median(r["load_ms"] for r in rows),
            "peak_rss_mib": max(r.get("peak_rss_mib", 0) for r in rows),
        }
    base = summary[names[0]]["median_ms"]
    for name in names:
        summary[name]["vs_first"] = summary[name]["median_ms"] / base
    out = {
        "cpu": args.cpu,
        "input": args.input,
        "blocks": args.blocks,
        "runs": args.runs,
        "load_avg_min": min(loads),
        "load_avg_max": max(loads),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": summary,
        "raw": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps({"load": [min(loads), max(loads)], **summary}, indent=1))


if __name__ == "__main__":
    main()
