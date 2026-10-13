#!/usr/bin/env python3
"""Check that each engine kernel stays inside its tier's instruction set.

Every function of `ppocr_native::engine` in the disassembly is classed by
its name: `*_avx2` may use AVX2 and FMA, `*_avx` only AVX (no FMA, no
256-bit integer or other AVX2-only instructions), `*_sse` only legacy SSE
(no VEX encoding, no ymm), and every other function, the scalar tier and the
dispatch around it, no VEX encoding at all. AVX-512 registers fail every
class. Closures are classed with the function that contains them. This
reads named symbols only: it is not a call-graph or CPU-emulation proof.

Usage:
    check_isa.py BINARY
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ENGINE = re.compile(r"\bppocr_native::engine::")
# AVX2-only instructions that do not start with "vp" and fit an AVX name.
AVX2_ONLY = re.compile(
    r"^(vgather|vpgather|vinserti128|vextracti128|vperm2i128|vpermps|vpermd)"
)


def tier(symbol: str) -> str:
    for name in ("avx2", "avx", "sse"):
        if re.search(rf"_{name}(?![0-9A-Za-z_])", symbol):
            return name
    return "scalar"


def problems(kind: str, mnemonic: str, operands: str) -> str | None:
    if re.search(r"%zmm|%k[0-7]\b", operands):
        return "AVX-512"
    if kind == "avx2":
        return None
    vex = mnemonic.startswith("v")
    if kind in ("sse", "scalar") and (vex or "%ymm" in operands):
        return "VEX or ymm below AVX"
    if kind == "avx":
        if mnemonic.startswith(("vfmadd", "vfmsub", "vfnmadd", "vfnmsub")):
            return "FMA in an AVX kernel"
        if AVX2_ONLY.match(mnemonic):
            return "AVX2 instruction in an AVX kernel"
        if mnemonic.startswith("vbroadcasts") and operands.startswith("%xmm"):
            return "register broadcast (AVX2) in an AVX kernel"
        if (
            mnemonic.startswith("vp")
            and "%ymm" in operands
            and not mnemonic.startswith(("vpermil", "vperm2f128", "vptest"))
        ):
            return "256-bit integer instruction (AVX2) in an AVX kernel"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    text = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-C", str(args.binary)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    rows, current = [], None
    for line in text.splitlines():
        header = re.match(r"^([0-9a-f]+) <(.+)>:$", line)
        if header:
            current = None
            if ENGINE.search(header[2]):
                current = {"symbol": header[2], "tier": tier(header[2])}
                current["violations"] = []
                rows.append(current)
            continue
        insn = re.match(r"\s*[0-9a-f]+:\s+(\S+)\s*(.*)", line)
        if current is None or not insn:
            continue
        found = problems(current["tier"], *insn.groups())
        if found:
            current["violations"].append(f"{found}: {line.strip()}")
    counts = {
        name: sum(r["tier"] == name for r in rows)
        for name in ("avx2", "avx", "sse", "scalar")
    }
    bad = [r for r in rows if r["violations"]]
    result = {
        "status": "PASS"
        if all(counts[t] for t in ("avx2", "avx", "sse")) and not bad
        else "FAIL",
        "functions": counts,
        "violations": bad,
    }
    print(json.dumps(result))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
