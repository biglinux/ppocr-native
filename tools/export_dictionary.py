#!/usr/bin/env python3
"""Write a PP-OCR recogniser's character list out of its ONNX metadata.

RapidOCR reads the classes from the model's `character` metadata entry and
splits it into lines; `Dictionary::from_metadata` does the same with the text
this writes, byte for byte as stored in the model.

Usage:
    export_dictionary.py MODEL.onnx OUT.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.model), load_external_data=False)
    entries = {p.key: p.value for p in model.metadata_props}
    if "character" not in entries:
        print(f"{args.model}: no 'character' metadata", file=sys.stderr)
        return 1
    args.out.write_bytes(entries["character"].encode("utf-8"))
    print(f"{len(entries['character'].splitlines())} characters -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
