#!/usr/bin/env python3
"""Export a PP-OCRv6 text recognizer or detector (ONNX, opset 11) to a plan.

It pattern-matches the graphs RapidOCR ships as `PP-OCRv6_rec_small.onnx`
and `PP-OCRv6_det_small.onnx` and refuses anything else; a model whose
input height is fixed is the recognizer. The plan serves every input size:
the graph is traced at a recognizer width of 320, RapidOCR's narrowest, or
a 64-by-64 detector page, and the engine derives every other size from each
window's padding and stride, the upsampling factors and `mean`, which
covers the whole map whatever its size.

What happens at export time:

- shapes are evaluated at the traced size, so every Shape/Slice/Concat on
  shapes is constant and the Reshape, Transpose, Squeeze and Identity nodes
  become storage aliases (NHWC storage makes `[1, C, 1, T]` and `[1, T, C]`
  the same bytes); none of those constants reaches the plan;
- weights are transposed once to the engine's NHWC layout;
- Conv + Add(bias), Conv + BatchNormalization and ConvTranspose + Add(bias)
  fold into one convolution, and the activation that follows (ReLU,
  HardSigmoid, Sigmoid, erf GELU, x*sigmoid(x)) becomes its `fused` tail;
- the squeeze-and-excite ReduceMean over height and width, and
  GlobalAveragePool, become `mean`;
- nearest Resize by whole factors becomes `upsample`, and an n-way channel
  Concat a chain of `concatc`;
- the written-out LayerNorm (ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div,
  Mul, Add) becomes `layernorm`;
- the multi-head attention block (Reshape, Transpose, Slice, Squeeze, scale,
  MatMul, Softmax, MatMul, Transpose, Reshape) becomes one `attention` op;
- MatMul by a constant becomes a 1x1 convolution over the T time steps;
- every tensor gets a buffer of its own; the engine decides which share
  memory.

The file layout is documented in `src/engine/plan.rs`, which reads it.
The engine input is HWC `[H][W][3]`: transpose RapidOCR's CHW tensor first.

Usage:
    export_ocr.py MODEL.onnx OUTPUT.plan [--argmax]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"PPOCRPL1"
# Codes in the file: the order of `KINDS` and `FUSED` in src/engine/plan.rs.
KINDS = [
    "conv",
    "dwconv",
    "add",
    "maxpool",
    "avgpool",
    "channelmul",
    "concatc",
    "layernorm",
    "attention",
    "mean",
    "ctchead",
    "ctcargmax",
    "upsample",
    "upsampleadd",
    "convtranspose",
]
FUSED = ["none", "relu", "sigmoid", "gelu", "swish", "hardsigmoid", "hardsigmoidfifth"]
# The integer fields of an operation record between its codes and its blocks.
FIELDS = (
    "x",
    "x2",
    "y",
    "ic",
    "oc",
    "kh",
    "kw",
    "sh",
    "sw",
    "pt",
    "pb",
    "pl",
    "pr",
    "n2",
)
WIDTH = 320
PAGE = 64
HARDSIGMOID = {
    (np.float32(0.16666670143604279), np.float32(0.5)): "hardsigmoid",
    (np.float32(0.2), np.float32(0.5)): "hardsigmoidfifth",
}


class OcrExporter:
    def __init__(self, model: onnx.ModelProto, argmax: bool = False) -> None:
        dims = model.graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_value = 1
        if not dims[2].dim_value:
            dims[2].dim_value = PAGE
        dims[3].dim_value = WIDTH if dims[2].dim_value == 48 else PAGE
        inferred = onnx.shape_inference.infer_shapes(model)
        self.graph = model.graph
        self.const = {t.name: numpy_helper.to_array(t) for t in self.graph.initializer}
        self.shape = {}
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        ):
            dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
            if dims:
                dims[0] = 1  # the batch axis is symbolic in these exports
            self.shape[value.name] = dims
        self.blob = bytearray()
        self.ops: list[dict] = []
        self.alias: dict[str, str] = {}  # tensor -> the tensor it shares storage with
        self.hwc: dict[str, tuple[int, int, int]] = {}
        self.argmax = argmax
        # Drop Identity nodes: rewrite every input through them.
        same: dict[str, str] = {}
        nodes = []
        for node in self.graph.node:
            if node.op_type == "Identity":
                same[node.output[0]] = same.get(node.input[0], node.input[0])
                continue
            nodes.append(node)
        for node in nodes:
            for i, name in enumerate(node.input):
                if name in same:
                    node.input[i] = same[name]
        self.nodes = nodes
        self.consumers: dict[str, list] = {}
        for node in nodes:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)
        self.done: set[int] = set()
        # Concat outputs a convolution reads as two inputs, without a tensor.
        self.pairs: dict[str, tuple[str, str]] = {}

    # Weights and tensors.

    def put(self, array: np.ndarray) -> list[int]:
        """Append an array on a 64-byte boundary; return its [start, length] in floats."""
        while len(self.blob) % 64:
            self.blob.append(0)
        start = len(self.blob) // 4
        data = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(data).all():
            raise SystemExit("refusing to export a weight that is not a finite number")
        self.blob.extend(data.tobytes())
        return [start, data.size]

    def root(self, name: str) -> str:
        while name in self.alias:
            name = self.alias[name]
        return name

    def note(self, name: str, h: int, w: int, c: int) -> None:
        self.hwc[name] = (h, w, c)

    def dims(self, name: str) -> tuple[int, int, int]:
        return self.hwc[name]

    def alias_to(self, out: str, src: str) -> None:
        self.alias[out] = src
        self.hwc[out] = self.hwc[src]

    def attrs(self, node) -> dict:
        return {a.name: (list(a.ints) if len(a.ints) else a.i) for a in node.attribute}

    def emit(
        self, kind: str, x: str, y: str, second: str | None = None, **fields
    ) -> None:
        op = {"kind": kind, "_x": x, "_y": y, **fields}
        if second is not None:
            op["_x2"] = second
        self.ops.append(op)

    # Graph queries.

    def only(self, name: str, op_type: str | None = None):
        """The single consumer of `name`, if it has exactly one (of `op_type`)."""
        users = self.consumers.get(name, [])
        if len(users) != 1 or (op_type and users[0].op_type != op_type):
            return None
        return users[0]

    def scalar(self, name: str) -> float | None:
        value = self.const.get(name)
        if value is None or value.size != 1:
            return None
        return float(value.reshape(-1)[0])

    def use(self, *nodes) -> None:
        for node in nodes:
            self.done.add(id(node))

    def tail(self, out: str) -> tuple[str, str]:
        """Fold the activation after `out`; returns (fused kind, final tensor)."""
        users = self.consumers.get(out, [])
        if len(users) == 1:
            n = users[0]
            if n.op_type == "Relu":
                self.use(n)
                return "relu", n.output[0]
            if n.op_type == "HardSigmoid":
                a = {x.name: np.float32(x.f) for x in n.attribute}
                fused = HARDSIGMOID.get((a.get("alpha"), a.get("beta")))
                if fused is None:
                    raise SystemExit(f"unexpected HardSigmoid constants {a}")
                self.use(n)
                return fused, n.output[0]
            if n.op_type == "Sigmoid":
                self.use(n)
                return "sigmoid", n.output[0]
        if len(users) == 2:
            kinds = sorted(u.op_type for u in users)
            if kinds == ["Div", "Mul"]:  # x/sqrt(2) -> erf -> +1 -> x*() -> *0.5
                div = next(u for u in users if u.op_type == "Div")
                mul = next(u for u in users if u.op_type == "Mul")
                erf = self.only(div.output[0], "Erf")
                plus = erf and self.only(erf.output[0], "Add")
                half = self.only(mul.output[0], "Mul")
                if (
                    abs(self.scalar(div.input[1]) - np.sqrt(2)) < 1e-6
                    and plus is not None
                    and self.scalar(plus.input[1]) == 1.0
                    and plus.output[0] in mul.input
                    and half is not None
                    and self.scalar(half.input[1]) == 0.5
                ):
                    self.use(div, erf, plus, mul, half)
                    return "gelu", half.output[0]
            if kinds == ["Mul", "Sigmoid"]:  # x * sigmoid(x)
                sig = next(u for u in users if u.op_type == "Sigmoid")
                mul = next(u for u in users if u.op_type == "Mul")
                if sig.output[0] in mul.input and out in mul.input:
                    self.use(sig, mul)
                    return "swish", mul.output[0]
        return "none", out

    def emit_op(self, kind: str, x: str, y: str, **fields) -> None:
        fused = fields.pop("fused", "none")
        if fused != "none":
            fields["fused"] = fused
        self.emit(kind, x, y, **fields)

    # Translation.

    def convert(self) -> None:
        source = self.graph.input[0].name
        _, c, h, w = self.shape[source]
        self.note(source, h, w, c)
        for node in self.nodes:
            if id(node) in self.done:
                continue
            handler = getattr(self, f"ocr_{node.op_type.lower()}", None)
            if handler is None:
                raise SystemExit(f"no rule for {node.op_type} {node.name}")
            handler(node)

    def ocr_conv(self, node) -> None:
        weight = self.const[node.input[1]].astype(np.float64)
        out_channels, in_per_group, kh, kw = weight.shape
        a = self.attrs(node)
        group = a.get("group", 1)
        sh, sw = a.get("strides", [1, 1])
        pads = a.get("pads", [0, 0, 0, 0])
        if a.get("dilations", [1, 1]) != [1, 1]:
            raise SystemExit(f"dilated convolution {node.name}")
        auto = next(
            (x.s.decode() for x in node.attribute if x.name == "auto_pad"), "NOTSET"
        )
        if auto == "SAME_UPPER":
            if (sh, sw) != (1, 1):
                raise SystemExit("SAME_UPPER only at stride 1")
            pads = [(kh - 1) // 2, (kw - 1) // 2, kh // 2, kw // 2]
        elif auto not in ("NOTSET", "VALID"):
            raise SystemExit(f"auto_pad {auto}")
        x, second = self.pairs.get(node.input[0], (node.input[0], None))
        ih, iw, ic = self.dims(x)
        n2 = self.dims(second)[2] if second else 0
        bias = np.zeros(out_channels)
        if len(node.input) > 2:
            bias = self.const[node.input[2]].astype(np.float64)
        out = node.output[0]
        nxt = self.only(out)
        if nxt is not None and nxt.op_type == "Add" and nxt.input[1] in self.const:
            bias = bias + self.const[nxt.input[1]].reshape(-1).astype(np.float64)
            self.use(nxt)
            out = nxt.output[0]
        elif nxt is not None and nxt.op_type == "BatchNormalization":
            gamma, beta, mean, var = (
                self.const[n].astype(np.float64) for n in nxt.input[1:5]
            )
            eps = next(x.f for x in nxt.attribute if x.name == "epsilon")
            scale = gamma / np.sqrt(var + np.float64(np.float32(eps)))
            weight = weight * scale[:, None, None, None]
            bias = (bias - mean) * scale + beta
            self.use(nxt)
            out = nxt.output[0]
        fused, out = self.tail(out)
        _, oc, oh, ow = self.shape[node.output[0]]
        self.note(out, oh, ow, oc)
        fields = {}
        depthwise = group > 1 and in_per_group == 1
        if depthwise:
            if not group == ic == out_channels:
                raise SystemExit("grouped convolution that is not depthwise")
            packed = weight.reshape(out_channels, kh, kw).transpose(1, 2, 0)
        else:
            if group != 1:
                raise SystemExit("grouped convolution")
            packed = weight.transpose(2, 3, 1, 0)
        if n2:
            fields["n2"] = n2
        self.emit_op(
            "dwconv" if depthwise else "conv",
            x,
            out,
            second=second,
            kh=kh,
            kw=kw,
            sh=sh,
            sw=sw,
            pt=pads[0],
            pl=pads[1],
            pr=pads[3],
            pb=pads[2],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
            w=self.put(packed.astype(np.float32)),
            b=self.put(bias.astype(np.float32)),
            fused=fused,
            **fields,
        )

    def ocr_maxpool(self, node) -> None:
        a = {x.name: x for x in node.attribute}
        kh, kw = a["kernel_shape"].ints
        sh, sw = a["strides"].ints
        dilated = "dilations" in a and list(a["dilations"].ints) != [1, 1]
        # SAME_UPPER at stride 1 pads (k - 1) // 2 before and k // 2 after;
        # the engine pads only after, which covers kernels up to 2.
        if (
            a["auto_pad"].s.decode() != "SAME_UPPER"
            or (sh, sw) != (1, 1)
            or dilated
            or max(kh, kw) > 2
        ):
            raise SystemExit("only the stem's SAME_UPPER stride-1 2x2 max pool")
        ih, iw, ic = self.dims(node.input[0])
        self.note(node.output[0], ih, iw, ic)
        self.emit(
            "maxpool",
            node.input[0],
            node.output[0],
            kh=kh,
            kw=kw,
            sh=1,
            sw=1,
            pr=kw - 1,
            pb=kh - 1,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=ih,
            ow=iw,
            oc=ic,
        )

    def ocr_concat(self, node) -> None:
        if all(name in self.const for name in node.input):
            self.const[node.output[0]] = np.concatenate(
                [self.const[name].reshape(-1) for name in node.input]
            )
            return
        if self.attrs(node).get("axis") != 1:
            raise SystemExit("only channel concatenation")
        # A dense convolution with a window reads the last input in place.
        conv = self.only(node.output[0], "Conv")
        folds = conv is not None and self.attrs(conv).get("group", 1) == 1
        folds = folds and self.const[conv.input[1]].shape[2:] != (1, 1)
        # Each further input is appended to what the ones before it made.
        head, last = node.input[0], len(node.input) - folds
        for k, name in enumerate(node.input[1:last], 2):
            ih, iw, ic = self.dims(head)
            h2, w2, c2 = self.dims(name)
            if (h2, w2) != (ih, iw):
                raise SystemExit("concatenated maps differ in size")
            out = node.output[0] if k == len(node.input) else f"{node.output[0]}/{k}"
            self.note(out, ih, iw, ic + c2)
            self.emit(
                "concatc",
                head,
                out,
                second=name,
                ih=ih,
                iw=iw,
                ic=ic,
                oh=ih,
                ow=iw,
                oc=ic + c2,
                n2=c2,
            )
            head = out
        if folds:
            (h, w, c), (h2, w2, c2) = self.dims(head), self.dims(node.input[-1])
            if (h2, w2) != (h, w):
                raise SystemExit("concatenated maps differ in size")
            self.note(node.output[0], h, w, c + c2)
            self.pairs[node.output[0]] = (head, node.input[-1])

    def ocr_reducemean(self, node) -> None:
        axes = self.attrs(node).get("axes")
        if axes == [2, 3]:  # squeeze-and-excite
            self.ocr_globalaveragepool(node)
            return
        if axes == [-1]:
            self.layer_norm(node)
            return
        raise SystemExit(f"ReduceMean over {axes}")

    def ocr_globalaveragepool(self, node) -> None:
        ih, iw, ic = self.dims(node.input[0])
        self.note(node.output[0], 1, 1, ic)
        self.emit(
            "mean",
            node.input[0],
            node.output[0],
            ih=ih,
            iw=iw,
            ic=ic,
            oh=1,
            ow=1,
            oc=ic,
        )

    def ocr_resize(self, node) -> None:
        a = {x.name: x.s.decode() for x in node.attribute if x.s}
        want = {
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
        }
        scales = self.const.get(node.input[2]) if len(node.input) > 2 else None
        if a != want or scales is None or len(scales) != 4:
            raise SystemExit(f"unsupported Resize {node.name}")
        fh, fw = (int(f) for f in scales[2:])
        if list(scales) != [1, 1, fh, fw]:
            raise SystemExit("Resize by a fractional factor")
        ih, iw, ic = self.dims(node.input[0])
        out, kind, second = node.output[0], "upsample", None
        add = self.only(out, "Add")
        if add is not None:
            other = add.input[1] if add.input[0] == out else add.input[0]
            if self.dims(other) == (ih * fh, iw * fw, ic):
                self.use(add)
                out, kind, second = add.output[0], "upsampleadd", other
        self.note(out, ih * fh, iw * fw, ic)
        self.emit(
            kind,
            node.input[0],
            out,
            second=second,
            sh=fh,
            sw=fw,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=ih * fh,
            ow=iw * fw,
            oc=ic,
        )

    def ocr_convtranspose(self, node) -> None:
        weight = self.const[node.input[1]]
        ic, oc, kh, kw = weight.shape
        a = self.attrs(node)
        if (
            a.get("strides") != [kh, kw]
            or any(a.get("pads", [0, 0, 0, 0]))
            or a.get("group", 1) != 1
            or a.get("dilations", [1, 1]) != [1, 1]
        ):
            raise SystemExit("only transposed convolutions whose windows tile")
        bias = np.zeros(oc, dtype=np.float32)
        out = node.output[0]
        nxt = self.only(out, "Add")
        if nxt is not None and nxt.input[1] in self.const:
            bias = self.const[nxt.input[1]].reshape(-1)
            self.use(nxt)
            out = nxt.output[0]
        fused, out = self.tail(out)
        ih, iw, _ = self.dims(node.input[0])
        self.note(out, ih * kh, iw * kw, oc)
        self.emit_op(
            "convtranspose",
            node.input[0],
            out,
            kh=kh,
            kw=kw,
            sh=kh,
            sw=kw,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=ih * kh,
            ow=iw * kw,
            oc=oc,
            w=self.put(weight.transpose(0, 2, 3, 1)),
            b=self.put(np.tile(bias, kh * kw)),
            fused=fused,
        )

    def layer_norm(self, mean) -> None:
        x = mean.input[0]
        users = self.consumers[x]
        sub = next(u for u in users if u.op_type == "Sub")
        pw = next(u for u in self.consumers[sub.output[0]] if u.op_type == "Pow")
        div = next(u for u in self.consumers[sub.output[0]] if u.op_type == "Div")
        var = self.only(pw.output[0], "ReduceMean")
        add = self.only(var.output[0], "Add")
        sqrt = self.only(add.output[0], "Sqrt")
        mul = self.only(div.output[0], "Mul")
        beta = self.only(mul.output[0], "Add")
        if self.scalar(pw.input[1]) != 2.0 or div.input[1] != sqrt.output[0]:
            raise SystemExit("not a layer normalization")
        eps = self.const[add.input[1]].reshape(-1)
        gamma = self.const[mul.input[1]].reshape(-1)
        bias = self.const[beta.input[1]].reshape(-1)
        self.use(mean, sub, pw, var, add, sqrt, div, mul, beta)
        h, w, c = self.dims(x)
        self.note(beta.output[0], h, w, c)
        self.emit(
            "layernorm",
            x,
            beta.output[0],
            ih=h,
            iw=w,
            ic=c,
            oh=h,
            ow=w,
            oc=c,
            w=self.put(np.concatenate([gamma, bias, eps]).astype(np.float32)),
        )

    def ocr_mul(self, node) -> None:
        a, b = node.input
        if b in self.const:
            raise SystemExit(f"unexpected constant Mul {node.name}")
        ha, wa, ca = self.dims(a)
        hb, wb, cb = self.dims(b)
        if (hb, wb) == (1, 1) and cb == ca and (ha, wa) != (1, 1):
            self.note(node.output[0], ha, wa, ca)
            self.emit(
                "channelmul",
                a,
                node.output[0],
                second=b,
                ih=ha,
                iw=wa,
                ic=ca,
                oh=ha,
                ow=wa,
                oc=ca,
            )
            return
        raise SystemExit(f"unsupported Mul {node.name}")

    def ocr_add(self, node) -> None:
        a, b = node.input
        if a in self.const or b in self.const:
            raise SystemExit(f"constant Add {node.name} not folded")
        if self.dims(a) != self.dims(b):
            raise SystemExit(f"Add of different shapes {node.name}")
        h, w, c = self.dims(a)
        self.note(node.output[0], h, w, c)
        self.emit(
            "add", a, node.output[0], second=b, ih=h, iw=w, ic=c, oh=h, ow=w, oc=c
        )

    def ocr_averagepool(self, node) -> None:
        a = self.attrs(node)
        auto = next(
            (x.s.decode() for x in node.attribute if x.name == "auto_pad"), "NOTSET"
        )
        if any(a.get("pads", [0, 0, 0, 0])) or auto not in ("NOTSET", "VALID"):
            raise SystemExit("padded average pool")
        if a.get("ceil_mode", 0):
            raise SystemExit("average pool rounding its output up")
        kh, kw = a["kernel_shape"]
        sh, sw = a["strides"]
        ih, iw, ic = self.dims(node.input[0])
        _, oc, oh, ow = self.shape[node.output[0]]
        self.note(node.output[0], oh, ow, oc)
        self.emit(
            "avgpool",
            node.input[0],
            node.output[0],
            kh=kh,
            kw=kw,
            sh=sh,
            sw=sw,
            ih=ih,
            iw=iw,
            ic=ic,
            oh=oh,
            ow=ow,
            oc=oc,
        )

    # Shapes are constant at the export width: evaluate them.
    def ocr_shape(self, node) -> None:
        h, w, c = self.dims(node.input[0])
        self.const[node.output[0]] = np.array([1, c, h, w], dtype=np.int64)

    def ocr_slice(self, node) -> None:
        if node.input[0] not in self.const:
            raise SystemExit("Slice of a runtime tensor outside attention")
        data = self.const[node.input[0]]
        starts, ends = self.const[node.input[1]], self.const[node.input[2]]
        axes = self.const[node.input[3]] if len(node.input) > 3 else [0]
        if list(axes) != [0]:
            raise SystemExit("Slice of a shape on another axis")
        self.const[node.output[0]] = data[int(starts[0]) : int(ends[0])]

    def ocr_squeeze(self, node) -> None:
        if node.input[0] in self.const:
            self.const[node.output[0]] = self.const[node.input[0]].reshape(())
            return
        # [1, C, 1, T] -> [1, C, T]: the same NHWC bytes.
        h, _, _ = self.dims(node.input[0])
        if self.attrs(node).get("axes") != [2] or h != 1:
            raise SystemExit("unsupported Squeeze")
        self.alias_to(node.output[0], node.input[0])

    def ocr_unsqueeze(self, node) -> None:
        self.const[node.output[0]] = self.const[node.input[0]].reshape(1)

    def ocr_reshape(self, node) -> None:
        if node.input[0] in self.const:
            self.const[node.output[0]] = self.const[node.input[0]].reshape(
                self.const[node.input[1]]
            )
            return
        target = list(self.const[node.input[1]])
        if target == [0, -1, 3, 8, 15]:
            self.attention(node)
            return
        h, w, c = self.dims(node.input[0])
        # [1,C,1,T] -> [1,C,T] (flatten) and [1,T,C] -> [1,1,T,C]: storage
        # stays T rows of C channels.
        if h == 1 and target in ([1, c, -1], [1, c, w], [0, 1, -1, c], [0, 1, w, c]):
            self.alias_to(node.output[0], node.input[0])
            return
        raise SystemExit(f"unsupported Reshape to {target}")

    def ocr_transpose(self, node) -> None:
        perm = self.attrs(node)["perm"]
        h, _, _ = self.dims(node.input[0])
        # [1,C,T] -> [1,T,C], [1,1,T,C] -> [1,C,1,T]: same bytes in NHWC.
        if h == 1 and perm in ([0, 2, 1], [0, 3, 1, 2]):
            self.alias_to(node.output[0], node.input[0])
            return
        raise SystemExit(f"unsupported Transpose {perm}")

    def ocr_matmul(self, node) -> None:
        x, wname = node.input
        if wname not in self.const:
            raise SystemExit("MatMul of two runtime tensors outside attention")
        weight = self.const[wname]
        ci, co = weight.shape
        h, t, c = self.dims(x)
        if (h, c) != (1, ci):
            raise SystemExit("MatMul input is not [T][C]")
        out = node.output[0]
        bias = np.zeros(co, dtype=np.float32)
        add = self.only(out, "Add")
        softmax = add and self.only(add.output[0], "Softmax")
        if co > 4096 and softmax is not None:
            self.ctc_head(x, weight, self.const[add.input[1]].reshape(-1), add, softmax)
            return
        nxt = self.only(out, "Add")
        if nxt is not None and nxt.input[1] in self.const:
            bias = self.const[nxt.input[1]].reshape(-1)
            self.use(nxt)
            out = nxt.output[0]
        fused, out = self.tail(out)
        self.note(out, 1, t, co)
        self.emit_op(
            "conv",
            x,
            out,
            kh=1,
            kw=1,
            sh=1,
            sw=1,
            ih=1,
            iw=t,
            ic=ci,
            oh=1,
            ow=t,
            oc=co,
            w=self.put(weight),
            b=self.put(bias),
            fused=fused,
        )

    def ctc_head(self, x, weight, bias, add, softmax) -> None:
        """MatMul + Add + Softmax over the vocabulary: one panel-packed head."""
        ci, vocab = weight.shape
        columns = -(-vocab // 16) * 16
        padded = np.zeros((ci, columns), dtype=np.float32)
        padded[:, :vocab] = weight
        panels = padded.reshape(ci, columns // 16, 16).transpose(1, 0, 2)
        b = np.full(columns, -1e30, dtype=np.float32)
        b[:vocab] = bias
        _, t, _ = self.dims(x)
        out = softmax.output[0]
        self.use(add, softmax)
        if self.argmax:
            self.note(out, 1, t, 2)
            self.emit(
                "ctcargmax",
                x,
                out,
                ih=1,
                iw=t,
                ic=ci,
                oh=1,
                ow=t,
                oc=2,
                n2=vocab,
                w=self.put(panels),
                b=self.put(b),
            )
        else:
            self.note(out, 1, t, vocab)
            self.emit(
                "ctchead",
                x,
                out,
                ih=1,
                iw=t,
                ic=ci,
                oh=1,
                ow=t,
                oc=vocab,
                w=self.put(panels),
                b=self.put(b),
            )

    def attention(self, reshape) -> None:
        """Reshape(qkv) ... Reshape([0,-1,120]): one multi-head attention."""
        qkv = reshape.input[0]
        _, t, c3 = self.dims(qkv)
        seen, frontier, end, scale = [reshape], [reshape], None, None
        while frontier:
            node = frontier.pop()
            for user in self.consumers.get(node.output[0], []):
                if user in seen:
                    continue
                seen.append(user)
                if user.op_type == "Reshape":
                    end = user
                    continue
                if user.op_type == "Mul":
                    scale = self.scalar(user.input[1])
                if user.op_type == "Add" and self.scalar(user.input[1]) not in (
                    0.0,
                    None,
                ):
                    raise SystemExit("attention bias is not zero")
                frontier.append(user)
        kinds = sorted(n.op_type for n in seen)
        expect = sorted(
            [
                "Reshape",
                "Transpose",
                "Slice",
                "Squeeze",
                "Slice",
                "Squeeze",
                "Slice",
                "Squeeze",
                "Mul",
                "Add",
                "Transpose",
                "MatMul",
                "Softmax",
                "MatMul",
                "Transpose",
                "Reshape",
            ]
        )
        if kinds != expect or end is None or scale is None:
            raise SystemExit(f"not the expected attention block: {kinds}")
        self.use(*seen)
        c = c3 // 3
        self.note(end.output[0], 1, t, c)
        self.emit(
            "attention",
            qkv,
            end.output[0],
            ih=1,
            iw=t,
            ic=c3,
            oh=1,
            ow=t,
            oc=c,
            n2=8,
            w=self.put(np.array([scale], dtype=np.float32)),
        )

    # Output.

    def assign_buffers(self) -> tuple[dict[str, int], int]:
        """Give every tensor a buffer of its own, aliases their root's.

        The engine works out from the order it runs the operations in which
        buffers can share memory, and lays them out.
        """
        for op in self.ops:
            for key in ("_x", "_x2", "_y"):
                if key in op:
                    op[key] = self.root(op[key])
        slot = {self.root(self.graph.input[0].name): 0}
        for op in self.ops:
            slot.setdefault(op["_y"], len(slot))
        return slot, len(slot)

    def write(self, path: Path) -> int:
        """Convert, assign buffers and write the plan; returns the buffer count."""
        self.convert()
        slot, buffers = self.assign_buffers()
        for op in self.ops:
            op["x"] = slot[op.pop("_x")]
            op["y"] = slot[op.pop("_y")]
            if "_x2" in op:
                op["x2"] = slot[op.pop("_x2")]
        name = self.graph.input[0].name
        h, w, c = self.dims(name)
        words = [
            slot[self.root(name)],
            h,
            w,
            c,
            buffers,
            len(self.graph.output),
            len(self.ops),
        ]
        words += [slot[self.root(o.name)] for o in self.graph.output]
        for op in self.ops:
            words += [KINDS.index(op["kind"]), FUSED.index(op.get("fused", "none"))]
            words += [op.get(field, 0) for field in FIELDS]
            words += [*op.get("w", [0, 0]), *op.get("b", [0, 0])]
        if min(words) < 0 or max(words) >= 1 << 32:
            raise SystemExit("a field does not fit the plan's 32-bit words")
        head = MAGIC + struct.pack(f"<{len(words)}I", *words)
        head += b"\0" * (-len(head) % 64)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(head + bytes(self.blob))
        return buffers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument(
        "--argmax",
        action="store_true",
        help="output each step's best class and probability instead of all probabilities",
    )
    args = parser.parse_args()
    exporter = OcrExporter(onnx.load(args.model), args.argmax)
    buffers = exporter.write(args.plan)
    kinds: dict[str, int] = {}
    for op in exporter.ops:
        key = op["kind"] + ("+" + op["fused"] if "fused" in op else "")
        kinds[key] = kinds.get(key, 0) + 1
    print(f"{args.plan}: {len(exporter.ops)} operations {kinds}")
    print(f"weights {len(exporter.blob) / 1e6:.2f} MB, {buffers} buffers")


if __name__ == "__main__":
    main()
