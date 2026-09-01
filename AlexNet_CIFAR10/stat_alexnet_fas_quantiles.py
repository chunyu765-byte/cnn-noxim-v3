#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


INT16_MIN = -32768
INT16_MAX = 32767
EPS = 1e-12


@dataclass
class LayerSpec:
    layer_type: str
    name: str
    out_h: int = 0
    out_w: int = 0
    out_c: int = 0
    k_h: int = 0
    k_w: int = 0
    in_c: int = 0
    stride: int = 1
    pad: int = 0
    act: str = "none"
    weight_scale: float = 1.0
    in_scale: float = 1.0
    out_scale: float = 1.0
    out_features: int = 0


def parse_model(path: Path) -> Tuple[float, List[LayerSpec]]:
    input_scale = None
    layers: List[LayerSpec] = []
    conv_idx = 0
    pool_idx = 0
    dense_idx = 0

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            kind = toks[0]
            if kind == "Input":
                input_scale = float(toks[4])
            elif kind == "Convolution":
                conv_idx += 1
                layers.append(
                    LayerSpec(
                        layer_type="Convolution",
                        name=f"conv{conv_idx}",
                        out_h=int(toks[1]),
                        out_w=int(toks[2]),
                        out_c=int(toks[3]),
                        k_h=int(toks[4]),
                        k_w=int(toks[5]),
                        in_c=int(toks[6]),
                        stride=int(toks[7]),
                        pad=int(toks[8]),
                        act=toks[9],
                        weight_scale=float(toks[10]),
                        in_scale=float(toks[13]),
                        out_scale=float(toks[15]),
                    )
                )
            elif kind == "Pooling":
                pool_idx += 1
                layers.append(
                    LayerSpec(
                        layer_type="Pooling",
                        name=f"pool{pool_idx}",
                        out_h=int(toks[1]),
                        out_w=int(toks[2]),
                        out_c=int(toks[3]),
                        k_h=int(toks[4]),
                        k_w=int(toks[5]),
                        stride=int(toks[6]),
                        act=toks[7],
                    )
                )
            elif kind == "Dense":
                dense_idx += 1
                layers.append(
                    LayerSpec(
                        layer_type="Dense",
                        name=f"fc{dense_idx}",
                        out_features=int(toks[1]),
                        act=toks[2],
                        weight_scale=float(toks[3]),
                        in_scale=float(toks[6]),
                        out_scale=float(toks[8]),
                    )
                )
            else:
                raise ValueError(f"Unsupported model line: {line}")

    if input_scale is None:
        raise RuntimeError(f"Input scale not found in {path}")
    return input_scale, layers


def read_input(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append([int(x) for x in line.split()])
    arr = np.asarray(rows, dtype=np.int32)
    if arr.shape != (96, 32):
        raise RuntimeError(f"Expected input txt shape 96x32, got {arr.shape}")
    return arr.reshape(3, 32, 32)


def read_weight_lines(path: Path) -> List[List[int]]:
    lines: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                lines.append([int(x) for x in line.split()])
    return lines


def quantize_int16(x: np.ndarray, scale: float) -> np.ndarray:
    q = np.rint(x / max(scale, EPS))
    q = np.clip(q, INT16_MIN, INT16_MAX)
    return q.astype(np.int32)


def relu_if_needed(x: np.ndarray, act: str) -> np.ndarray:
    if act.lower() == "relu":
        return np.maximum(x, 0)
    return x


def conv2d_float(x_q: np.ndarray, in_scale: float, spec: LayerSpec, w_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    x = x_q.astype(np.float64) * in_scale
    w = w_q.astype(np.float64) * spec.weight_scale
    b = b_q.astype(np.float64) * spec.weight_scale
    padded = np.pad(x, ((0, 0), (spec.pad, spec.pad), (spec.pad, spec.pad)), mode="constant")
    out = np.zeros((spec.out_c, spec.out_h, spec.out_w), dtype=np.float64)

    for oc in range(spec.out_c):
        out_oc = out[oc]
        for oh in range(spec.out_h):
            ih = oh * spec.stride
            for ow in range(spec.out_w):
                iw = ow * spec.stride
                region = padded[:, ih : ih + spec.k_h, iw : iw + spec.k_w]
                out_oc[oh, ow] = float(np.sum(region * w[oc]) + b[oc])
    return relu_if_needed(out, spec.act)


def pool2d_max_q(x_q: np.ndarray, spec: LayerSpec) -> np.ndarray:
    out = np.zeros((spec.out_c, spec.out_h, spec.out_w), dtype=np.int32)
    for c in range(spec.out_c):
        for oh in range(spec.out_h):
            ih = oh * spec.stride
            for ow in range(spec.out_w):
                iw = ow * spec.stride
                out[c, oh, ow] = int(np.max(x_q[c, ih : ih + spec.k_h, iw : iw + spec.k_w]))
    return out


def dense_float(x_q: np.ndarray, in_scale: float, spec: LayerSpec, w_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    x = x_q.reshape(-1).astype(np.float64) * in_scale
    w = w_q.astype(np.float64) * spec.weight_scale
    b = b_q.astype(np.float64) * spec.weight_scale
    out = w.dot(x) + b
    return relu_if_needed(out, spec.act)


def consume_weights(layers: Sequence[LayerSpec], weight_lines: Sequence[Sequence[int]]):
    cursor = 0
    weights: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for spec in layers:
        if spec.layer_type == "Convolution":
            kernels = []
            for _oc in range(spec.out_c):
                ic_kernels = []
                for _ic in range(spec.in_c):
                    vals = weight_lines[cursor]
                    cursor += 1
                    if len(vals) != spec.k_h * spec.k_w:
                        raise RuntimeError(f"{spec.name}: expected {spec.k_h * spec.k_w} kernel values, got {len(vals)}")
                    ic_kernels.append(np.asarray(vals, dtype=np.int32).reshape(spec.k_h, spec.k_w))
                kernels.append(ic_kernels)
            bias = np.asarray(weight_lines[cursor], dtype=np.int32)
            cursor += 1
            if bias.size != spec.out_c:
                raise RuntimeError(f"{spec.name}: expected {spec.out_c} bias values, got {bias.size}")
            weights[spec.name] = (np.asarray(kernels, dtype=np.int32), bias)
        elif spec.layer_type == "Dense":
            bias = np.asarray(weight_lines[cursor], dtype=np.int32)
            cursor += 1
            if bias.size != spec.out_features:
                raise RuntimeError(f"{spec.name}: expected {spec.out_features} bias values, got {bias.size}")
            rows = []
            for _oc in range(spec.out_features):
                rows.append(np.asarray(weight_lines[cursor], dtype=np.int32))
                cursor += 1
            weights[spec.name] = (np.asarray(rows, dtype=np.int32), bias)
    if cursor != len(weight_lines):
        raise RuntimeError(f"Unused weight lines: consumed {cursor}, total {len(weight_lines)}")
    return weights


def forward_quantized(input_q: np.ndarray, input_scale: float, layers: Sequence[LayerSpec], weights) -> List[np.ndarray]:
    activations: List[np.ndarray] = []
    x_q = input_q.astype(np.int32)
    current_scale = input_scale

    for spec in layers:
        if spec.layer_type == "Convolution":
            w_q, b_q = weights[spec.name]
            y = conv2d_float(x_q, current_scale, spec, w_q, b_q)
            x_q = quantize_int16(y, spec.out_scale)
            current_scale = spec.out_scale
        elif spec.layer_type == "Pooling":
            x_q = pool2d_max_q(x_q, spec)
        elif spec.layer_type == "Dense":
            w_q, b_q = weights[spec.name]
            y = dense_float(x_q, current_scale, spec, w_q, b_q)
            x_q = quantize_int16(y, spec.out_scale)
            current_scale = spec.out_scale
        else:
            raise ValueError(spec.layer_type)
        activations.append(x_q.copy())
    return activations


def percentile_thresholds(arr: np.ndarray, percentiles: Sequence[float]) -> List[int]:
    flat = arr.reshape(-1).astype(np.int64)
    nonzero = flat[flat != 0]
    if nonzero.size == 0:
        return [0 for _ in percentiles]
    vals = np.abs(nonzero) if np.any(nonzero < 0) else nonzero
    return [int(round(float(np.percentile(vals, p)))) for p in percentiles]


def write_approx_file(path: Path, layers: Sequence[LayerSpec], thresholds: Sequence[Sequence[int]]) -> None:
    lines = []
    for spec, th in zip(layers, thresholds):
        if len(th) != 4:
            raise RuntimeError(f"{spec.name}: expected four thresholds")
        # cnn-noxim reverses the four values while loading alexnet_approx.txt:
        # file order is threshold3 threshold2 threshold1 threshold0.
        # Write q80 q60 q40 q20 so internal approx_th[0] is q20.
        lines.append(f"{spec.layer_type} {th[3]} {th[2]} {th[1]} {th[0]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, layers: Sequence[LayerSpec], activations: Sequence[np.ndarray], thresholds: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "layer_id",
            "layer_type",
            "layer_name",
            "num_values",
            "num_zero",
            "zero_ratio",
            "num_nonzero",
            "q20_nonzero",
            "q40_nonzero",
            "q60_nonzero",
            "q80_nonzero",
            "removed_by_q20",
            "removed_ratio_q20",
            "expected_zero_plus_nonzero20",
        ])
        for idx, (spec, arr, th) in enumerate(zip(layers, activations, thresholds), start=1):
            flat = arr.reshape(-1).astype(np.int64)
            zero = int(np.count_nonzero(flat == 0))
            nonzero = flat[flat != 0]
            if nonzero.size == 0:
                removed = zero
            else:
                vals = np.abs(nonzero) if np.any(nonzero < 0) else nonzero
                removed = zero + int(np.count_nonzero(vals <= th[0]))
            total = int(flat.size)
            expected = (float(zero) + float(total - zero) * 0.2) / float(total) if total else 0.0
            w.writerow([
                idx,
                spec.layer_type,
                spec.name,
                total,
                zero,
                f"{zero / total:.8f}",
                int(nonzero.size),
                th[0],
                th[1],
                th[2],
                th[3],
                removed,
                f"{removed / total:.8f}",
                f"{expected:.8f}",
            ])


def write_summary(path: Path, image: Path, approx_out: Path, csv_out: Path, layers: Sequence[LayerSpec], thresholds: Sequence[Sequence[int]]) -> None:
    rows = [
        "| Layer | Type | Name | q20 | q40 | q60 | q80 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for idx, (spec, th) in enumerate(zip(layers, thresholds), start=1):
        rows.append(f"| {idx} | {spec.layer_type} | {spec.name} | {th[0]} | {th[1]} | {th[2]} | {th[3]} |")
    text = "\n".join([
        "# AlexNet-CIFAR10 FAS Quantile Thresholds",
        "",
        f"- Image: `{image}`",
        f"- Updated approx file: `{approx_out}`",
        f"- CSV: `{csv_out}`",
        "- Quantiles: non-zero activation values at 20%, 40%, 60%, 80%.",
        "- cnn-noxim loads `alexnet_approx.txt` in reverse threshold order, so the file is written as q80 q60 q40 q20.",
        "- For layers with negative values, non-zero absolute values are used; this only affects the final logits layer, which is not transmitted further in this AlexNet run.",
        "",
        *rows,
        "",
    ])
    path.write_text(text, encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Compute simple FAS-NoC quantile thresholds for AlexNet-CIFAR10 noxim export.")
    parser.add_argument("--image", type=Path, default=script_dir / "idx_01824_label_truck.png")
    parser.add_argument("--model", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_model.txt")
    parser.add_argument("--input", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_input.txt")
    parser.add_argument("--weights", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_weight_fc_wb2parser.txt")
    parser.add_argument("--approx-out", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_approx.txt")
    parser.add_argument("--csv-out", type=Path, default=script_dir / "analysis_csv" / "alexnet_fas_nonzero_quantiles.csv")
    parser.add_argument("--summary-out", type=Path, default=script_dir / "analysis_csv" / "alexnet_fas_nonzero_quantiles.md")
    parser.add_argument("--percentiles", type=str, default="20,40,60,80")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)
    percentiles = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    if len(percentiles) != 4:
        raise ValueError("--percentiles must contain exactly four values")

    input_scale, layers = parse_model(args.model)
    input_q = read_input(args.input)
    weight_lines = read_weight_lines(args.weights)
    weights = consume_weights(layers, weight_lines)
    activations = forward_quantized(input_q, input_scale, layers, weights)
    thresholds = [percentile_thresholds(arr, percentiles) for arr in activations]

    write_approx_file(args.approx_out, layers, thresholds)
    write_csv(args.csv_out, layers, activations, thresholds)
    write_summary(args.summary_out, args.image, args.approx_out, args.csv_out, layers, thresholds)

    print(f"[OK] updated {args.approx_out}")
    print(f"[OK] wrote {args.csv_out}")
    print(f"[OK] wrote {args.summary_out}")
    for idx, (spec, th) in enumerate(zip(layers, thresholds), start=1):
        print(f"layer {idx:02d} {spec.name:<5s} {spec.layer_type:<11s}: {th[0]} {th[1]} {th[2]} {th[3]}")


if __name__ == "__main__":
    main()
