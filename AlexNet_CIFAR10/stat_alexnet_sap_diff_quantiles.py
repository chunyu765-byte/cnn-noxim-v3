#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Sequence

import numpy as np

from stat_alexnet_fas_quantiles import (
    consume_weights,
    forward_quantized,
    parse_model,
    read_input,
    read_weight_lines,
)


def parse_percentiles(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("--percentiles must contain exactly four values")
    for p in vals:
        if p < 0.0 or p > 100.0:
            raise ValueError("--percentiles values must be in [0, 100]")
    return vals


def nonzero_adjacent_abs_diffs(values: np.ndarray, same_sign_only: bool) -> np.ndarray:
    flat = values.reshape(-1).astype(np.int64)
    if flat.size < 2:
        return np.asarray([], dtype=np.int64)

    prev = flat[:-1]
    cur = flat[1:]
    diffs = np.abs(cur - prev)
    mask = diffs != 0
    if same_sign_only:
        mask &= ((prev >= 0) & (cur >= 0)) | ((prev < 0) & (cur < 0))
    return diffs[mask]


def nearest_rank_quantiles(values: np.ndarray, percentiles: Sequence[float]) -> List[int]:
    if values.size == 0:
        return [0 for _ in percentiles]
    sorted_vals = np.sort(values.astype(np.int64))
    n = sorted_vals.size
    out: List[int] = []
    for p in percentiles:
        # "去0重排序之后保留百分位位置的值": use nearest-rank indexing
        # instead of interpolating between two adjacent integer deltas.
        idx = int(np.ceil((p / 100.0) * n)) - 1
        idx = max(0, min(n - 1, idx))
        out.append(int(sorted_vals[idx]))
    return out


def write_threshold_file(path: Path, layer_types: Sequence[str], thresholds: Sequence[Sequence[int]]) -> None:
    if len(layer_types) != len(thresholds):
        raise RuntimeError("layer type count and threshold count mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for layer_type, th in zip(layer_types, thresholds):
            if len(th) != 4:
                raise RuntimeError(f"{layer_type}: expected four thresholds")
            f.write(f"{layer_type} {th[0]} {th[1]} {th[2]} {th[3]}\n")


def write_csv(
    path: Path,
    layer_types: Sequence[str],
    layer_names: Sequence[str],
    activations: Sequence[np.ndarray],
    diff_values: Sequence[np.ndarray],
    thresholds: Sequence[Sequence[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "layer_id",
            "layer_type",
            "layer_name",
            "num_activations",
            "num_adjacent_pairs",
            "num_nonzero_diffs",
            "q20_diff",
            "q40_diff",
            "q60_diff",
            "q80_diff",
            "min_diff",
            "max_diff",
            "mean_diff",
        ])
        for idx, (layer_type, layer_name, arr, diffs, th) in enumerate(
            zip(layer_types, layer_names, activations, diff_values, thresholds),
            start=1,
        ):
            if diffs.size:
                min_diff = int(diffs.min())
                max_diff = int(diffs.max())
                mean_diff = f"{float(diffs.mean()):.8f}"
            else:
                min_diff = 0
                max_diff = 0
                mean_diff = "0.00000000"
            num_values = int(arr.size)
            w.writerow([
                idx,
                layer_type,
                layer_name,
                num_values,
                max(0, num_values - 1),
                int(diffs.size),
                th[0],
                th[1],
                th[2],
                th[3],
                min_diff,
                max_diff,
                mean_diff,
            ])


def write_summary(
    path: Path,
    model_path: Path,
    input_path: Path,
    weights_path: Path,
    threshold_path: Path,
    csv_path: Path,
    same_sign_only: bool,
    layer_types: Sequence[str],
    layer_names: Sequence[str],
    diff_values: Sequence[np.ndarray],
    thresholds: Sequence[Sequence[int]],
) -> None:
    rows = [
        "| Layer | Type | Name | Non-zero diffs | q20 | q40 | q60 | q80 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, (layer_type, layer_name, diffs, th) in enumerate(
        zip(layer_types, layer_names, diff_values, thresholds),
        start=1,
    ):
        rows.append(
            f"| {idx} | {layer_type} | {layer_name} | {int(diffs.size)} | "
            f"{th[0]} | {th[1]} | {th[2]} | {th[3]} |"
        )

    text = "\n".join([
        "# AlexNet-CIFAR10 SAP-RLE Differential Thresholds",
        "",
        f"- Model file: `{model_path}`",
        f"- Input file: `{input_path}`",
        f"- Weight file: `{weights_path}`",
        f"- Updated threshold file: `{threshold_path}`",
        f"- CSV: `{csv_path}`",
        "- Quantization path: cnn-noxim exported model/input/weight files, then int16 layer outputs.",
        "- Differential statistic: absolute adjacent activation difference after flattening each quantized layer output.",
        f"- Same-sign filter: `{same_sign_only}`.",
        "- Zero differences are removed before sorting; q20/q40/q60/q80 use nearest-rank positions.",
        "",
        *rows,
        "",
    ])
    path.write_text(text, encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate SAP-RLE per-layer differential thresholds for AlexNet-CIFAR10."
    )
    parser.add_argument("--model", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_model.txt")
    parser.add_argument("--input", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_input.txt")
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_weight_fc_wb2parser.txt",
    )
    parser.add_argument(
        "--threshold-out",
        type=Path,
        default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_sap_threshold.txt",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=script_dir / "analysis_csv" / "alexnet_sap_diff_quantiles.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=script_dir / "analysis_csv" / "alexnet_sap_diff_quantiles.md",
    )
    parser.add_argument("--percentiles", type=str, default="20,40,60,80")
    parser.add_argument(
        "--include-cross-sign",
        action="store_true",
        help="Also count adjacent pairs with different signs. By default they are skipped to match SAP-RLE compression.",
    )
    args = parser.parse_args()

    percentiles = parse_percentiles(args.percentiles)
    input_scale, layers = parse_model(args.model)
    input_q = read_input(args.input)
    weight_lines = read_weight_lines(args.weights)
    weights = consume_weights(layers, weight_lines)
    activations = forward_quantized(input_q, input_scale, layers, weights)

    layer_types = [spec.layer_type for spec in layers]
    layer_names = [spec.name for spec in layers]
    same_sign_only = not args.include_cross_sign
    diff_values = [nonzero_adjacent_abs_diffs(arr, same_sign_only) for arr in activations]
    thresholds = [nearest_rank_quantiles(diffs, percentiles) for diffs in diff_values]

    write_threshold_file(args.threshold_out, layer_types, thresholds)
    write_csv(args.csv_out, layer_types, layer_names, activations, diff_values, thresholds)
    write_summary(
        args.summary_out,
        args.model,
        args.input,
        args.weights,
        args.threshold_out,
        args.csv_out,
        same_sign_only,
        layer_types,
        layer_names,
        diff_values,
        thresholds,
    )

    print(f"[OK] updated {args.threshold_out}")
    print(f"[OK] wrote {args.csv_out}")
    print(f"[OK] wrote {args.summary_out}")
    for idx, (layer_type, layer_name, diffs, th) in enumerate(
        zip(layer_types, layer_names, diff_values, thresholds),
        start=1,
    ):
        print(
            f"layer {idx:02d} {layer_name:<5s} {layer_type:<11s}: "
            f"{th[0]} {th[1]} {th[2]} {th[3]} "
            f"(nonzero_diffs={int(diffs.size)})"
        )


if __name__ == "__main__":
    main()
