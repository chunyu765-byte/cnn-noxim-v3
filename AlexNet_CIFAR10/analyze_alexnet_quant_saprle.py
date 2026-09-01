#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from AlexNet import AlexNetCIFAR10


INT16_MIN = -32768
INT16_MAX = 32767
EPS = 1e-12

CONV_ORDER = ["conv1", "conv2", "conv3", "conv4", "conv5"]
DENSE_ORDER = ["fc1", "fc2", "fc3"]
POOL_SCALE_FROM = {
    "pool1": "conv1",
    "pool2": "conv2",
    "pool3": "conv5",
}


@dataclass
class LayerActivation:
    layer_id: int
    name: str
    tensor: torch.Tensor


def parse_float_triplet(text: str) -> Tuple[float, float, float]:
    vals = [x.strip() for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected 3 comma-separated floats")
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def load_state_dict_compat(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(obj)}")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def load_image_tensor(
    image_path: Path,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


@torch.no_grad()
def forward_collect(model: AlexNetCIFAR10, x: torch.Tensor) -> List[LayerActivation]:
    acts: List[LayerActivation] = []

    acts.append(LayerActivation(0, "input", x.detach().cpu()))

    x = torch.relu(model.conv1(x))
    acts.append(LayerActivation(1, "conv1", x.detach().cpu()))
    x = model.max_pooling_1(x)
    acts.append(LayerActivation(2, "pool1", x.detach().cpu()))

    x = torch.relu(model.conv2(x))
    acts.append(LayerActivation(3, "conv2", x.detach().cpu()))
    x = model.max_pooling_2(x)
    acts.append(LayerActivation(4, "pool2", x.detach().cpu()))

    x = torch.relu(model.conv3(x))
    acts.append(LayerActivation(5, "conv3", x.detach().cpu()))
    x = torch.relu(model.conv4(x))
    acts.append(LayerActivation(6, "conv4", x.detach().cpu()))
    x = torch.relu(model.conv5(x))
    acts.append(LayerActivation(7, "conv5", x.detach().cpu()))
    x = model.max_pooling_3(x)
    acts.append(LayerActivation(8, "pool3", x.detach().cpu()))

    x = torch.flatten(x, 1)
    x = torch.relu(model.fc1(x))
    acts.append(LayerActivation(9, "fc1", x.detach().cpu()))
    x = torch.relu(model.fc2(x))
    acts.append(LayerActivation(10, "fc2", x.detach().cpu()))
    x = model.fc3(x)
    acts.append(LayerActivation(11, "fc3", x.detach().cpu()))

    return acts


def max_abs(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item())


def quantize_int16(t: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.round(t / max(scale, EPS))
    q = torch.clamp(q, INT16_MIN, INT16_MAX)
    return q.to(torch.int32)


def parse_model_scales(model_txt: Path) -> Tuple[float, Dict[str, float]]:
    input_scale: float | None = None
    conv_scales: List[float] = []
    dense_scales: List[float] = []

    with model_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            if toks[0] == "Input":
                input_scale = float(toks[4])
            elif toks[0] == "Convolution":
                conv_scales.append(float(toks[-2]))
            elif toks[0] == "Dense":
                dense_scales.append(float(toks[-2]))

    if input_scale is None:
        raise RuntimeError(f"Input scale not found in {model_txt}")
    if len(conv_scales) < len(CONV_ORDER):
        raise RuntimeError(f"Expected {len(CONV_ORDER)} conv scales, got {len(conv_scales)}")
    if len(dense_scales) < len(DENSE_ORDER):
        raise RuntimeError(f"Expected {len(DENSE_ORDER)} dense scales, got {len(dense_scales)}")

    scale_map: Dict[str, float] = {}
    for i, name in enumerate(CONV_ORDER):
        scale_map[name] = conv_scales[i]
    for i, name in enumerate(DENSE_ORDER):
        scale_map[name] = dense_scales[i]
    for pool_name, from_name in POOL_SCALE_FROM.items():
        scale_map[pool_name] = scale_map[from_name]

    return input_scale, scale_map


def collect_quantized_by_layer(
    acts: Sequence[LayerActivation],
    input_scale: float,
    scale_map: Dict[str, float],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for act in acts:
        scale = input_scale if act.name == "input" else scale_map[act.name]
        q = quantize_int16(act.tensor, scale)
        out[act.name] = [int(v) for v in q.flatten().tolist()]
    return out


def sap_rle_compressed_size(group: Sequence[int], threshold: int) -> int:
    if not group:
        return 0

    kept = 1
    anchor = int(group[0])
    for x in group[1:]:
        cur = int(x)
        same_sign = ((cur >= 0 and anchor >= 0) or (cur < 0 and anchor < 0))
        if same_sign and abs(cur - anchor) <= threshold:
            continue
        kept += 1
        anchor = cur
    return kept


def build_thresholds_pow2_minus1(k_min: int, k_max: int) -> List[Tuple[int, int]]:
    if k_min < 0 or k_max < k_min:
        raise ValueError("Require 0 <= k_min <= k_max")
    return [(k, (1 << k) - 1) for k in range(k_min, k_max + 1)]


def selected_layers(acts: Sequence[LayerActivation], include_input: bool, include_final: bool) -> List[LayerActivation]:
    last_name = acts[-1].name if acts else ""
    out: List[LayerActivation] = []
    for act in acts:
        if act.name == "input" and not include_input:
            continue
        if act.name == last_name and not include_final:
            continue
        out.append(act)
    return out


def write_layer_stats_csv(
    out_csv: Path,
    acts: Sequence[LayerActivation],
    quantized_flat: Dict[str, List[int]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "layer_id",
            "layer_name",
            "num_values",
            "min",
            "max",
            "mean",
            "std",
            "num_zero",
            "zero_ratio",
            "num_positive",
            "num_negative",
            "max_abs",
        ])
        for act in acts:
            arr = np.asarray(quantized_flat[act.name], dtype=np.int64)
            if arr.size == 0:
                continue
            num_zero = int(np.count_nonzero(arr == 0))
            w.writerow([
                act.layer_id,
                act.name,
                int(arr.size),
                int(arr.min()),
                int(arr.max()),
                f"{float(arr.mean()):.8f}",
                f"{float(arr.std()):.8f}",
                num_zero,
                f"{float(num_zero) / float(arr.size):.8f}",
                int(np.count_nonzero(arr > 0)),
                int(np.count_nonzero(arr < 0)),
                int(np.max(np.abs(arr))),
            ])


def write_distribution_csv(
    out_csv: Path,
    acts: Sequence[LayerActivation],
    quantized_flat: Dict[str, List[int]],
    bins: int,
) -> None:
    if bins <= 0:
        raise ValueError("bins must be > 0")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer_id", "layer_name", "bin_index", "range_left", "range_right", "count"])
        for act in acts:
            arr = np.asarray(quantized_flat[act.name], dtype=np.int64)
            if arr.size == 0:
                continue
            lo = int(arr.min())
            hi = int(arr.max())
            if lo == hi:
                w.writerow([act.layer_id, act.name, 0, lo, hi, int(arr.size)])
                continue
            hist, edges = np.histogram(arr, bins=bins, range=(lo, hi))
            for i, count in enumerate(hist.tolist()):
                w.writerow([
                    act.layer_id,
                    act.name,
                    i,
                    f"{float(edges[i]):.8f}",
                    f"{float(edges[i + 1]):.8f}",
                    int(count),
                ])


def write_zero_skip_csv(
    out_csv: Path,
    acts: Sequence[LayerActivation],
    quantized_flat: Dict[str, List[int]],
    group_size: int,
    include_input: bool,
    include_final: bool,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "layer_id",
            "layer_name",
            "group_size",
            "num_values",
            "total_groups",
            "all_zero_groups",
            "kept_values",
            "removed_values",
            "compression_ratio",
        ])

        total_values = 0
        total_groups = 0
        total_all_zero = 0
        total_kept = 0

        for act in selected_layers(acts, include_input, include_final):
            vals = quantized_flat[act.name]
            groups = 0
            all_zero = 0
            kept = 0
            for i in range(0, len(vals), group_size):
                g = vals[i : i + group_size]
                if not g:
                    continue
                groups += 1
                nonzero = sum(1 for x in g if int(x) != 0)
                kept += nonzero
                if nonzero == 0:
                    all_zero += 1

            removed = len(vals) - kept
            ratio = float(removed) / float(len(vals)) if vals else 0.0
            w.writerow([
                act.layer_id,
                act.name,
                group_size,
                len(vals),
                groups,
                all_zero,
                kept,
                removed,
                f"{ratio:.8f}",
            ])

            total_values += len(vals)
            total_groups += groups
            total_all_zero += all_zero
            total_kept += kept

        total_removed = total_values - total_kept
        total_ratio = float(total_removed) / float(total_values) if total_values else 0.0
        w.writerow([
            -1,
            "TOTAL",
            group_size,
            total_values,
            total_groups,
            total_all_zero,
            total_kept,
            total_removed,
            f"{total_ratio:.8f}",
        ])


def write_saprle_csv(
    out_csv: Path,
    acts: Sequence[LayerActivation],
    quantized_flat: Dict[str, List[int]],
    group_size: int,
    thresholds: Sequence[Tuple[int, int]],
    include_input: bool,
    include_final: bool,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "layer_id",
        "layer_name",
        "k",
        "threshold",
        "group_size",
        "num_values",
        "total_groups",
    ]
    for i in range(1, group_size + 1):
        header.append(f"compressed_size_{i}")
    header.extend(["kept_values", "removed_values", "avg_compressed_size", "compression_ratio"])

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        layers = selected_layers(acts, include_input, include_final)
        for k, threshold in thresholds:
            k_total_values = 0
            k_total_groups = 0
            k_total_kept = 0
            k_counts = {i: 0 for i in range(1, group_size + 1)}

            for act in layers:
                vals = quantized_flat[act.name]
                counts = {i: 0 for i in range(1, group_size + 1)}
                groups = 0
                kept = 0
                for i in range(0, len(vals), group_size):
                    g = vals[i : i + group_size]
                    if not g:
                        continue
                    csz = sap_rle_compressed_size(g, threshold)
                    csz = max(1, min(group_size, csz))
                    counts[csz] += 1
                    groups += 1
                    kept += csz

                removed = len(vals) - kept
                avg = float(kept) / float(groups) if groups else 0.0
                ratio = float(removed) / float(len(vals)) if vals else 0.0
                row = [
                    act.layer_id,
                    act.name,
                    k,
                    threshold,
                    group_size,
                    len(vals),
                    groups,
                ]
                for i in range(1, group_size + 1):
                    row.append(counts[i])
                row.extend([kept, removed, f"{avg:.8f}", f"{ratio:.8f}"])
                w.writerow(row)

                k_total_values += len(vals)
                k_total_groups += groups
                k_total_kept += kept
                for i in range(1, group_size + 1):
                    k_counts[i] += counts[i]

            k_removed = k_total_values - k_total_kept
            k_avg = float(k_total_kept) / float(k_total_groups) if k_total_groups else 0.0
            k_ratio = float(k_removed) / float(k_total_values) if k_total_values else 0.0
            total_row = [-1, "TOTAL", k, threshold, group_size, k_total_values, k_total_groups]
            for i in range(1, group_size + 1):
                total_row.append(k_counts[i])
            total_row.extend([k_total_kept, k_removed, f"{k_avg:.8f}", f"{k_ratio:.8f}"])
            w.writerow(total_row)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Analyze quantized AlexNet-CIFAR10 activations for cnn-noxim communication schemes."
    )
    parser.add_argument("--model", type=Path, default=script_dir / "best_model.pth")
    parser.add_argument("--image", type=Path, default=script_dir / "idx_01824_label_truck.png")
    parser.add_argument("--model-txt", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "model_new.txt")
    parser.add_argument("--outdir", type=Path, default=script_dir / "analysis_csv")
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--k-min", type=int, default=0)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--mean", type=str, default="0.4914,0.4822,0.4465")
    parser.add_argument("--std", type=str, default="0.2023,0.1994,0.2010")
    parser.add_argument("--include-input", action="store_true")
    parser.add_argument("--include-final", action="store_true")
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("group-size must be > 0")

    mean = parse_float_triplet(args.mean)
    std = parse_float_triplet(args.std)

    model = AlexNetCIFAR10(num_classes=10)
    state_dict = load_state_dict_compat(args.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:10]}")
    model.eval()

    x = load_image_tensor(args.image, mean, std)
    acts = forward_collect(model, x)
    input_scale, scale_map = parse_model_scales(args.model_txt)
    quantized_flat = collect_quantized_by_layer(acts, input_scale, scale_map)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stats_csv = args.outdir / "alexnet_quantized_layer_stats.csv"
    dist_csv = args.outdir / "alexnet_activation_distribution_100bins.csv"
    zero_csv = args.outdir / f"alexnet_zeroskip_group{args.group_size}_summary.csv"
    saprle_csv = args.outdir / f"alexnet_saprle_group{args.group_size}_summary.csv"

    write_layer_stats_csv(stats_csv, acts, quantized_flat)
    write_distribution_csv(dist_csv, acts, quantized_flat, args.bins)
    write_zero_skip_csv(zero_csv, acts, quantized_flat, args.group_size, args.include_input, args.include_final)
    write_saprle_csv(
        saprle_csv,
        acts,
        quantized_flat,
        args.group_size,
        build_thresholds_pow2_minus1(args.k_min, args.k_max),
        args.include_input,
        args.include_final,
    )

    logits = acts[-1].tensor.squeeze(0)
    probs = torch.softmax(logits, dim=0)
    print(f"[OK] scale source: {args.model_txt}")
    print(f"[OK] predicted_class_index: {int(torch.argmax(logits).item())}")
    print("[OK] probs: " + " ".join(f"{float(p):.8f}" for p in probs.tolist()))
    print(f"[OK] stats csv   : {stats_csv}")
    print(f"[OK] dist csv    : {dist_csv}")
    print(f"[OK] zero-skip csv: {zero_csv}")
    print(f"[OK] sap-rle csv  : {saprle_csv}")


if __name__ == "__main__":
    main()
