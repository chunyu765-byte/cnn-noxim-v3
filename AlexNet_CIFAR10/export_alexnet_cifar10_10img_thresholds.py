#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image

from stat_alexnet_fas_quantiles import (
    consume_weights,
    forward_quantized,
    parse_model,
    read_weight_lines,
)
from stat_alexnet_sap_diff_quantiles import (
    nearest_rank_quantiles,
    nonzero_adjacent_abs_diffs,
)


INT16_MIN = -32768
INT16_MAX = 32767

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_float_triplet(text: str) -> tuple[float, float, float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("Expected 3 comma-separated floats")
    return (vals[0], vals[1], vals[2])


def parse_label_from_name(image_name: str) -> int:
    lower = image_name.lower()
    for idx, name in enumerate(CIFAR10_CLASSES):
        if f"label_{name}" in lower:
            return idx
    raise ValueError(f"Cannot parse CIFAR-10 label from filename: {image_name}")


def load_image_tensor(
    image_path: Path,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def quantize_with_scale(t: torch.Tensor, scale: float) -> np.ndarray:
    q = torch.round(t / scale)
    q = torch.clamp(q, INT16_MIN, INT16_MAX)
    return q.to(torch.int32).cpu().numpy().astype(np.int32)


def write_line_of_ints(f, values: Iterable[int]) -> None:
    f.write(" ".join(str(int(v)) for v in values))
    f.write("\n")


def write_input_file(path: Path, images_q: Sequence[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for x_q in images_q:
            if x_q.shape != (3, 32, 32):
                raise RuntimeError(f"Expected image shape (3,32,32), got {x_q.shape}")
            for c in range(3):
                for h in range(32):
                    write_line_of_ints(f, x_q[c, h, :].tolist())


def percentile_thresholds_multi(values: Sequence[np.ndarray], percentiles: Sequence[float]) -> List[int]:
    if not values:
        return [0 for _ in percentiles]
    flat = np.concatenate([arr.reshape(-1).astype(np.int64) for arr in values])
    nonzero = flat[flat != 0]
    if nonzero.size == 0:
        return [0 for _ in percentiles]
    vals = np.abs(nonzero) if np.any(nonzero < 0) else nonzero
    return [int(round(float(np.percentile(vals, p)))) for p in percentiles]


def write_fas_threshold_file(path: Path, layer_types: Sequence[str], thresholds: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for layer_type, th in zip(layer_types, thresholds):
            if len(th) != 4:
                raise RuntimeError(f"{layer_type}: expected four thresholds")
            # cnn-noxim reverses this line while loading:
            # file order q80 q60 q40 q20 -> internal app_th0=q20, ..., app_th3=q80.
            f.write(f"{layer_type} {th[3]} {th[2]} {th[1]} {th[0]}\n")


def write_level_file(path: Path, layer_types: Sequence[str], source: Path | None) -> None:
    if source is not None and source.exists():
        path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        for layer_type in layer_types:
            f.write(f"{layer_type} 0 1 2 3 3 3\n")


def write_sap_threshold_file(path: Path, layer_types: Sequence[str], thresholds: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for layer_type, th in zip(layer_types, thresholds):
            if len(th) != 4:
                raise RuntimeError(f"{layer_type}: expected four thresholds")
            f.write(f"{layer_type} {th[0]} {th[1]} {th[2]} {th[3]}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export AlexNet-CIFAR10 10-image input and threshold files")
    parser.add_argument("--image-dir", type=Path, default=repo_root / "CIFAR10")
    parser.add_argument("--outdir", type=Path, default=repo_root / "bin" / "alexnet_cifar10")
    parser.add_argument("--model", type=Path, default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_model.txt")
    parser.add_argument(
        "--weights",
        type=Path,
        default=repo_root / "bin" / "alexnet_cifar10" / "alexnet_weight_fc_wb2parser.txt",
    )
    parser.add_argument("--mean", type=str, default="0.4914,0.4822,0.4465")
    parser.add_argument("--std", type=str, default="0.2023,0.1994,0.2010")
    parser.add_argument("--percentiles", type=str, default="20,40,60,80")
    args = parser.parse_args()

    mean = parse_float_triplet(args.mean)
    std = parse_float_triplet(args.std)
    percentiles = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    if len(percentiles) != 4:
        raise ValueError("--percentiles must contain exactly four values")

    image_paths = sorted(p for p in args.image_dir.glob("*.png") if p.is_file())
    if len(image_paths) != 10:
        raise RuntimeError(f"Expected 10 PNG images in {args.image_dir}, found {len(image_paths)}")

    input_scale, layers = parse_model(args.model)
    weights = consume_weights(layers, read_weight_lines(args.weights))
    layer_types = [spec.layer_type for spec in layers]

    images_q: List[np.ndarray] = []
    labels: List[int] = []
    all_activations: List[List[np.ndarray]] = [[] for _ in layers]
    clipped_values = 0

    for image_path in image_paths:
        x = load_image_tensor(image_path, mean=mean, std=std)
        unclipped = torch.round(x / input_scale)
        clipped_values += int(((unclipped < INT16_MIN) | (unclipped > INT16_MAX)).sum().item())
        x_q = quantize_with_scale(x, input_scale)[0]
        images_q.append(x_q)
        labels.append(parse_label_from_name(image_path.name))

        activations = forward_quantized(x_q, input_scale, layers, weights)
        for idx, act in enumerate(activations):
            all_activations[idx].append(act)

    fas_thresholds = [percentile_thresholds_multi(acts, percentiles) for acts in all_activations]
    sap_thresholds = []
    for acts in all_activations:
        diffs = [nonzero_adjacent_abs_diffs(act, same_sign_only=True) for act in acts]
        merged = np.concatenate([d for d in diffs if d.size]) if any(d.size for d in diffs) else np.asarray([], dtype=np.int64)
        sap_thresholds.append(nearest_rank_quantiles(merged, percentiles))

    args.outdir.mkdir(parents=True, exist_ok=True)
    input_out = args.outdir / "alexnet_input_10img.txt"
    label_out = args.outdir / "alexnet_label_10img.txt"
    approx_out = args.outdir / "alexnet_approx_10img.txt"
    approx_level_out = args.outdir / "alexnet_approx_level_table_10img.txt"
    sap_threshold_out = args.outdir / "alexnet_sap_threshold_10img.txt"
    sap_level_out = args.outdir / "alexnet_sap_level_table_10img.txt"
    summary_out = args.outdir / "alexnet_10img_threshold_summary.txt"

    write_input_file(input_out, images_q)
    label_out.write_text("\n".join(str(v) for v in labels) + "\n", encoding="utf-8")
    write_fas_threshold_file(approx_out, layer_types, fas_thresholds)
    write_level_file(approx_level_out, layer_types, args.outdir / "alexnet_approx_level_table.txt")
    write_sap_threshold_file(sap_threshold_out, layer_types, sap_thresholds)
    write_level_file(sap_level_out, layer_types, args.outdir / "alexnet_sap_level_table.txt")

    with summary_out.open("w", encoding="utf-8") as f:
        f.write(f"image_dir={args.image_dir}\n")
        f.write("images=" + ",".join(p.name for p in image_paths) + "\n")
        f.write("labels=" + " ".join(str(v) for v in labels) + "\n")
        f.write(f"input_scale={input_scale:.18g}\n")
        f.write(f"input_clipped_values={clipped_values}\n")
        f.write("layer type fas_q20 fas_q40 fas_q60 fas_q80 sap_q20 sap_q40 sap_q60 sap_q80\n")
        for idx, (layer_type, fas, sap) in enumerate(zip(layer_types, fas_thresholds, sap_thresholds), start=1):
            f.write(
                f"{idx} {layer_type} {fas[0]} {fas[1]} {fas[2]} {fas[3]} "
                f"{sap[0]} {sap[1]} {sap[2]} {sap[3]}\n"
            )

    print(f"[OK] wrote {input_out}")
    print(f"[OK] wrote {label_out}")
    print(f"[OK] wrote {approx_out}")
    print(f"[OK] wrote {approx_level_out}")
    print(f"[OK] wrote {sap_threshold_out}")
    print(f"[OK] wrote {sap_level_out}")
    print(f"[OK] wrote {summary_out}")
    print(f"[OK] images={len(image_paths)}, labels={labels}, input_clipped_values={clipped_values}")


if __name__ == "__main__":
    main()
