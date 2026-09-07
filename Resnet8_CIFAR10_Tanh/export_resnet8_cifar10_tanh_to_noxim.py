#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "Resnet8_CIFAR10"))
sys.path.insert(0, str(SCRIPT_DIR))

from ResNet8Fused import ResNet8Fused  # noqa: E402
from export_resnet8_cifar10_to_noxim import (  # noqa: E402
    EPS,
    LayerScale,
    RESNET8_EDGES,
    choose_activation_scale,
    choose_layer_scale,
    ensure_clean_txt_dir,
    load_image_tensor,
    load_state_dict_compat,
    max_abs,
    parse_float_triplet,
    parse_label_from_name,
    print_confidences,
    quantize_int16,
    write_alias,
    write_input_file,
    write_line_of_ints,
    write_weight_file,
    write_weight_scale_file,
)


LAYER_ORDER: Sequence[Tuple[str, str]] = [
    ("Convolution", "conv1"),
    ("Convolution", "block1.conv1"),
    ("Convolution", "block1.conv2"),
    ("Add", "block1.add"),
    ("Convolution", "block2.conv1"),
    ("Convolution", "block2.conv2"),
    ("Convolution", "block2.shortcut.0"),
    ("Add", "block2.add"),
    ("Convolution", "block3.conv1"),
    ("Convolution", "block3.conv2"),
    ("Convolution", "block3.shortcut.0"),
    ("Add", "block3.add"),
    ("Pooling", "avgpool"),
    ("Dense", "fc"),
]


def quantile_thresholds(values: torch.Tensor) -> List[int]:
    vals = values.detach().abs().reshape(-1).to(torch.int64)
    vals = vals[vals != 0]
    if vals.numel() == 0:
        return [0, 0, 0, 0]
    vals, _ = torch.sort(vals)
    n = int(vals.numel())
    result: List[int] = []
    for p in (0.2, 0.4, 0.6, 0.8):
        idx = max(0, min(n - 1, int(torch.ceil(torch.tensor(p * n)).item()) - 1))
        result.append(int(vals[idx].item()))
    return result


def channelwise_diffs(q: torch.Tensor) -> torch.Tensor:
    q = q.detach().to(torch.int64)
    if q.ndim == 4:
        parts = []
        for n in range(q.shape[0]):
            for c in range(q.shape[1]):
                flat = q[n, c].reshape(-1)
                if flat.numel() > 1:
                    parts.append(flat[1:] - flat[:-1])
        if not parts:
            return torch.empty(0, dtype=torch.int64)
        return torch.cat(parts)
    flat = q.reshape(-1)
    if flat.numel() <= 1:
        return torch.empty(0, dtype=torch.int64)
    return flat[1:] - flat[:-1]


@torch.no_grad()
def forward_collect(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    feats: Dict[str, torch.Tensor] = {}
    x1 = torch.tanh(model.conv1(x))
    feats["conv1"] = x1

    b1c1 = torch.tanh(model.block1.conv1(x1))
    feats["block1.conv1"] = b1c1
    b1c2 = model.block1.conv2(b1c1)
    feats["block1.conv2"] = b1c2
    b1add = torch.tanh(b1c2 + x1)
    feats["block1.add"] = b1add

    b2c1 = torch.tanh(model.block2.conv1(b1add))
    feats["block2.conv1"] = b2c1
    b2c2 = model.block2.conv2(b2c1)
    feats["block2.conv2"] = b2c2
    b2sc = model.block2.shortcut[0](b1add)
    feats["block2.shortcut.0"] = b2sc
    b2add = torch.tanh(b2c2 + b2sc)
    feats["block2.add"] = b2add

    b3c1 = torch.tanh(model.block3.conv1(b2add))
    feats["block3.conv1"] = b3c1
    b3c2 = model.block3.conv2(b3c1)
    feats["block3.conv2"] = b3c2
    b3sc = model.block3.shortcut[0](b2add)
    feats["block3.shortcut.0"] = b3sc
    b3add = torch.tanh(b3c2 + b3sc)
    feats["block3.add"] = b3add

    pool = model.avgpool(b3add)
    feats["avgpool"] = pool
    logits = model.fc(torch.flatten(pool, 1))
    feats["fc"] = logits
    return feats


def write_model_file(path: Path, input_scale: float, s: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Input 32 32 3 {input_scale:.18g}\n")
        f.write(f"Convolution 32 32 16 3 3 3 1 1 tanh {s['conv1']:.18g} 0 0 {input_scale:.18g} 0 {s['conv1']:.18g} 0 0\n")
        f.write(f"Convolution 32 32 16 3 3 16 1 1 tanh {s['block1.conv1']:.18g} 0 0 {s['conv1']:.18g} 0 {s['block1.conv1']:.18g} 0 1\n")
        f.write(f"Convolution 32 32 16 3 3 16 1 1 none {s['block1.conv2']:.18g} 0 0 {s['block1.conv1']:.18g} 0 {s['block1.conv2']:.18g} 0 2\n")
        f.write(f"Add 32 32 16 3 1 tanh {s['block1.add']:.18g} 0\n")
        f.write(f"Convolution 16 16 32 3 3 16 2 1 tanh {s['block2.conv1']:.18g} 0 0 {s['block1.add']:.18g} 0 {s['block2.conv1']:.18g} 0 4\n")
        f.write(f"Convolution 16 16 32 3 3 32 1 1 none {s['block2.conv2']:.18g} 0 0 {s['block2.conv1']:.18g} 0 {s['block2.conv2']:.18g} 0 5\n")
        f.write(f"Convolution 16 16 32 1 1 16 2 0 none {s['block2.shortcut.0']:.18g} 0 0 {s['block1.add']:.18g} 0 {s['block2.shortcut.0']:.18g} 0 4\n")
        f.write(f"Add 16 16 32 6 7 tanh {s['block2.add']:.18g} 0\n")
        f.write(f"Convolution 8 8 64 3 3 32 2 1 tanh {s['block3.conv1']:.18g} 0 0 {s['block2.add']:.18g} 0 {s['block3.conv1']:.18g} 0 8\n")
        f.write(f"Convolution 8 8 64 3 3 64 1 1 none {s['block3.conv2']:.18g} 0 0 {s['block3.conv1']:.18g} 0 {s['block3.conv2']:.18g} 0 9\n")
        f.write(f"Convolution 8 8 64 1 1 32 2 0 none {s['block3.shortcut.0']:.18g} 0 0 {s['block2.add']:.18g} 0 {s['block3.shortcut.0']:.18g} 0 8\n")
        f.write(f"Add 8 8 64 10 11 tanh {s['block3.add']:.18g} 0\n")
        f.write("Pooling 1 1 64 8 8 8 average 0\n")
        f.write(f"Dense 10 none {s['fc']:.18g} 0 0 {s['block3.add']:.18g} 0 {s['fc']:.18g} 0\n")


def layer_quantized_outputs(feats: Dict[str, torch.Tensor], scales: Dict[str, float]) -> Dict[str, torch.Tensor]:
    q: Dict[str, torch.Tensor] = {}
    for _, lname in LAYER_ORDER:
        scale = scales["block3.add"] if lname == "avgpool" else scales[lname]
        q[lname] = quantize_int16(feats[lname], max(scale, EPS))
    return q


def write_threshold_files(path: Path, thresholds: Dict[str, List[int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for layer_type, lname in LAYER_ORDER:
            f.write(f"{layer_type} {' '.join(str(v) for v in thresholds[lname])}\n")


def copy_level_table(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


def parse_edge_template(path: Path, keyword: str) -> List[Tuple[int, int, int, List[int]]]:
    rows: List[Tuple[int, int, int, List[int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if parts[0] != keyword or len(parts) < 14:
            continue
        src = int(parts[1])
        dst = int(parts[2])
        config_sel = int(parts[3])
        levels = [int(v) for v in parts[8:14]]
        rows.append((src, dst, config_sel, levels))
    return rows


def write_edge_approx_file(path: Path, keyword: str, rows: List[Tuple[int, int, int, List[int]]], thresholds_by_layer: List[List[int]]) -> None:
    header = "Edge" if keyword == "Edge" else "SapEdge"
    with path.open("w", encoding="utf-8") as f:
        f.write(f"% {header} src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5\n")
        for src, dst, config_sel, levels in rows:
            th = thresholds_by_layer[src - 1]
            f.write(
                f"{keyword} {src} {dst} {config_sel} "
                f"{th[0]} {th[1]} {th[2]} {th[3]} "
                f"{' '.join(str(v) for v in levels)}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fused tanh ResNet8-CIFAR10 into cnn-noxim txt files")
    parser.add_argument("--model", type=Path, default=Path("Resnet8_CIFAR10_Tanh/best_model_fused.pth"))
    parser.add_argument("--image", type=Path, default=Path("CIFAR10/idx_01824_label_truck.png"))
    parser.add_argument("--outdir", type=Path, default=Path("bin/resnet8_cifar10_tanh"))
    parser.add_argument("--template-dir", type=Path, default=Path("bin/resnet8_cifar10"))
    parser.add_argument("--mean", type=str, default="0.4914,0.4822,0.4465")
    parser.add_argument("--std", type=str, default="0.2023,0.1994,0.2010")
    args = parser.parse_args()

    mean = parse_float_triplet(args.mean)
    std = parse_float_triplet(args.std)
    if mean is None or std is None:
        raise ValueError("Both --mean and --std are required")

    ensure_clean_txt_dir(args.outdir)

    model = ResNet8Fused(num_classes=10)
    state_dict = load_state_dict_compat(args.model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    x = load_image_tensor(args.image, mean, std)
    feats = forward_collect(model, x)
    logits = feats["fc"].squeeze(0)
    pred = int(torch.argmax(logits).item())
    print_confidences(logits)

    input_scale = max(max_abs(x) / 32767, EPS)
    x_q = quantize_int16(x, input_scale)

    weighted_layers: Sequence[str] = [
        "conv1",
        "block1.conv1",
        "block1.conv2",
        "block2.conv1",
        "block2.conv2",
        "block2.shortcut.0",
        "block3.conv1",
        "block3.conv2",
        "block3.shortcut.0",
        "fc",
    ]
    scales: Dict[str, LayerScale] = {}
    named_modules = dict(model.named_modules())
    for lname in weighted_layers:
        layer = named_modules[lname]
        scales[lname] = choose_layer_scale(layer.weight, layer.bias, feats[lname])
        scales[lname].name = lname
    for lname in ("block1.add", "block2.add", "block3.add"):
        scales[lname] = choose_activation_scale(lname, feats[lname])
    scale_values = {k: v.k for k, v in scales.items()}

    q_outputs = layer_quantized_outputs(feats, scale_values)
    fas_thresholds = {lname: quantile_thresholds(q_outputs[lname]) for _, lname in LAYER_ORDER}
    sap_thresholds = {lname: quantile_thresholds(channelwise_diffs(q_outputs[lname])) for _, lname in LAYER_ORDER}

    model_main = args.outdir / "resnet8_tanh_model.txt"
    weight_main = args.outdir / "resnet8_tanh_weight_fc_wb2parser.txt"
    input_main = args.outdir / "resnet8_tanh_input.txt"
    label_main = args.outdir / "resnet8_tanh_label.txt"
    fas_threshold_path = args.outdir / "resnet8_tanh_fas_threshold.txt"
    fas_level_path = args.outdir / "resnet8_tanh_fas_level_table.txt"
    fas_edge_approx_path = args.outdir / "resnet8_tanh_fas_edge_approx.txt"
    sap_threshold_path = args.outdir / "resnet8_tanh_sap_threshold.txt"
    sap_level_path = args.outdir / "resnet8_tanh_sap_level_table.txt"
    sap_edge_approx_path = args.outdir / "resnet8_tanh_sap_edge_approx.txt"
    weight_scale_path = args.outdir / "resnet8_tanh_weight_scale.txt"
    drop_path = args.outdir / "resnet8_tanh_drop.txt"
    abdtr_edge_drop_path = args.outdir / "resnet8_tanh_abdtr_edge_drop.txt"
    summary_path = args.outdir / "resnet8_tanh_quant_summary.txt"
    logits_path = args.outdir / "resnet8_tanh_pytorch_logits.txt"

    write_model_file(model_main, input_scale, scale_values)
    write_weight_file(weight_main, model, scale_values)
    write_input_file(input_main, x_q)
    write_threshold_files(fas_threshold_path, fas_thresholds)
    write_threshold_files(sap_threshold_path, sap_thresholds)
    copy_level_table(args.template_dir / "resnet8_fas_level_table.txt", fas_level_path)
    copy_level_table(args.template_dir / "resnet8_sap_level_table.txt", sap_level_path)

    fas_rows = parse_edge_template(args.template_dir / "resnet8_fas_edge_approx.txt", "Edge")
    sap_rows = parse_edge_template(args.template_dir / "resnet8_sap_edge_approx.txt", "SapEdge")
    fas_threshold_rows = [fas_thresholds[lname] for _, lname in LAYER_ORDER]
    sap_threshold_rows = [sap_thresholds[lname] for _, lname in LAYER_ORDER]
    write_edge_approx_file(fas_edge_approx_path, "Edge", fas_rows, fas_threshold_rows)
    write_edge_approx_file(sap_edge_approx_path, "SapEdge", sap_rows, sap_threshold_rows)

    write_weight_scale_file(weight_scale_path, scale_values)
    shutil.copyfile(args.template_dir / "resnet8_drop.txt", drop_path)
    shutil.copyfile(args.template_dir / "resnet8_abdtr_edge_drop.txt", abdtr_edge_drop_path)

    label_from_name = parse_label_from_name(args.image.name)
    label = label_from_name if label_from_name is not None else pred
    label_main.write_text(f"{label}\n", encoding="utf-8")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"image={args.image}\n")
        f.write(f"input_scale={input_scale:.18g}\n")
        f.write(f"mean={mean}\n")
        f.write(f"std={std}\n")
        f.write(f"pred={pred}\n")
        f.write(f"label={label}\n")
        f.write("threshold_rule_fas=nonzero abs(int16_activation) quantiles 20/40/60/80 per layer\n")
        f.write("threshold_rule_sap=nonzero abs(channelwise int16 activation diff) quantiles 20/40/60/80 per layer\n")
        f.write("level_and_edge_selection=copy ordinary resnet8 configs, with tanh thresholds substituted\n")
        for lname in list(weighted_layers) + ["block1.add", "block2.add", "block3.add"]:
            s = scales[lname]
            f.write(
                f"{lname}: K={s.k:.18g}, w_scale={s.w_scale:.18g}, "
                f"b_scale={s.b_scale:.18g}, act_scale={s.act_scale:.18g}\n"
            )

    probs = torch.softmax(logits, dim=0)
    with logits_path.open("w", encoding="utf-8") as f:
        f.write("logits: " + " ".join(f"{float(v):.12g}" for v in logits.tolist()) + "\n")
        f.write("probs : " + " ".join(f"{float(v):.12g}" for v in probs.tolist()) + "\n")
        f.write(f"pred={pred}\n")
        f.write(f"label={label}\n")

    aliases = [
        (model_main, args.outdir / "model.txt"),
        (model_main, args.outdir / "resnet8_model.txt"),
        (weight_main, args.outdir / "weight_fc_wb2parser.txt"),
        (weight_main, args.outdir / "resnet8_weight_fc_wb2parser.txt"),
        (input_main, args.outdir / "input.txt"),
        (input_main, args.outdir / "resnet8_input.txt"),
        (label_main, args.outdir / "label.txt"),
        (label_main, args.outdir / "resnet8_label.txt"),
        (fas_threshold_path, args.outdir / "approx.txt"),
        (fas_threshold_path, args.outdir / "resnet8_approx.txt"),
        (fas_threshold_path, args.outdir / "resnet8_fas_threshold.txt"),
        (fas_level_path, args.outdir / "approx_level_table.txt"),
        (fas_level_path, args.outdir / "resnet8_approx_level_table.txt"),
        (fas_level_path, args.outdir / "resnet8_fas_level_table.txt"),
        (fas_edge_approx_path, args.outdir / "edge_approx.txt"),
        (fas_edge_approx_path, args.outdir / "resnet8_edge_approx.txt"),
        (fas_edge_approx_path, args.outdir / "resnet8_fas_edge_approx.txt"),
        (sap_threshold_path, args.outdir / "sap_threshold.txt"),
        (sap_threshold_path, args.outdir / "resnet8_sap_threshold.txt"),
        (sap_level_path, args.outdir / "sap_level_table.txt"),
        (sap_level_path, args.outdir / "resnet8_sap_level_table.txt"),
        (sap_edge_approx_path, args.outdir / "sap_edge_approx.txt"),
        (sap_edge_approx_path, args.outdir / "resnet8_sap_edge_approx.txt"),
        (weight_scale_path, args.outdir / "weight_scale.txt"),
        (weight_scale_path, args.outdir / "resnet8_weight_scale.txt"),
        (drop_path, args.outdir / "drop.txt"),
        (drop_path, args.outdir / "resnet8_drop.txt"),
        (abdtr_edge_drop_path, args.outdir / "abdtr_edge_drop.txt"),
        (abdtr_edge_drop_path, args.outdir / "resnet8_abdtr_edge_drop.txt"),
        (summary_path, args.outdir / "quant_summary.txt"),
        (summary_path, args.outdir / "resnet8_quant_summary.txt"),
        (logits_path, args.outdir / "pytorch_logits.txt"),
        (logits_path, args.outdir / "resnet8_pytorch_logits.txt"),
    ]
    for src, dst in aliases:
        write_alias(src, dst)

    print(f"[OK] Export complete: {args.outdir}")


if __name__ == "__main__":
    main()
