#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


INT16_MIN = -32768
INT16_MAX = 32767
EPS = 1e-12

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


@dataclass
class LayerScale:
    name: str
    k: float
    w_scale: float
    b_scale: float
    act_scale: float


class FusedBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=True)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class FusedResNet8CIFAR10(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=True)
        self.block1 = FusedBasicBlock(16, 16, stride=1)
        self.block2 = FusedBasicBlock(16, 32, stride=2)
        self.block3 = FusedBasicBlock(32, 64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


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


def parse_float_triplet(text: str | None) -> Tuple[float, float, float] | None:
    if text is None:
        return None
    vals = [x.strip() for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected 3 comma-separated floats")
    return (float(vals[0]), float(vals[1]), float(vals[2]))


@torch.no_grad()
def load_image_tensor(image_path: Path, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


@torch.no_grad()
def forward_collect(model: FusedResNet8CIFAR10, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    feats: Dict[str, torch.Tensor] = {}
    x1 = F.relu(model.conv1(x))
    feats["conv1"] = x1

    b1c1 = F.relu(model.block1.conv1(x1))
    feats["block1.conv1"] = b1c1
    b1c2 = model.block1.conv2(b1c1)
    feats["block1.conv2"] = b1c2
    b1add = F.relu(b1c2 + x1)
    feats["block1.add"] = b1add

    b2c1 = F.relu(model.block2.conv1(b1add))
    feats["block2.conv1"] = b2c1
    b2c2 = model.block2.conv2(b2c1)
    feats["block2.conv2"] = b2c2
    b2sc = model.block2.shortcut[0](b1add)
    feats["block2.shortcut.0"] = b2sc
    b2add = F.relu(b2c2 + b2sc)
    feats["block2.add"] = b2add

    b3c1 = F.relu(model.block3.conv1(b2add))
    feats["block3.conv1"] = b3c1
    b3c2 = model.block3.conv2(b3c1)
    feats["block3.conv2"] = b3c2
    b3sc = model.block3.shortcut[0](b2add)
    feats["block3.shortcut.0"] = b3sc
    b3add = F.relu(b3c2 + b3sc)
    feats["block3.add"] = b3add

    pool = model.avgpool(b3add)
    feats["avgpool"] = pool
    logits = model.fc(torch.flatten(pool, 1))
    feats["fc"] = logits
    return feats


def max_abs(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item())


def choose_layer_scale(weight: torch.Tensor, bias: torch.Tensor, act: torch.Tensor) -> LayerScale:
    w_scale = max_abs(weight) / INT16_MAX
    b_scale = max_abs(bias) / INT16_MAX
    act_scale = max_abs(act) / INT16_MAX
    k = max(w_scale, b_scale, act_scale, EPS)
    return LayerScale(name="", k=k, w_scale=w_scale, b_scale=b_scale, act_scale=act_scale)


def choose_activation_scale(name: str, act: torch.Tensor) -> LayerScale:
    act_scale = max(max_abs(act) / INT16_MAX, EPS)
    return LayerScale(name=name, k=act_scale, w_scale=0.0, b_scale=0.0, act_scale=act_scale)


def quantize_int16(t: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.round(t / scale)
    q = torch.clamp(q, INT16_MIN, INT16_MAX)
    return q.to(torch.int32)


def write_line_of_ints(f, values: Iterable[int]) -> None:
    f.write(" ".join(str(int(v)) for v in values))
    f.write("\n")


def ensure_clean_txt_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.txt"):
        p.unlink()


def write_model_file(path: Path, input_scale: float, s: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Input 32 32 3 {input_scale:.18g}\n")
        f.write(f"Convolution 32 32 16 3 3 3 1 1 relu {s['conv1']:.18g} 0 0 {input_scale:.18g} 0 {s['conv1']:.18g} 0 0\n")
        f.write(f"Convolution 32 32 16 3 3 16 1 1 relu {s['block1.conv1']:.18g} 0 0 {s['conv1']:.18g} 0 {s['block1.conv1']:.18g} 0 1\n")
        f.write(f"Convolution 32 32 16 3 3 16 1 1 none {s['block1.conv2']:.18g} 0 0 {s['block1.conv1']:.18g} 0 {s['block1.conv2']:.18g} 0 2\n")
        f.write(f"Add 32 32 16 3 1 relu {s['block1.add']:.18g} 0\n")
        f.write(f"Convolution 16 16 32 3 3 16 2 1 relu {s['block2.conv1']:.18g} 0 0 {s['block1.add']:.18g} 0 {s['block2.conv1']:.18g} 0 4\n")
        f.write(f"Convolution 16 16 32 3 3 32 1 1 none {s['block2.conv2']:.18g} 0 0 {s['block2.conv1']:.18g} 0 {s['block2.conv2']:.18g} 0 5\n")
        f.write(f"Convolution 16 16 32 1 1 16 2 0 none {s['block2.shortcut.0']:.18g} 0 0 {s['block1.add']:.18g} 0 {s['block2.shortcut.0']:.18g} 0 4\n")
        f.write(f"Add 16 16 32 6 7 relu {s['block2.add']:.18g} 0\n")
        f.write(f"Convolution 8 8 64 3 3 32 2 1 relu {s['block3.conv1']:.18g} 0 0 {s['block2.add']:.18g} 0 {s['block3.conv1']:.18g} 0 8\n")
        f.write(f"Convolution 8 8 64 3 3 64 1 1 none {s['block3.conv2']:.18g} 0 0 {s['block3.conv1']:.18g} 0 {s['block3.conv2']:.18g} 0 9\n")
        f.write(f"Convolution 8 8 64 1 1 32 2 0 none {s['block3.shortcut.0']:.18g} 0 0 {s['block2.add']:.18g} 0 {s['block3.shortcut.0']:.18g} 0 8\n")
        f.write(f"Add 8 8 64 10 11 relu {s['block3.add']:.18g} 0\n")
        f.write("Pooling 1 1 64 8 8 8 average 0\n")
        f.write(f"Dense 10 none {s['fc']:.18g} 0 0 {s['block3.add']:.18g} 0 {s['fc']:.18g} 0\n")


def write_weight_file(path: Path, model: FusedResNet8CIFAR10, scales: Dict[str, float]) -> None:
    conv_layers: List[Tuple[str, nn.Conv2d]] = [
        ("conv1", model.conv1),
        ("block1.conv1", model.block1.conv1),
        ("block1.conv2", model.block1.conv2),
        ("block2.conv1", model.block2.conv1),
        ("block2.conv2", model.block2.conv2),
        ("block2.shortcut.0", model.block2.shortcut[0]),
        ("block3.conv1", model.block3.conv1),
        ("block3.conv2", model.block3.conv2),
        ("block3.shortcut.0", model.block3.shortcut[0]),
    ]
    with path.open("w", encoding="utf-8") as f:
        for lname, layer in conv_layers:
            scale = scales[lname]
            w = layer.weight.detach().cpu()
            b = layer.bias.detach().cpu()
            for oc in range(w.shape[0]):
                for ic in range(w.shape[1]):
                    write_line_of_ints(f, quantize_int16(w[oc, ic], scale).reshape(-1).tolist())
            write_line_of_ints(f, quantize_int16(b, scale).tolist())

        scale = scales["fc"]
        w = model.fc.weight.detach().cpu()
        b = model.fc.bias.detach().cpu()
        write_line_of_ints(f, quantize_int16(b, scale).tolist())
        for oc in range(w.shape[0]):
            write_line_of_ints(f, quantize_int16(w[oc], scale).tolist())


def write_input_file(path: Path, x_q: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in range(3):
            for h in range(32):
                write_line_of_ints(f, x_q[0, c, h, :].tolist())


RESNET8_EDGES = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 4),
    (4, 5),
    (5, 6),
    (4, 7),
    (6, 8),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (10, 12),
    (11, 12),
    (12, 13),
    (13, 14),
]

RESNET8_SHORTCUT_EDGES = {
    (1, 4),
    (4, 7),
    (7, 8),
    (8, 11),
    (11, 12),
}


def write_threshold_and_level_files(path_threshold: Path, path_level: Path) -> None:
    layer_types = [
        "Convolution",
        "Convolution",
        "Convolution",
        "Add",
        "Convolution",
        "Convolution",
        "Convolution",
        "Add",
        "Convolution",
        "Convolution",
        "Convolution",
        "Add",
        "Pooling",
        "Dense",
    ]
    with path_threshold.open("w", encoding="utf-8") as f:
        for t in layer_types:
            f.write(f"{t} 0 0 0 0\n")
    with path_level.open("w", encoding="utf-8") as f:
        for t in layer_types:
            f.write(f"{t} 0 1 2 3 3 3\n")


def write_fas_edge_approx_file(path: Path) -> None:
    # Layer ids follow resnet8_model.txt. Thresholds default to zero so the
    # generated table is a safe template for per-edge experiments.
    with path.open("w", encoding="utf-8") as f:
        f.write("% Edge src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5\n")
        for src, dst in RESNET8_EDGES:
            f.write(f"Edge {src} {dst} 0 0 0 0 0 0 1 2 3 3 3\n")


def write_sap_edge_approx_file(path: Path) -> None:
    # Use level -1 in a selected config to disable SAP-RLE compression on an edge.
    with path.open("w", encoding="utf-8") as f:
        f.write("% SapEdge src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5\n")
        for src, dst in RESNET8_EDGES:
            f.write(f"SapEdge {src} {dst} 0 0 0 0 0 0 1 2 3 3 3\n")


def write_abdtr_edge_drop_file(path: Path, default_interval: int = 14) -> None:
    # interval=-1 disables ABDTR on a specific edge.
    with path.open("w", encoding="utf-8") as f:
        f.write("% AbdtrEdge src_layer dst_layer interval\n")
        for src, dst in RESNET8_EDGES:
            interval = -1 if (src, dst) in RESNET8_SHORTCUT_EDGES else default_interval
            f.write(f"AbdtrEdge {src} {dst} {interval}\n")


def write_weight_scale_file(path: Path, s: Dict[str, float]) -> None:
    entries = [
        ("conv1", 16),
        ("block1.conv1", 16),
        ("block1.conv2", 16),
        ("block2.conv1", 32),
        ("block2.conv2", 32),
        ("block2.shortcut.0", 32),
        ("block3.conv1", 64),
        ("block3.conv2", 64),
        ("block3.shortcut.0", 64),
        ("fc", 10),
    ]
    with path.open("w", encoding="utf-8") as f:
        for lname, n in entries:
            f.write(" ".join([f"{s[lname]:.18g}"] * n))
            f.write("\n")


def parse_label_from_name(image_name: str) -> int | None:
    lower = image_name.lower()
    for idx, name in enumerate(CIFAR10_CLASSES):
        if f"label_{name}" in lower:
            return idx
    return None


def print_confidences(logits: torch.Tensor) -> None:
    probs = torch.softmax(logits, dim=0)
    top_idx = int(torch.argmax(probs).item())
    print("10-class confidences (softmax):")
    for i, p in enumerate(probs.tolist()):
        print(f"  class[{i}] {CIFAR10_CLASSES[i]:<10s}: {p:.8f}")
    print(f"predicted_class_index: {top_idx}")
    print(f"predicted_class_name : {CIFAR10_CLASSES[top_idx]}")


def write_alias(src: Path, dst: Path) -> None:
    if src != dst:
        shutil.copyfile(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fused ResNet8-CIFAR10 into cnn-noxim txt files")
    parser.add_argument("--model", type=Path, default=Path("Resnet8_CIFAR10/best_model_fused.pth"))
    parser.add_argument("--image", type=Path, default=Path("CIFAR10/idx_01824_label_truck.png"))
    parser.add_argument("--outdir", type=Path, default=Path("bin/resnet8_cifar10"))
    parser.add_argument("--mean", type=str, default="0.4914,0.4822,0.4465")
    parser.add_argument("--std", type=str, default="0.2023,0.1994,0.2010")
    args = parser.parse_args()

    mean = parse_float_triplet(args.mean)
    std = parse_float_triplet(args.std)
    if mean is None or std is None:
        raise ValueError("Both --mean and --std are required")

    ensure_clean_txt_dir(args.outdir)

    model = FusedResNet8CIFAR10(num_classes=10)
    state_dict = load_state_dict_compat(args.model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    x = load_image_tensor(args.image, mean, std)
    feats = forward_collect(model, x)
    logits = feats["fc"].squeeze(0)
    pred = int(torch.argmax(logits).item())
    print_confidences(logits)

    input_scale = max(max_abs(x) / INT16_MAX, EPS)
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

    model_main = args.outdir / "resnet8_model.txt"
    weight_main = args.outdir / "resnet8_weight_fc_wb2parser.txt"
    input_main = args.outdir / "resnet8_input.txt"
    label_main = args.outdir / "resnet8_label.txt"
    fas_threshold_path = args.outdir / "resnet8_fas_threshold.txt"
    fas_level_path = args.outdir / "resnet8_fas_level_table.txt"
    fas_edge_approx_path = args.outdir / "resnet8_fas_edge_approx.txt"
    sap_threshold_path = args.outdir / "resnet8_sap_threshold.txt"
    sap_level_path = args.outdir / "resnet8_sap_level_table.txt"
    sap_edge_approx_path = args.outdir / "resnet8_sap_edge_approx.txt"
    weight_scale_path = args.outdir / "resnet8_weight_scale.txt"
    drop_path = args.outdir / "resnet8_drop.txt"
    abdtr_edge_drop_path = args.outdir / "resnet8_abdtr_edge_drop.txt"
    summary_path = args.outdir / "resnet8_quant_summary.txt"
    logits_path = args.outdir / "resnet8_pytorch_logits.txt"

    scale_values = {k: v.k for k, v in scales.items()}
    write_model_file(model_main, input_scale, scale_values)
    write_weight_file(weight_main, model, scale_values)
    write_input_file(input_main, x_q)
    write_threshold_and_level_files(fas_threshold_path, fas_level_path)
    write_fas_edge_approx_file(fas_edge_approx_path)
    write_threshold_and_level_files(sap_threshold_path, sap_level_path)
    write_sap_edge_approx_file(sap_edge_approx_path)
    write_weight_scale_file(weight_scale_path, scale_values)
    drop_path.write_text("14\n", encoding="utf-8")
    write_abdtr_edge_drop_file(abdtr_edge_drop_path)

    label_from_name = parse_label_from_name(args.image.name)
    label = label_from_name if label_from_name is not None else pred
    label_main.write_text(f"{label}\n", encoding="utf-8")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"image={args.image}\n")
        f.write(f"input_scale={input_scale:.18g}\n")
        f.write(f"mean={mean}\n")
        f.write(f"std={std}\n")
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
        (weight_main, args.outdir / "weight_fc_wb2parser.txt"),
        (input_main, args.outdir / "input.txt"),
        (label_main, args.outdir / "label.txt"),
        (fas_threshold_path, args.outdir / "approx.txt"),
        (fas_threshold_path, args.outdir / "resnet8_approx.txt"),
        (fas_level_path, args.outdir / "approx_level_table.txt"),
        (fas_level_path, args.outdir / "resnet8_approx_level_table.txt"),
        (fas_edge_approx_path, args.outdir / "edge_approx.txt"),
        (fas_edge_approx_path, args.outdir / "resnet8_edge_approx.txt"),
        (sap_threshold_path, args.outdir / "sap_threshold.txt"),
        (sap_level_path, args.outdir / "sap_level_table.txt"),
        (sap_edge_approx_path, args.outdir / "sap_edge_approx.txt"),
        (weight_scale_path, args.outdir / "weight_scale.txt"),
        (drop_path, args.outdir / "drop.txt"),
        (abdtr_edge_drop_path, args.outdir / "abdtr_edge_drop.txt"),
        (summary_path, args.outdir / "quant_summary.txt"),
        (logits_path, args.outdir / "pytorch_logits.txt"),
    ]
    for src, dst in aliases:
        write_alias(src, dst)

    print(f"[OK] Export complete: {args.outdir}")


if __name__ == "__main__":
    main()
