#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from AlexNet import AlexNetCIFAR10


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
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def parse_float_triplet(text: str | None) -> Tuple[float, float, float] | None:
    if text is None:
        return None
    vals = [x.strip() for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected 3 comma-separated floats")
    return (float(vals[0]), float(vals[1]), float(vals[2]))


@torch.no_grad()
def load_image_tensor(
    image_path: Path,
    mean: Tuple[float, float, float] | None,
    std: Tuple[float, float, float] | None,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean_t) / std_t
    return x


@torch.no_grad()
def forward_collect(model: AlexNetCIFAR10, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    feats: Dict[str, torch.Tensor] = {}

    x = torch.relu(model.conv1(x))
    feats["conv1"] = x
    x = model.max_pooling_1(x)

    x = torch.relu(model.conv2(x))
    feats["conv2"] = x
    x = model.max_pooling_2(x)

    x = torch.relu(model.conv3(x))
    feats["conv3"] = x
    x = torch.relu(model.conv4(x))
    feats["conv4"] = x
    x = torch.relu(model.conv5(x))
    feats["conv5"] = x
    x = model.max_pooling_3(x)

    x = torch.flatten(x, 1)
    x = torch.relu(model.fc1(x))
    feats["fc1"] = x
    x = torch.relu(model.fc2(x))
    feats["fc2"] = x
    x = model.fc3(x)
    feats["fc3"] = x

    return feats


def max_abs(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item())


def choose_layer_scale(weight: torch.Tensor, bias: torch.Tensor, act: torch.Tensor) -> LayerScale:
    w_scale = max_abs(weight) / INT16_MAX
    b_scale = max_abs(bias) / INT16_MAX
    act_scale = max_abs(act) / INT16_MAX
    k = max(w_scale, b_scale, act_scale, EPS)
    return LayerScale(name="", k=k, w_scale=w_scale, b_scale=b_scale, act_scale=act_scale)


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

        f.write(f"Convolution 32 32 16 3 3 3 1 1 relu {s['conv1']:.18g} 0 0 {input_scale:.18g} 0 {s['conv1']:.18g} 0\n")
        f.write("Pooling 16 16 16 2 2 2 maximum 0\n")

        f.write(f"Convolution 16 16 48 3 3 16 1 1 relu {s['conv2']:.18g} 0 0 {s['conv1']:.18g} 0 {s['conv2']:.18g} 0\n")
        f.write("Pooling 8 8 48 2 2 2 maximum 0\n")

        f.write(f"Convolution 8 8 96 3 3 48 1 1 relu {s['conv3']:.18g} 0 0 {s['conv2']:.18g} 0 {s['conv3']:.18g} 0\n")
        f.write(f"Convolution 8 8 64 3 3 96 1 1 relu {s['conv4']:.18g} 0 0 {s['conv3']:.18g} 0 {s['conv4']:.18g} 0\n")
        f.write(f"Convolution 8 8 64 3 3 64 1 1 relu {s['conv5']:.18g} 0 0 {s['conv4']:.18g} 0 {s['conv5']:.18g} 0\n")
        f.write("Pooling 4 4 64 2 2 2 maximum 0\n")

        f.write(f"Dense 512 relu {s['fc1']:.18g} 0 0 {s['conv5']:.18g} 0 {s['fc1']:.18g} 0\n")
        f.write(f"Dense 256 relu {s['fc2']:.18g} 0 0 {s['fc1']:.18g} 0 {s['fc2']:.18g} 0\n")
        f.write(f"Dense 10 none {s['fc3']:.18g} 0 0 {s['fc2']:.18g} 0 {s['fc3']:.18g} 0\n")


def write_approx_files(path_approx: Path, path_level: Path) -> None:
    layer_types = [
        "Convolution",
        "Pooling",
        "Convolution",
        "Pooling",
        "Convolution",
        "Convolution",
        "Convolution",
        "Pooling",
        "Dense",
        "Dense",
        "Dense",
    ]
    with path_approx.open("w", encoding="utf-8") as f:
        for t in layer_types:
            f.write(f"{t} 0 0 0 0\n")
    with path_level.open("w", encoding="utf-8") as f:
        for t in layer_types:
            f.write(f"{t} 0 1 2 3 3 3\n")


def write_input_file(path: Path, x_q: torch.Tensor) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in range(3):
            for h in range(32):
                write_line_of_ints(f, x_q[0, c, h, :].tolist())


def write_weight_file(path: Path, model: AlexNetCIFAR10, scales: Dict[str, float]) -> None:
    conv_layers: List[Tuple[str, torch.nn.Conv2d]] = [
        ("conv1", model.conv1),
        ("conv2", model.conv2),
        ("conv3", model.conv3),
        ("conv4", model.conv4),
        ("conv5", model.conv5),
    ]
    with path.open("w", encoding="utf-8") as f:
        for lname, layer in conv_layers:
            scale = scales[lname]
            w = layer.weight.detach().cpu()
            b = layer.bias.detach().cpu()
            for oc in range(w.shape[0]):
                for ic in range(w.shape[1]):
                    kernel_q = quantize_int16(w[oc, ic], scale).reshape(-1).tolist()
                    write_line_of_ints(f, kernel_q)
            bias_q = quantize_int16(b, scale).tolist()
            write_line_of_ints(f, bias_q)

        dense_layers: List[Tuple[str, torch.nn.Linear]] = [
            ("fc1", model.fc1),
            ("fc2", model.fc2),
            ("fc3", model.fc3),
        ]
        for lname, layer in dense_layers:
            scale = scales[lname]
            w = layer.weight.detach().cpu()
            b = layer.bias.detach().cpu()
            write_line_of_ints(f, quantize_int16(b, scale).tolist())
            for oc in range(w.shape[0]):
                row_q = quantize_int16(w[oc], scale).tolist()
                write_line_of_ints(f, row_q)


def write_weight_scale_file(path: Path, s: Dict[str, float]) -> None:
    entries = [
        ("conv1", 16),
        ("conv2", 48),
        ("conv3", 96),
        ("conv4", 64),
        ("conv5", 64),
        ("fc1", 512),
        ("fc2", 256),
        ("fc3", 10),
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
    parser = argparse.ArgumentParser(description="Export AlexNet-CIFAR10 + single image into cnn-noxim txt files")
    parser.add_argument("--model", type=Path, default=Path("AlexNet_CIFAR10/best_model.pth"))
    parser.add_argument("--image", type=Path, default=Path("AlexNet_CIFAR10/idx_01824_label_truck.png"))
    parser.add_argument("--outdir", type=Path, default=Path("bin/alexnet_cifar10"))
    parser.add_argument("--mean", type=str, default="0.4914,0.4822,0.4465")
    parser.add_argument("--std", type=str, default="0.2023,0.1994,0.2010")
    args = parser.parse_args()

    mean = parse_float_triplet(args.mean)
    std = parse_float_triplet(args.std)
    if mean is None or std is None:
        raise ValueError("Both --mean and --std are required")

    ensure_clean_txt_dir(args.outdir)

    model = AlexNetCIFAR10(num_classes=10)
    state_dict = load_state_dict_compat(args.model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:10]}")
    model.eval()

    x = load_image_tensor(args.image, mean=mean, std=std)
    feats = forward_collect(model, x)

    with torch.no_grad():
        logits = feats["fc3"].squeeze(0)
        pred = int(torch.argmax(logits).item())

    print_confidences(logits)

    input_scale = max(max_abs(x) / INT16_MAX, EPS)
    x_q = quantize_int16(x, input_scale)

    layer_names: Sequence[str] = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "fc3"]
    scales: Dict[str, LayerScale] = {}
    for lname in layer_names:
        layer = getattr(model, lname)
        scales[lname] = choose_layer_scale(layer.weight, layer.bias, feats[lname])
        scales[lname].name = lname

    model_main = args.outdir / "alexnet_model_new.txt"
    weight_main = args.outdir / "alexnet_weight_fc_wb2parser.txt"
    input_main = args.outdir / "alexnet_input_cifar10_new.txt"
    label_main = args.outdir / "alexnet_label_cifar10.txt"
    approx_path = args.outdir / "alexnet_approx.txt"
    approx_level_path = args.outdir / "alexnet_approx_level_table.txt"
    weight_scale_path = args.outdir / "alexnet_weight_scale.txt"
    drop_path = args.outdir / "alexnet_drop.txt"
    summary_path = args.outdir / "alexnet_quant_summary.txt"
    logits_path = args.outdir / "alexnet_pytorch_logits.txt"

    write_model_file(model_main, input_scale, {k: v.k for k, v in scales.items()})
    write_weight_file(weight_main, model, {k: v.k for k, v in scales.items()})
    write_input_file(input_main, x_q)
    write_approx_files(approx_path, approx_level_path)
    write_weight_scale_file(weight_scale_path, {k: v.k for k, v in scales.items()})
    drop_path.write_text("9\n", encoding="utf-8")

    label_from_name = parse_label_from_name(args.image.name)
    label = label_from_name if label_from_name is not None else pred
    label_main.write_text(f"{label}\n", encoding="utf-8")

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"model={args.model}\n")
        f.write(f"image={args.image}\n")
        f.write(f"input_scale={input_scale:.18g}\n")
        f.write(f"mean={mean}\n")
        f.write(f"std={std}\n")
        for lname in layer_names:
            s = scales[lname]
            f.write(
                f"{lname}: K={s.k:.18g}, w_scale={s.w_scale:.18g}, "
                f"b_scale={s.b_scale:.18g}, act_scale={s.act_scale:.18g}\n"
            )

    with logits_path.open("w", encoding="utf-8") as f:
        probs = torch.softmax(logits, dim=0)
        f.write("logits: " + " ".join(f"{float(v):.12g}" for v in logits.tolist()) + "\n")
        f.write("probs : " + " ".join(f"{float(v):.12g}" for v in probs.tolist()) + "\n")
        f.write(f"pred={pred}\n")
        f.write(f"label={label}\n")

    # Compatibility aliases for existing noxim command templates.
    aliases = [
        (model_main, args.outdir / "model_new.txt"),
        (model_main, args.outdir / "model.txt"),
        (weight_main, args.outdir / "weight_fc_wb2parser.txt"),
        (weight_main, args.outdir / "weight.txt"),
        (input_main, args.outdir / "input_cifar10_new.txt"),
        (input_main, args.outdir / "input.txt"),
        (label_main, args.outdir / "label_cifar10.txt"),
        (label_main, args.outdir / "label.txt"),
        (approx_path, args.outdir / "approx.txt"),
        (approx_level_path, args.outdir / "approx_level_table.txt"),
        (weight_scale_path, args.outdir / "weight_scale.txt"),
        (drop_path, args.outdir / "drop.txt"),
        (summary_path, args.outdir / "quant_summary.txt"),
        (logits_path, args.outdir / "pytorch_logits.txt"),
    ]
    for src, dst in aliases:
        write_alias(src, dst)

    print(f"[OK] Export complete: {args.outdir}")
    print(f"[OK] model : {model_main}")
    print(f"[OK] weight: {weight_main}")
    print(f"[OK] input : {input_main}")
    print(f"[OK] label : {label_main} (value={label})")


if __name__ == "__main__":
    main()
