#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from export_resnet8_cifar10_to_noxim import FusedResNet8CIFAR10, load_state_dict_compat


INT16_MIN = -32768
INT16_MAX = 32767
EPS = 1e-12
QUANTILES = (0.2, 0.4, 0.6, 0.8)
BatchList = List[Tuple[torch.Tensor, torch.Tensor]]

LAYER_NAMES: Dict[int, str] = {
    1: "conv1",
    2: "block1.conv1",
    3: "block1.conv2",
    4: "block1.add",
    5: "block2.conv1",
    6: "block2.conv2",
    7: "block2.shortcut.0",
    8: "block2.add",
    9: "block3.conv1",
    10: "block3.conv2",
    11: "block3.shortcut.0",
    12: "block3.add",
    13: "avgpool",
    14: "fc",
}

LAYER_TYPES: Dict[int, str] = {
    1: "Convolution",
    2: "Convolution",
    3: "Convolution",
    4: "Add",
    5: "Convolution",
    6: "Convolution",
    7: "Convolution",
    8: "Add",
    9: "Convolution",
    10: "Convolution",
    11: "Convolution",
    12: "Add",
    13: "Pooling",
    14: "Dense",
}

LAYER_NEURONS: Dict[int, int] = {
    1: 32 * 32 * 16,
    2: 32 * 32 * 16,
    3: 32 * 32 * 16,
    4: 32 * 32 * 16,
    5: 16 * 16 * 32,
    6: 16 * 16 * 32,
    7: 16 * 16 * 32,
    8: 16 * 16 * 32,
    9: 8 * 8 * 64,
    10: 8 * 8 * 64,
    11: 8 * 8 * 64,
    12: 8 * 8 * 64,
    13: 1 * 1 * 64,
    14: 10,
}

EDGES: Tuple[Tuple[int, int], ...] = (
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
)

SHORTCUT_EDGES = {
    (1, 4),
    (4, 7),
    (7, 8),
    (8, 11),
    (11, 12),
}

ACTIVE_SOURCE_LAYERS = sorted({src for src, dst in EDGES if (src, dst) not in SHORTCUT_EDGES})
LAYER_SEARCH_ORDER = sorted(ACTIVE_SOURCE_LAYERS, key=lambda layer: (-LAYER_NEURONS[layer], layer))


@dataclass
class EvalResult:
    accuracy: float
    correct: int
    total: int
    approx_values: int
    total_values: int
    seconds: float

    @property
    def approx_ratio(self) -> float:
        if self.total_values == 0:
            return 0.0
        return self.approx_values / self.total_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search ResNet8 CIFAR10 approximate communication configs on CPU."
    )
    parser.add_argument("--model", type=Path, default=Path("Resnet8_CIFAR10/best_model_fused.pth"))
    parser.add_argument("--data-root", type=Path, default=Path("Resnet8_CIFAR10/data"))
    parser.add_argument("--download-cifar10", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-abdtr-interval", type=int, default=128)
    parser.add_argument(
        "--schemes",
        type=str,
        default="fas,sap-rle,sap-rle-v2,abdtr",
        help="Comma-separated subset: fas,sap-rle,sap-rle-v2,abdtr",
    )
    parser.add_argument("--out-md", type=Path, default=Path("Resnet8_CIFAR10/resnet8_python_search_results.md"))
    parser.add_argument("--out-json", type=Path, default=Path("Resnet8_CIFAR10/resnet8_python_search_results.json"))
    return parser.parse_args()


def quantize_int16(x: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.round(x / max(scale, EPS))
    return torch.clamp(q, INT16_MIN, INT16_MAX).to(torch.int32)


def dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantile_thresholds_from_hist(hist: torch.Tensor) -> List[int]:
    total = int(hist[1:].sum().item())
    if total <= 0:
        return [0, 0, 0, 0]
    cdf = torch.cumsum(hist, dim=0)
    thresholds: List[int] = []
    for q in QUANTILES:
        target = max(1, int(math.ceil(total * q)))
        idx = int(torch.searchsorted(cdf, target + int(hist[0].item()), right=False).item())
        thresholds.append(max(0, min(INT16_MAX, idx)))
    return thresholds


def normalize_cifar10_tensor(data: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.2023, 0.1994, 0.2010), dtype=torch.float32).view(1, 3, 1, 1)
    return (data - mean) / std


def batchify_tensors(data: torch.Tensor, labels: torch.Tensor, batch_size: int) -> BatchList:
    batches: BatchList = []
    for start in range(0, int(labels.numel()), batch_size):
        end = min(start + batch_size, int(labels.numel()))
        batches.append((data[start:end].contiguous(), labels[start:end].contiguous()))
    return batches


def load_cifar10(args: argparse.Namespace) -> BatchList:
    test_batch = args.data_root / "cifar-10-batches-py" / "test_batch"
    if test_batch.exists():
        with test_batch.open("rb") as f:
            obj = pickle.load(f, encoding="latin1")
        data = torch.from_numpy(obj["data"]).to(torch.float32).reshape(-1, 3, 32, 32) / 255.0
        labels = torch.tensor(obj["labels"], dtype=torch.long)
        return batchify_tensors(normalize_cifar10_tensor(data), labels, args.batch_size)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=str(args.data_root),
        train=False,
        transform=transform,
        download=args.download_cifar10,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    return [(x.contiguous(), y.contiguous()) for x, y in loader]


def load_model(model_path: Path) -> FusedResNet8CIFAR10:
    model = FusedResNet8CIFAR10(num_classes=10)
    model.load_state_dict(load_state_dict_compat(model_path), strict=True)
    model.eval()
    return model


def forward_exact(model: FusedResNet8CIFAR10, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    feats: Dict[int, torch.Tensor] = {}
    l1 = F.relu(model.conv1(x))
    feats[1] = l1

    l2 = F.relu(model.block1.conv1(l1))
    feats[2] = l2
    l3 = model.block1.conv2(l2)
    feats[3] = l3
    l4 = F.relu(l3 + l1)
    feats[4] = l4

    l5 = F.relu(model.block2.conv1(l4))
    feats[5] = l5
    l6 = model.block2.conv2(l5)
    feats[6] = l6
    l7 = model.block2.shortcut[0](l4)
    feats[7] = l7
    l8 = F.relu(l6 + l7)
    feats[8] = l8

    l9 = F.relu(model.block3.conv1(l8))
    feats[9] = l9
    l10 = model.block3.conv2(l9)
    feats[10] = l10
    l11 = model.block3.shortcut[0](l8)
    feats[11] = l11
    l12 = F.relu(l10 + l11)
    feats[12] = l12

    l13 = model.avgpool(l12)
    feats[13] = l13
    logits = model.fc(torch.flatten(l13, 1))
    feats[14] = logits
    return logits, feats


def tensor_rows_by_channel(q: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    shape = tuple(q.shape)
    if q.dim() == 4:
        rows = q.reshape(q.shape[0], q.shape[1], -1).reshape(-1, q.shape[2] * q.shape[3])
    elif q.dim() == 2:
        rows = q.reshape(q.shape[0], 1, q.shape[1]).reshape(-1, q.shape[1])
    else:
        rows = q.reshape(q.shape[0], 1, -1).reshape(-1, q.numel() // q.shape[0])
    return rows, shape


def rows_to_tensor(rows: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    if len(shape) == 4:
        return rows.reshape(shape[0], shape[1], shape[2], shape[3])
    if len(shape) == 2:
        return rows.reshape(shape[0], shape[1])
    return rows.reshape(shape)


def collect_scales(model: FusedResNet8CIFAR10, loader: BatchList) -> Tuple[Dict[int, float], EvalResult]:
    max_abs = {layer: 0.0 for layer in LAYER_NAMES}
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            logits, feats = forward_exact(model, x)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
            for layer, t in feats.items():
                cur = float(t.detach().abs().max().item())
                if cur > max_abs[layer]:
                    max_abs[layer] = cur
    scales = {layer: max(max_abs[layer] / INT16_MAX, EPS) for layer in LAYER_NAMES}
    result = EvalResult(correct / total, correct, total, 0, 0, time.time() - start)
    return scales, result


def collect_thresholds(
    model: FusedResNet8CIFAR10,
    loader: BatchList,
    scales: Mapping[int, float],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    fas_hists = {layer: torch.zeros(INT16_MAX + 1, dtype=torch.long) for layer in LAYER_NAMES}
    sap_hists = {layer: torch.zeros(INT16_MAX + 1, dtype=torch.long) for layer in LAYER_NAMES}

    with torch.no_grad():
        for x, _ in loader:
            _, feats = forward_exact(model, x)
            for layer, t in feats.items():
                q = quantize_int16(t, scales[layer])
                abs_q = q.abs().clamp_(0, INT16_MAX).reshape(-1)
                fas_hists[layer] += torch.bincount(abs_q, minlength=INT16_MAX + 1)

                rows, _ = tensor_rows_by_channel(q)
                if rows.shape[1] > 1:
                    diffs = (rows[:, 1:] - rows[:, :-1]).abs().clamp_(0, INT16_MAX).reshape(-1)
                    sap_hists[layer] += torch.bincount(diffs, minlength=INT16_MAX + 1)

    fas = {layer: quantile_thresholds_from_hist(hist) for layer, hist in fas_hists.items()}
    sap = {layer: quantile_thresholds_from_hist(hist) for layer, hist in sap_hists.items()}
    return fas, sap


def apply_fas(
    x: torch.Tensor,
    layer: int,
    level: int,
    thresholds: Mapping[int, Sequence[int]],
    scales: Mapping[int, float],
) -> Tuple[torch.Tensor, int, int]:
    total = int(x.numel())
    if level < 0:
        return x, 0, total

    th = thresholds[layer]
    q = quantize_int16(x, scales[layer])
    abs_q = q.abs()
    sign = torch.sign(q)
    out = q.clone()
    approx_mask = torch.zeros_like(abs_q, dtype=torch.bool)

    if level >= 0:
        mask = abs_q <= th[0]
        out = torch.where(mask, torch.zeros_like(out), out)
        approx_mask |= mask
    if level >= 1:
        mask = (~approx_mask) & (abs_q <= th[1])
        out = torch.where(mask, sign * int(th[0]), out)
        approx_mask |= mask
    if level >= 2:
        mask = (~approx_mask) & (abs_q <= th[2])
        out = torch.where(mask, sign * int(th[1]), out)
        approx_mask |= mask
    if level >= 3:
        mask = (~approx_mask) & (abs_q <= th[3])
        out = torch.where(mask, sign * int(th[2]), out)
        approx_mask |= mask

    return dequantize(out, scales[layer]), int(approx_mask.sum().item()), total


def apply_sap(
    x: torch.Tensor,
    layer: int,
    level: int,
    thresholds: Mapping[int, Sequence[int]],
    scales: Mapping[int, float],
    group_size: int = 6,
    v2: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    total = int(x.numel())
    if level < 0:
        return x, 0, total

    delta = int(thresholds[layer][level])
    q = quantize_int16(x, scales[layer])
    rows, shape = tensor_rows_by_channel(q)
    if rows.shape[1] <= 1:
        return x, 0, total

    out_rows = rows.clone()
    approx_count = 0
    row_len = rows.shape[1]
    padded_len = int(math.ceil(row_len / group_size) * group_size)
    pad = padded_len - row_len
    if pad:
        rows_work = F.pad(rows, (0, pad), value=0)
        valid = torch.zeros(rows_work.shape, dtype=torch.bool)
        valid[:, :row_len] = True
    else:
        rows_work = rows
        valid = torch.ones(rows_work.shape, dtype=torch.bool)

    groups = rows_work.reshape(rows_work.shape[0], -1, group_size)
    valid_groups = valid.reshape(valid.shape[0], -1, group_size)
    out_groups = groups.clone()

    if v2:
        first = groups[:, :, 0]
        all_similar = ((groups - first.unsqueeze(-1)).abs() <= delta) | (~valid_groups)
        all_similar = all_similar.all(dim=2)
        replace_all = all_similar.unsqueeze(-1) & valid_groups
        approx_count += int((replace_all.sum() - all_similar.sum()).item())
        out_groups = torch.where(replace_all, first.unsqueeze(-1).expand_as(out_groups), out_groups)
        partial_groups = ~all_similar
    else:
        partial_groups = torch.ones(groups.shape[:2], dtype=torch.bool)

    anchor = groups[:, :, 0]
    for pos in range(1, group_size):
        valid_pos = valid_groups[:, :, pos] & partial_groups
        cur = groups[:, :, pos]
        same_sign = ((cur >= 0) & (anchor >= 0)) | ((cur < 0) & (anchor < 0))
        similar = valid_pos & same_sign & ((cur - anchor).abs() <= delta)
        out_groups[:, :, pos] = torch.where(similar, anchor, out_groups[:, :, pos])
        approx_count += int(similar.sum().item())
        anchor = torch.where(valid_pos & (~similar), cur, anchor)

    out_flat = out_groups.reshape(rows_work.shape)
    out_rows[:, :] = out_flat[:, :row_len]
    return dequantize(rows_to_tensor(out_rows, shape), scales[layer]), approx_count, total


def apply_abdtr(
    x: torch.Tensor,
    layer: int,
    interval: int,
    scales: Mapping[int, float],
) -> Tuple[torch.Tensor, int, int]:
    total = int(x.numel())
    if interval < 0:
        return x, 0, total

    step = max(1, interval + 1)
    q = quantize_int16(x, scales[layer])
    flat = q.reshape(q.shape[0], -1)
    if flat.shape[1] <= 1:
        return x, 0, total

    out = flat.clone()
    positions = torch.arange(1, flat.shape[1] + 1)
    drop_mask = (positions % step == 0).unsqueeze(0).expand_as(flat)
    if not drop_mask.any():
        return x, 0, total

    left = torch.roll(flat, shifts=1, dims=1)
    right = torch.roll(flat, shifts=-1, dims=1)
    recovered = ((left.to(torch.int64) + right.to(torch.int64)) // 2).to(torch.int32)
    recovered[:, 0] = right[:, 0]
    recovered[:, -1] = left[:, -1]
    out = torch.where(drop_mask, recovered, out)
    return dequantize(out.reshape_as(q), scales[layer]), int(drop_mask.sum().item()), total


def edge_level_from_layer_config(layer_config: Mapping[int, int], src: int, dst: int) -> int:
    if (src, dst) in SHORTCUT_EDGES:
        return -1
    return int(layer_config.get(src, -1))


def edge_interval_from_layer_config(layer_config: Mapping[int, int], src: int, dst: int) -> int:
    if (src, dst) in SHORTCUT_EDGES:
        return -1
    return int(layer_config.get(src, -1))


def make_edge_apply(
    scheme: str,
    layer_config: Mapping[int, int],
    thresholds: Mapping[int, Sequence[int]],
    scales: Mapping[int, float],
):
    stats = {"approx": 0, "total": 0}

    def apply(src: int, dst: int, x: torch.Tensor) -> torch.Tensor:
        if scheme == "exact":
            y, approx, total = x, 0, int(x.numel())
        elif scheme == "fas":
            level = edge_level_from_layer_config(layer_config, src, dst)
            y, approx, total = apply_fas(x, src, level, thresholds, scales)
        elif scheme == "sap-rle":
            level = edge_level_from_layer_config(layer_config, src, dst)
            y, approx, total = apply_sap(x, src, level, thresholds, scales, v2=False)
        elif scheme == "sap-rle-v2":
            level = edge_level_from_layer_config(layer_config, src, dst)
            y, approx, total = apply_sap(x, src, level, thresholds, scales, v2=True)
        elif scheme == "abdtr":
            interval = edge_interval_from_layer_config(layer_config, src, dst)
            y, approx, total = apply_abdtr(x, src, interval, scales)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        if (src, dst) not in SHORTCUT_EDGES:
            stats["approx"] += approx
            stats["total"] += total
        return y

    return apply, stats


def forward_approx(
    model: FusedResNet8CIFAR10,
    x: torch.Tensor,
    scheme: str,
    layer_config: Mapping[int, int],
    thresholds: Mapping[int, Sequence[int]],
    scales: Mapping[int, float],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    edge, stats = make_edge_apply(scheme, layer_config, thresholds, scales)

    l1 = F.relu(model.conv1(x))
    l2 = F.relu(model.block1.conv1(edge(1, 2, l1)))
    l3 = model.block1.conv2(edge(2, 3, l2))
    l4 = F.relu(edge(3, 4, l3) + edge(1, 4, l1))

    l5 = F.relu(model.block2.conv1(edge(4, 5, l4)))
    l6 = model.block2.conv2(edge(5, 6, l5))
    l7 = model.block2.shortcut[0](edge(4, 7, l4))
    l8 = F.relu(edge(6, 8, l6) + edge(7, 8, l7))

    l9 = F.relu(model.block3.conv1(edge(8, 9, l8)))
    l10 = model.block3.conv2(edge(9, 10, l9))
    l11 = model.block3.shortcut[0](edge(8, 11, l8))
    l12 = F.relu(edge(10, 12, l10) + edge(11, 12, l11))

    l13 = model.avgpool(edge(12, 13, l12))
    logits = model.fc(torch.flatten(edge(13, 14, l13), 1))
    return logits, stats


def evaluate(
    model: FusedResNet8CIFAR10,
    loader: BatchList,
    scheme: str,
    layer_config: Mapping[int, int],
    thresholds: Mapping[int, Sequence[int]],
    scales: Mapping[int, float],
) -> EvalResult:
    correct = 0
    total = 0
    approx_values = 0
    total_values = 0
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            logits, stats = forward_approx(model, x, scheme, layer_config, thresholds, scales)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
            approx_values += int(stats["approx"])
            total_values += int(stats["total"])
    return EvalResult(correct / total, correct, total, approx_values, total_values, time.time() - start)


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def result_dict(result: EvalResult) -> Dict[str, float | int]:
    return {
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total,
        "approx_values": result.approx_values,
        "total_values": result.total_values,
        "approx_ratio": result.approx_ratio,
        "seconds": result.seconds,
    }


def search_level_scheme(
    model: FusedResNet8CIFAR10,
    loader: BatchList,
    scheme: str,
    thresholds: Mapping[int, Sequence[int]],
    scales: Mapping[int, float],
    min_accuracy: float,
) -> Dict[str, object]:
    history: List[Dict[str, object]] = []
    config = {layer: -1 for layer in ACTIVE_SOURCE_LAYERS}
    best_global = -1

    for level in range(4):
        trial = {layer: level for layer in ACTIVE_SOURCE_LAYERS}
        result = evaluate(model, loader, scheme, trial, thresholds, scales)
        ok = result.accuracy >= min_accuracy
        history.append(
            {
                "stage": "global",
                "level": level,
                "ok": ok,
                **result_dict(result),
            }
        )
        print(f"[{scheme}] global level {level}: acc={format_pct(result.accuracy)} ok={ok}", flush=True)
        if ok:
            config = trial
            best_global = level
        else:
            break

    for layer in LAYER_SEARCH_ORDER:
        original = config[layer]
        for level in range(original + 1, 4):
            trial = dict(config)
            trial[layer] = level
            result = evaluate(model, loader, scheme, trial, thresholds, scales)
            ok = result.accuracy >= min_accuracy
            history.append(
                {
                    "stage": "layer",
                    "layer": layer,
                    "layer_name": LAYER_NAMES[layer],
                    "level": level,
                    "ok": ok,
                    **result_dict(result),
                }
            )
            print(
                f"[{scheme}] layer {layer:02d} -> level {level}: "
                f"acc={format_pct(result.accuracy)} ok={ok}",
                flush=True,
            )
            if ok:
                config = trial
            else:
                config[layer] = original
                break

    final = evaluate(model, loader, scheme, config, thresholds, scales)
    return {
        "scheme": scheme,
        "best_global_level": best_global,
        "layer_config": config,
        "final": result_dict(final),
        "history": history,
    }


def search_abdtr(
    model: FusedResNet8CIFAR10,
    loader: BatchList,
    scales: Mapping[int, float],
    min_accuracy: float,
    max_interval: int,
) -> Dict[str, object]:
    history: List[Dict[str, object]] = []
    config = {layer: max_interval for layer in ACTIVE_SOURCE_LAYERS}
    best_global = max_interval
    empty_thresholds = {layer: [0, 0, 0, 0] for layer in LAYER_NAMES}

    for interval in range(1, max_interval + 1):
        trial = {layer: interval for layer in ACTIVE_SOURCE_LAYERS}
        result = evaluate(model, loader, "abdtr", trial, empty_thresholds, scales)
        ok = result.accuracy >= min_accuracy
        history.append(
            {
                "stage": "global",
                "interval": interval,
                "ok": ok,
                **result_dict(result),
            }
        )
        print(f"[abdtr] global interval {interval}: acc={format_pct(result.accuracy)} ok={ok}", flush=True)
        if ok:
            config = trial
            best_global = interval
            break

    for layer in LAYER_SEARCH_ORDER:
        original = config[layer]
        for interval in range(original - 1, 0, -1):
            trial = dict(config)
            trial[layer] = interval
            result = evaluate(model, loader, "abdtr", trial, empty_thresholds, scales)
            ok = result.accuracy >= min_accuracy
            history.append(
                {
                    "stage": "layer",
                    "layer": layer,
                    "layer_name": LAYER_NAMES[layer],
                    "interval": interval,
                    "ok": ok,
                    **result_dict(result),
                }
            )
            print(
                f"[abdtr] layer {layer:02d} -> interval {interval}: "
                f"acc={format_pct(result.accuracy)} ok={ok}",
                flush=True,
            )
            if ok:
                config = trial
            else:
                config[layer] = original
                break

    final = evaluate(model, loader, "abdtr", config, empty_thresholds, scales)
    return {
        "scheme": "abdtr",
        "best_global_interval": best_global,
        "layer_config": config,
        "final": result_dict(final),
        "history": history,
    }


def write_noxim_templates(
    out_dir: Path,
    fas_thresholds: Mapping[int, Sequence[int]],
    sap_thresholds: Mapping[int, Sequence[int]],
    results: Mapping[str, Dict[str, object]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_threshold(path: Path, thresholds: Mapping[int, Sequence[int]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for layer in range(1, 15):
                th = thresholds[layer]
                f.write(f"{LAYER_TYPES[layer]} {th[0]} {th[1]} {th[2]} {th[3]}\n")

    def write_level(path: Path, scheme: str) -> None:
        layer_cfg = results[scheme]["layer_config"] if scheme in results else {}
        with path.open("w", encoding="utf-8") as f:
            for layer in range(1, 15):
                level = int(layer_cfg.get(layer, -1)) if isinstance(layer_cfg, dict) else -1
                f.write(f"{LAYER_TYPES[layer]} {level} {level} {level} {level} {level} {level}\n")

    def write_edge(path: Path, scheme: str, tag: str, thresholds: Mapping[int, Sequence[int]]) -> None:
        layer_cfg = results[scheme]["layer_config"] if scheme in results else {}
        with path.open("w", encoding="utf-8") as f:
            f.write(f"% {tag} src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5\n")
            for src, dst in EDGES:
                level = -1
                if (src, dst) not in SHORTCUT_EDGES and isinstance(layer_cfg, dict):
                    level = int(layer_cfg.get(src, -1))
                th = thresholds[src]
                f.write(f"{tag} {src} {dst} 0 {th[0]} {th[1]} {th[2]} {th[3]} {level} {level} {level} {level} {level} {level}\n")

    write_threshold(out_dir / "resnet8_fas_threshold_python.txt", fas_thresholds)
    write_threshold(out_dir / "resnet8_sap_threshold_python.txt", sap_thresholds)
    if "fas" in results:
        write_level(out_dir / "resnet8_fas_level_table_python.txt", "fas")
        write_edge(out_dir / "resnet8_fas_edge_approx_python.txt", "fas", "Edge", fas_thresholds)
    if "sap-rle" in results:
        write_level(out_dir / "resnet8_sap_level_table_python_saprle.txt", "sap-rle")
        write_edge(out_dir / "resnet8_sap_edge_approx_python_saprle.txt", "sap-rle", "SapEdge", sap_thresholds)
    if "sap-rle-v2" in results:
        write_level(out_dir / "resnet8_sap_level_table_python_saprlev2.txt", "sap-rle-v2")
        write_edge(out_dir / "resnet8_sap_edge_approx_python_saprlev2.txt", "sap-rle-v2", "SapEdge", sap_thresholds)
    if "abdtr" in results:
        layer_cfg = results["abdtr"]["layer_config"]
        fallback_interval = max(layer_cfg.values()) if isinstance(layer_cfg, dict) and layer_cfg else 0
        with (out_dir / "resnet8_abdtr_drop_python.txt").open("w", encoding="utf-8") as f:
            for layer in range(1, 15):
                interval = int(layer_cfg.get(layer, fallback_interval)) if isinstance(layer_cfg, dict) else 0
                f.write(f"{max(0, interval)}\n")
        with (out_dir / "resnet8_abdtr_edge_drop_python.txt").open("w", encoding="utf-8") as f:
            f.write("% AbdtrEdge src_layer dst_layer interval\n")
            for src, dst in EDGES:
                interval = -1
                if (src, dst) not in SHORTCUT_EDGES and isinstance(layer_cfg, dict):
                    interval = int(layer_cfg.get(src, fallback_interval))
                f.write(f"AbdtrEdge {src} {dst} {interval}\n")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    baseline: EvalResult,
    min_accuracy: float,
    scales: Mapping[int, float],
    fas_thresholds: Mapping[int, Sequence[int]],
    sap_thresholds: Mapping[int, Sequence[int]],
    results: Mapping[str, Dict[str, object]],
    elapsed: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# ResNet8 CIFAR10 Python Approximate Communication Search Results")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Dataset: CIFAR10 full test set, {baseline.total} images.")
    lines.append(f"- Exact baseline accuracy: {format_pct(baseline.accuracy)} ({baseline.correct}/{baseline.total}).")
    lines.append(f"- Accuracy constraint: approximate accuracy >= {format_pct(min_accuracy)}.")
    lines.append(f"- CPU-only search elapsed time: {elapsed / 60:.2f} min.")
    lines.append("- Shortcut edges are always exact: `1->4`, `4->7`, `7->8`, `8->11`, `11->12`.")
    lines.append("- Python thresholds are activation-q16 simulation thresholds; noxim also quantizes weights, so use these mainly as level/sensitivity guidance.")
    lines.append("")

    if results:
        lines.append("## Final Results")
        lines.append("")
        rows = []
        for scheme, data in results.items():
            final = data["final"]
            rows.append(
                [
                    scheme,
                    format_pct(float(final["accuracy"])),
                    f"{int(final['correct'])}/{int(final['total'])}",
                    format_pct(float(final["approx_ratio"])),
                    f"{float(final['seconds']):.1f}",
                ]
            )
        lines.append(markdown_table(["scheme", "accuracy", "correct", "approx value ratio", "eval seconds"], rows))
        lines.append("")

    lines.append("## Layer Thresholds")
    lines.append("")
    rows = []
    for layer in range(1, 15):
        rows.append(
            [
                layer,
                LAYER_NAMES[layer],
                LAYER_TYPES[layer],
                LAYER_NEURONS[layer],
                f"{scales[layer]:.8g}",
                " ".join(map(str, fas_thresholds[layer])),
                " ".join(map(str, sap_thresholds[layer])),
            ]
        )
    lines.append(markdown_table(["layer", "name", "type", "neurons", "q16 scale", "FAS th0..th3", "SAP delta th0..th3"], rows))
    lines.append("")

    for scheme, data in results.items():
        lines.append(f"## {scheme} Search")
        lines.append("")
        cfg = data["layer_config"]
        cfg_rows = []
        for layer in range(1, 15):
            value = cfg.get(layer, -1) if isinstance(cfg, dict) else -1
            cfg_rows.append([layer, LAYER_NAMES[layer], LAYER_NEURONS[layer], value])
        value_name = "level" if scheme != "abdtr" else "interval"
        lines.append(markdown_table(["layer", "name", "neurons", value_name], cfg_rows))
        lines.append("")
        lines.append("### Search History")
        lines.append("")
        hist_rows = []
        for item in data["history"]:
            label = item["stage"]
            if item["stage"] == "layer":
                label = f"layer {item['layer']} {item['layer_name']}"
            knob = item.get("level", item.get("interval", ""))
            hist_rows.append(
                [
                    label,
                    knob,
                    "yes" if item["ok"] else "no",
                    format_pct(float(item["accuracy"])),
                    format_pct(float(item["approx_ratio"])),
                ]
            )
        lines.append(markdown_table(["stage", value_name, "meets constraint", "accuracy", "approx value ratio"], hist_rows))
        lines.append("")

    lines.append("## noxim Mapping Notes")
    lines.append("")
    lines.append("- FAS templates from this search are written as `resnet8_fas_*_python.txt`.")
    lines.append("- SAP templates from this search are written as `resnet8_sap_*_python_*.txt`.")
    lines.append("- ABDTR templates from this search are written as `resnet8_abdtr_drop_python.txt` and `resnet8_abdtr_edge_drop_python.txt`.")
    lines.append("- Re-check final choices in noxim because Python only simulates activation q16 communication effects.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)

    selected_schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    model = load_model(args.model)
    loader = load_cifar10(args)

    start = time.time()
    print("[setup] collecting q16 scales and exact baseline...", flush=True)
    scales, baseline = collect_scales(model, loader)
    min_accuracy = baseline.accuracy - args.accuracy_drop
    print(f"[baseline] acc={format_pct(baseline.accuracy)} min={format_pct(min_accuracy)}", flush=True)

    print("[setup] collecting per-layer FAS/SAP thresholds...", flush=True)
    fas_thresholds, sap_thresholds = collect_thresholds(model, loader, scales)

    results: Dict[str, Dict[str, object]] = {}
    if "fas" in selected_schemes:
        results["fas"] = search_level_scheme(model, loader, "fas", fas_thresholds, scales, min_accuracy)
    if "sap-rle" in selected_schemes:
        results["sap-rle"] = search_level_scheme(model, loader, "sap-rle", sap_thresholds, scales, min_accuracy)
    if "sap-rle-v2" in selected_schemes:
        results["sap-rle-v2"] = search_level_scheme(model, loader, "sap-rle-v2", sap_thresholds, scales, min_accuracy)
    if "abdtr" in selected_schemes:
        results["abdtr"] = search_abdtr(model, loader, scales, min_accuracy, args.max_abdtr_interval)

    template_dir = args.out_md.parent / "python_search_outputs"
    write_noxim_templates(template_dir, fas_thresholds, sap_thresholds, results)

    elapsed = time.time() - start
    json_data = {
        "baseline": result_dict(baseline),
        "min_accuracy": min_accuracy,
        "scales": scales,
        "fas_thresholds": fas_thresholds,
        "sap_thresholds": sap_thresholds,
        "results": results,
        "elapsed_seconds": elapsed,
    }
    args.out_json.write_text(json.dumps(json_data, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.out_md, baseline, min_accuracy, scales, fas_thresholds, sap_thresholds, results, elapsed)
    print(f"[done] wrote {args.out_md}", flush=True)
    print(f"[done] wrote {args.out_json}", flush=True)
    print(f"[done] wrote templates under {template_dir}", flush=True)


if __name__ == "__main__":
    main()
