#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from microstructure_e1_cae import list_images


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
        return tensor, path


@torch.no_grad()
def spectral_metrics(images, analysis_size, min_spacing, max_spacing):
    gray = images.mean(dim=1, keepdim=True)
    gray = F.interpolate(gray, size=(analysis_size, analysis_size), mode="area")
    gray = gray - F.avg_pool2d(gray, 31, stride=1, padding=15)
    window = torch.hann_window(analysis_size, device=gray.device, dtype=gray.dtype)
    gray = gray[:, 0] * window[None, :, None] * window[None, None, :]
    power = torch.fft.rfft2(gray, norm="ortho").abs().square()

    ky = torch.fft.fftfreq(analysis_size, device=gray.device) * analysis_size
    kx = torch.fft.rfftfreq(analysis_size, device=gray.device) * analysis_size
    radial_index = torch.round(
        torch.sqrt(ky[:, None].square() + kx[None, :].square())
    ).long()
    maximum_index = int(radial_index.max().item())
    minimum_k = max(1, int(math.ceil(512.0 / max_spacing)))
    maximum_k = min(maximum_index, int(math.floor(512.0 / min_spacing)))

    rows = []
    for sample in power:
        sums = torch.zeros(maximum_index + 1, device=sample.device)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, radial_index.flatten(), sample.flatten())
        counts.scatter_add_(
            0,
            radial_index.flatten(),
            torch.ones_like(sample).flatten(),
        )
        radial_power = sums / counts.clamp_min(1)
        k = torch.arange(radial_power.numel(), device=sample.device).float()
        whitened = radial_power * k.square()
        candidate = whitened[minimum_k : maximum_k + 1]
        candidate = F.avg_pool1d(
            candidate[None, None],
            3,
            stride=1,
            padding=1,
        )[0, 0]
        peak_offset = int(candidate.argmax().item())
        peak_k = minimum_k + peak_offset
        spacing = 512.0 / peak_k
        median = candidate.median().clamp_min(1e-12)
        confidence = float((candidate[peak_offset] / median).item())

        local_peak = (
            (candidate[1:-1] > candidate[:-2])
            & (candidate[1:-1] >= candidate[2:])
        )
        local_indices = torch.nonzero(local_peak, as_tuple=False).flatten() + 1
        ordered = local_indices[torch.argsort(candidate[local_indices], descending=True)]
        alternatives = [
            {
                "spacing_px": 512.0 / (minimum_k + int(index)),
                "relative_power": float(
                    (candidate[index] / candidate[peak_offset].clamp_min(1e-12)).item()
                ),
            }
            for index in ordered[:3].tolist()
        ]
        rows.append(
            {
                "spectral_spacing_px": spacing,
                "spectral_peak_k": peak_k,
                "spectral_peak_confidence": confidence,
                "spectral_alternatives": alternatives,
            }
        )
    return rows


def summarize(rows):
    result = {}
    for key in ("spectral_spacing_px", "spectral_peak_confidence"):
        values = torch.tensor([row[key] for row in rows])
        result[key] = {
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "q05": float(torch.quantile(values, 0.05).item()),
            "q95": float(torch.quantile(values, 0.95).item()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--analysis_size", type=int, default=256)
    parser.add_argument("--min_spacing", type=float, default=16.0)
    parser.add_argument("--max_spacing", type=float, default=160.0)
    parser.add_argument("--plain", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Spectral ripple analysis requires CUDA.")
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)
    paths = list_images(args.data_root)
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    rows = []
    for images, batch_paths in loader:
        metrics = spectral_metrics(
            images.to(device, non_blocking=True),
            args.analysis_size,
            args.min_spacing,
            args.max_spacing,
        )
        for values, path in zip(metrics, batch_paths):
            values["path"] = path
            values["process"] = (
                "generated" if args.plain else Path(path).stem.split(".", 1)[0]
            )
            rows.append(values)

    with open(os.path.join(args.out_root, "spectral_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "process",
                "spectral_spacing_px",
                "spectral_peak_k",
                "spectral_peak_confidence",
                "spectral_alternatives",
            ),
        )
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["spectral_alternatives"] = json.dumps(
                serializable["spectral_alternatives"]
            )
            writer.writerow(serializable)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["process"]].append(row)
    report = {
        "method": {
            "analysis_size": args.analysis_size,
            "spacing_range_px": [args.min_spacing, args.max_spacing],
            "radial_power_whitening": "power * spatial_frequency^2",
        },
        "all": summarize(rows),
        "by_process": {
            process: {"count": len(values), **summarize(values)}
            for process, values in sorted(grouped.items())
        },
    }
    with open(os.path.join(args.out_root, "spectral_summary.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
