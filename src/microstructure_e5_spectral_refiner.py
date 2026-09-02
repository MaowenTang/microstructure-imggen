#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from microstructure_e1_cae import build_spatial_split, list_images
from microstructure_e2_diagnostics import (
    DESCRIPTOR_NAMES,
    descriptor_summary,
    morphology_descriptors,
)
from microstructure_e4_spatial_condition import channel_correlation, structure_map


class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous() / 127.5 - 1.0


def process_id(path):
    return Path(path).stem.split(".", 1)[0]


def radial_indices(size, device):
    ky = torch.fft.fftfreq(size, device=device) * size
    kx = torch.fft.rfftfreq(size, device=device) * size
    indices = torch.round(torch.sqrt(ky[:, None].square() + kx[None, :].square()))
    return indices.long()


@torch.no_grad()
def average_radial_power(loader, device):
    indices = radial_indices(512, device)
    bins = int(indices.max().item()) + 1
    total = torch.zeros(bins, device=device, dtype=torch.float64)
    count = torch.zeros(bins, device=device, dtype=torch.float64)
    image_count = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        gray = ((images + 1.0) * 0.5).mean(dim=1)
        gray = gray - gray.mean(dim=(-2, -1), keepdim=True)
        power = torch.fft.rfft2(gray, norm="ortho").abs().square().double()
        for sample in power:
            total.scatter_add_(0, indices.flatten(), sample.flatten())
            count.scatter_add_(
                0,
                indices.flatten(),
                torch.ones_like(sample, dtype=torch.float64).flatten(),
            )
        image_count += images.shape[0]
    return (total / count.clamp_min(1)).float(), image_count


def radial_power(images, indices):
    gray = ((images + 1.0) * 0.5).mean(dim=1)
    gray = gray - gray.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.rfft2(gray, norm="ortho").abs().square()
    bins = int(indices.max().item()) + 1
    rows = []
    for sample in power:
        sums = torch.zeros(bins, device=images.device)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, indices.flatten(), sample.flatten())
        counts.scatter_add_(0, indices.flatten(), torch.ones_like(sample).flatten())
        rows.append(sums / counts.clamp_min(1))
    return torch.stack(rows)


@torch.no_grad()
def refine(images, target_power, cutoff_period, max_gain):
    device = images.device
    indices = radial_indices(images.shape[-1], device)
    current = radial_power(images, indices)
    target = target_power[None].to(device)
    gain = torch.sqrt(target / current.clamp_min(1e-12)).clamp(
        min=1.0 / max_gain,
        max=max_gain,
    )
    gain = F.avg_pool1d(gain[:, None], 9, stride=1, padding=4)[:, 0]
    cutoff_k = images.shape[-1] / cutoff_period
    k = torch.arange(gain.shape[1], device=device).float()
    ramp = torch.sigmoid((k - cutoff_k) / 3.0)
    gain = 1.0 + ramp[None] * (gain - 1.0)
    gain_map = gain[:, indices]

    refined_channels = []
    for channel in range(images.shape[1]):
        fft = torch.fft.rfft2(images[:, channel].float(), norm="ortho")
        refined_channels.append(
            torch.fft.irfft2(
                fft * gain_map,
                s=images.shape[-2:],
                norm="ortho",
            )
        )
    return torch.stack(refined_channels, dim=1).clamp(-1, 1), gain


def save_images(images, out_root):
    os.makedirs(out_root, exist_ok=True)
    for index, image in enumerate(images):
        array = ((image.clamp(-1, 1) + 1.0) * 127.5)
        array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        Image.fromarray(array).save(os.path.join(out_root, f"refined_{index:03d}.png"))


def descriptor_means(descriptors):
    return {
        name: float(descriptors[:, index].mean().item())
        for index, name in enumerate(DESCRIPTOR_NAMES)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--process", default="9")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cutoff_period", type=float, default=24.0)
    parser.add_argument("--max_gain", type=float, default=2.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Spectral refinement requires CUDA.")
    device = torch.device("cuda")
    all_paths = list_images(args.data_root)
    train_paths, _, _, _ = build_spatial_split(all_paths)
    process_paths = [path for path in train_paths if process_id(path) == args.process]
    generated_paths = list_images(args.generated_root)
    if not process_paths or not generated_paths:
        raise RuntimeError("Missing real process images or generated images.")

    target_loader = DataLoader(
        ImageDataset(process_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    generated_loader = DataLoader(
        ImageDataset(generated_paths),
        batch_size=len(generated_paths),
        shuffle=False,
        num_workers=0,
    )
    target_power, target_count = average_radial_power(target_loader, device)
    generated = next(iter(generated_loader)).to(device)
    refined, gain = refine(
        generated,
        target_power,
        args.cutoff_period,
        args.max_gain,
    )
    save_images(refined, args.out_root)

    before_desc = morphology_descriptors(generated).cpu()
    after_desc = morphology_descriptors(refined).cpu()
    real_descriptors = []
    for images in target_loader:
        real_descriptors.append(morphology_descriptors(images.to(device)).cpu())
    real_desc = torch.cat(real_descriptors)

    before_structure = structure_map(generated)
    after_structure = structure_map(refined)
    correlations = channel_correlation(before_structure, after_structure).cpu()
    result = {
        "process": args.process,
        "real_training_images": target_count,
        "generated_images": len(generated_paths),
        "cutoff_period_px": args.cutoff_period,
        "max_gain": args.max_gain,
        "descriptors": {
            "real": descriptor_means(real_desc),
            "before": descriptor_means(before_desc),
            "after": descriptor_means(after_desc),
        },
        "structure_correlation_before_after": {
            name: float(correlations[:, index].mean().item())
            for index, name in enumerate(("low_frequency", "dark_ridge", "arc_edge"))
        },
        "gain": {
            "mean": float(gain.mean().item()),
            "maximum": float(gain.max().item()),
            "fraction_at_maximum": float(
                (gain >= args.max_gain - 1e-6).float().mean().item()
            ),
        },
    }
    with open(os.path.join(args.out_root, "refinement_metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
