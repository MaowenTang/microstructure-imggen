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

from microstructure_e1_cae import build_spatial_split, list_images
from microstructure_e2_ldm import (
    build_noise_scheduler,
    build_unet,
    generate,
    load_cae,
    save_grid,
    seed_all,
)


DESCRIPTOR_NAMES = (
    "intensity_mean",
    "intensity_std",
    "dark_fraction",
    "bright_fraction",
    "low_frequency_contrast",
    "gradient_mean",
    "orientation_coherence",
    "orientation_cos2",
    "orientation_sin2",
    "spectral_low_fraction",
    "spectral_centroid",
)


class PathDataset(Dataset):
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
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor / 127.5 - 1.0, path


def process_id(path):
    return Path(path).stem.split(".", 1)[0]


def sobel_components(gray):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    return (
        F.conv2d(gray, kernel_x, padding=1),
        F.conv2d(gray, kernel_y, padding=1),
    )


def morphology_descriptors(images):
    gray = ((images + 1.0) * 0.5).mean(dim=1, keepdim=True).clamp(0, 1)
    flat = gray.flatten(1)
    intensity_mean = flat.mean(dim=1)
    intensity_std = flat.std(dim=1)
    dark_fraction = (flat < 0.25).float().mean(dim=1)
    bright_fraction = (flat > 0.75).float().mean(dim=1)
    low_frequency = F.avg_pool2d(gray, kernel_size=31, stride=1, padding=15)
    low_frequency_contrast = low_frequency.flatten(1).std(dim=1)

    gx, gy = sobel_components(gray)
    gradient_mean = torch.sqrt(gx.square() + gy.square() + 1e-8).flatten(1).mean(dim=1)
    jxx = gx.square().flatten(1).mean(dim=1)
    jyy = gy.square().flatten(1).mean(dim=1)
    jxy = (gx * gy).flatten(1).mean(dim=1)
    denominator = jxx + jyy + 1e-8
    orientation_coherence = torch.sqrt((jxx - jyy).square() + 4 * jxy.square()) / denominator
    orientation_cos2 = (jxx - jyy) / denominator
    orientation_sin2 = 2 * jxy / denominator

    centered = gray - gray.mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.rfft2(centered.float(), norm="ortho").abs().square()
    height, width_half = power.shape[-2:]
    frequency_y = torch.fft.fftfreq(height, device=images.device)
    frequency_x = torch.fft.rfftfreq((width_half - 1) * 2, device=images.device)
    radius = torch.sqrt(frequency_y[:, None].square() + frequency_x[None, :].square())
    nonzero = radius > 0
    valid_power = power[:, 0] * nonzero
    total_power = valid_power.sum(dim=(-2, -1)).clamp_min(1e-12)
    low_mask = (radius <= 0.08) & nonzero
    spectral_low_fraction = (valid_power * low_mask).sum(dim=(-2, -1)) / total_power
    spectral_centroid = (valid_power * radius).sum(dim=(-2, -1)) / total_power

    return torch.stack(
        (
            intensity_mean,
            intensity_std,
            dark_fraction,
            bright_fraction,
            low_frequency_contrast,
            gradient_mean,
            orientation_coherence,
            orientation_cos2,
            orientation_sin2,
            spectral_low_fraction,
            spectral_centroid,
        ),
        dim=1,
    )


@torch.no_grad()
def encode_collection(cae, loader, device):
    descriptor_batches = []
    feature_batches = []
    paths = []
    for images, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        descriptor_batches.append(morphology_descriptors(images).cpu())
        latent = cae.encode(images)
        feature = F.adaptive_avg_pool2d(latent, (8, 8)).flatten(1)
        feature_batches.append(feature.float().cpu())
        paths.extend(batch_paths)
    return torch.cat(descriptor_batches), torch.cat(feature_batches), paths


@torch.no_grad()
def encode_generated(cae, images, device, batch_size):
    descriptor_batches = []
    feature_batches = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        descriptor_batches.append(morphology_descriptors(batch).cpu())
        latent = cae.encode(batch)
        feature_batches.append(
            F.adaptive_avg_pool2d(latent, (8, 8)).flatten(1).float().cpu()
        )
    return torch.cat(descriptor_batches), torch.cat(feature_batches)


def nearest_distances(query, reference, batch_size=64):
    distances = []
    reference = reference.cuda()
    for start in range(0, query.shape[0], batch_size):
        batch = query[start : start + batch_size].cuda()
        distances.append(torch.cdist(batch, reference).min(dim=1).values.cpu())
    return torch.cat(distances)


def pairwise_diversity(features):
    if features.shape[0] < 2:
        return 0.0
    distances = torch.pdist(features)
    return float(distances.mean().item())


def summarize_tensor(values):
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std().item()),
        "median": float(values.median().item()),
        "q05": float(torch.quantile(values, 0.05).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
    }


def descriptor_summary(descriptors):
    return {
        name: summarize_tensor(descriptors[:, index])
        for index, name in enumerate(DESCRIPTOR_NAMES)
    }


def process_statistics(descriptors, paths):
    grouped = defaultdict(list)
    for index, path in enumerate(paths):
        grouped[process_id(path)].append(index)
    result = {}
    for process, indices in sorted(grouped.items()):
        values = descriptors[indices]
        result[process] = {
            "count": len(indices),
            "mean": {
                name: float(values[:, column].mean().item())
                for column, name in enumerate(DESCRIPTOR_NAMES)
            },
            "std": {
                name: float(values[:, column].std().item())
                for column, name in enumerate(DESCRIPTOR_NAMES)
            },
        }
    return result


def eta_squared(descriptors, paths):
    labels = [process_id(path) for path in paths]
    grand_mean = descriptors.mean(dim=0)
    total = (descriptors - grand_mean).square().sum(dim=0).clamp_min(1e-12)
    between = torch.zeros_like(total)
    for label in sorted(set(labels)):
        indices = [index for index, value in enumerate(labels) if value == label]
        group = descriptors[indices]
        between += len(indices) * (group.mean(dim=0) - grand_mean).square()
    values = between / total
    return {
        name: float(values[index].item())
        for index, name in enumerate(DESCRIPTOR_NAMES)
    }


def assign_process_centroids(generated, training, training_paths):
    descriptor_mean = training.mean(dim=0)
    descriptor_std = training.std(dim=0).clamp_min(1e-6)
    processes = sorted(set(process_id(path) for path in training_paths))
    centroids = []
    for process in processes:
        indices = [
            index
            for index, path in enumerate(training_paths)
            if process_id(path) == process
        ]
        centroids.append(((training[indices].mean(dim=0) - descriptor_mean) / descriptor_std))
    centroids = torch.stack(centroids)
    standardized = (generated - descriptor_mean) / descriptor_std
    distances = torch.cdist(standardized, centroids)
    nearest = distances.argmin(dim=1)
    assignments = [processes[index] for index in nearest.tolist()]
    counts = {process: assignments.count(process) for process in processes}
    return assignments, counts, distances.min(dim=1).values


def save_descriptor_csv(path, datasets):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "index", *DESCRIPTOR_NAMES))
        for dataset_name, values in datasets.items():
            for index, row in enumerate(values.tolist()):
                writer.writerow((dataset_name, index, *row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cae_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--generation_batch", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--infer_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train_max_x", type=float, default=0.60)
    parser.add_argument("--val_min_x", type=float, default=0.80)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("E2 diagnostics require CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "generated"), exist_ok=True)

    all_paths = list_images(args.data_root)
    train_paths, val_paths, _, _ = build_spatial_split(
        all_paths,
        train_max=args.train_max_x,
        val_min=args.val_min_x,
    )
    train_loader = DataLoader(
        PathDataset(train_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PathDataset(val_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    checkpoint = torch.load(args.ldm_checkpoint, map_location="cpu")
    unet = build_unet(latent_channels).to(device)
    unet.load_state_dict(checkpoint["ema"], strict=True)
    unet.eval()
    latent_mean = checkpoint["latent_mean"].float().to(device)
    latent_std = checkpoint["latent_std"].float().to(device)

    generated_batches = []
    for start in range(0, args.count, args.generation_batch):
        count = min(args.generation_batch, args.count - start)
        generated_batches.append(
            generate(
                cae,
                unet,
                build_noise_scheduler(),
                latent_mean,
                latent_std,
                device,
                count,
                args.infer_steps,
                args.seed + start,
            )
        )
    generated_images = torch.cat(generated_batches)
    save_grid(generated_images, os.path.join(args.out_root, "generated_grid.png"))
    for index, image in enumerate(generated_images):
        array = ((image.clamp(-1, 1) + 1.0) * 127.5)
        array = array.to(torch.uint8).permute(1, 2, 0).numpy()
        Image.fromarray(array).save(
            os.path.join(args.out_root, "generated", f"sample_{index:04d}.png")
        )

    train_desc, train_features, train_paths = encode_collection(cae, train_loader, device)
    val_desc, val_features, val_paths = encode_collection(cae, val_loader, device)
    gen_desc, gen_features = encode_generated(
        cae,
        generated_images,
        device,
        args.batch_size,
    )

    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0).clamp_min(1e-5)
    train_standard = (train_features - feature_mean) / feature_std
    val_standard = (val_features - feature_mean) / feature_std
    gen_standard = (gen_features - feature_mean) / feature_std
    val_nearest = nearest_distances(val_standard, train_standard) / math.sqrt(
        train_standard.shape[1]
    )
    gen_nearest = nearest_distances(gen_standard, train_standard) / math.sqrt(
        train_standard.shape[1]
    )

    assignments, assignment_counts, centroid_distances = assign_process_centroids(
        gen_desc,
        train_desc,
        train_paths,
    )
    train_desc_mean = train_desc.mean(dim=0)
    train_desc_std = train_desc.std(dim=0).clamp_min(1e-6)
    generated_descriptor_z = torch.abs((gen_desc - train_desc_mean) / train_desc_std)

    result = {
        "counts": {
            "train": len(train_paths),
            "validation": len(val_paths),
            "generated": generated_images.shape[0],
        },
        "descriptors": {
            "train": descriptor_summary(train_desc),
            "validation": descriptor_summary(val_desc),
            "generated": descriptor_summary(gen_desc),
        },
        "process_train_statistics": process_statistics(train_desc, train_paths),
        "process_descriptor_eta_squared": eta_squared(train_desc, train_paths),
        "generated_nearest_process_centroid": {
            "counts": assignment_counts,
            "assignments": assignments,
            "distance": summarize_tensor(centroid_distances),
        },
        "cae_feature_space": {
            "validation_to_train_nearest": summarize_tensor(val_nearest),
            "generated_to_train_nearest": summarize_tensor(gen_nearest),
            "nearest_median_ratio_generated_over_validation": float(
                gen_nearest.median().item() / val_nearest.median().item()
            ),
            "validation_pairwise_diversity": pairwise_diversity(val_standard),
            "generated_pairwise_diversity": pairwise_diversity(gen_standard),
        },
        "generated_descriptor_outlier": {
            "mean_abs_z": float(generated_descriptor_z.mean().item()),
            "fraction_abs_z_gt_2": float((generated_descriptor_z > 2).float().mean().item()),
            "fraction_abs_z_gt_3": float((generated_descriptor_z > 3).float().mean().item()),
        },
    }
    with open(os.path.join(args.out_root, "diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    save_descriptor_csv(
        os.path.join(args.out_root, "descriptors.csv"),
        {"train": train_desc, "validation": val_desc, "generated": gen_desc},
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
