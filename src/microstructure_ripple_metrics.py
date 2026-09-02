#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from microstructure_e1_cae import build_spatial_split, list_images


class ImagePathDataset(Dataset):
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


def gaussian_kernel(sigma, device, dtype):
    radius = int(math.ceil(3 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel /= kernel.sum()
    return kernel


def smooth_gray(gray, sigma):
    kernel = gaussian_kernel(sigma, gray.device, gray.dtype)
    radius = kernel.numel() // 2
    gray = F.conv2d(gray, kernel.view(1, 1, 1, -1), padding=(0, radius))
    return F.conv2d(gray, kernel.view(1, 1, -1, 1), padding=(radius, 0))


def sobel(gray):
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


def weighted_circle_centers(gray, gradient_quantile):
    gx, gy = sobel(gray)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)
    batch, _, height, width = gray.shape
    flattened = magnitude.flatten(1)
    threshold = torch.quantile(flattened, gradient_quantile, dim=1)
    weights = F.relu(magnitude[:, 0] - threshold[:, None, None])
    weights /= weights.flatten(1).mean(dim=1)[:, None, None].clamp_min(1e-8)

    nx = gx[:, 0] / magnitude[:, 0].clamp_min(1e-8)
    ny = gy[:, 0] / magnitude[:, 0].clamp_min(1e-8)
    scale_x = 512.0 / width
    scale_y = 512.0 / height
    y, x = torch.meshgrid(
        torch.arange(height, device=gray.device, dtype=gray.dtype) * scale_y,
        torch.arange(width, device=gray.device, dtype=gray.dtype) * scale_x,
        indexing="ij",
    )

    p00 = 1.0 - nx.square()
    p01 = -nx * ny
    p11 = 1.0 - ny.square()
    a00 = (weights * p00).sum(dim=(-2, -1))
    a01 = (weights * p01).sum(dim=(-2, -1))
    a11 = (weights * p11).sum(dim=(-2, -1))
    b0 = (weights * (p00 * x + p01 * y)).sum(dim=(-2, -1))
    b1 = (weights * (p01 * x + p11 * y)).sum(dim=(-2, -1))

    ridge = 1e-4 * (a00 + a11).clamp_min(1e-6)
    a00 = a00 + ridge
    a11 = a11 + ridge
    determinant = (a00 * a11 - a01.square()).clamp_min(1e-8)
    center_x = (a11 * b0 - a01 * b1) / determinant
    center_y = (a00 * b1 - a01 * b0) / determinant

    trace = a00 + a11
    discriminant = torch.sqrt((a00 - a11).square() + 4 * a01.square())
    eigen_max = 0.5 * (trace + discriminant)
    eigen_min = 0.5 * (trace - discriminant).clamp_min(1e-8)
    condition_number = eigen_max / eigen_min

    dx = x[None] - center_x[:, None, None]
    dy = y[None] - center_y[:, None, None]
    radius = torch.sqrt(dx.square() + dy.square() + 1e-8)
    radial_x = dx / radius
    radial_y = dy / radius
    alignment_map = torch.abs(nx * radial_x + ny * radial_y)
    radial_alignment = (
        (weights * alignment_map).sum(dim=(-2, -1))
        / weights.sum(dim=(-2, -1)).clamp_min(1e-8)
    )
    mean_radius = (
        (weights * radius).sum(dim=(-2, -1))
        / weights.sum(dim=(-2, -1)).clamp_min(1e-8)
    )
    gradient_mean = magnitude.flatten(1).mean(dim=1)

    return {
        "center_x": center_x,
        "center_y": center_y,
        "condition_number": condition_number,
        "radius": radius,
        "mean_radius": mean_radius,
        "weights": weights,
        "radial_alignment": radial_alignment,
        "gradient_mean": gradient_mean,
        "gray": gray[:, 0],
    }


def radial_profile_metrics(gray, radius, weights, bin_width, min_spacing, max_spacing):
    minimum = float(radius.min().item())
    indices = torch.floor((radius - minimum) / bin_width).long()
    bin_count = int(indices.max().item()) + 1
    if bin_count < 12:
        return 0.0, float("nan"), float("nan"), 0, 0

    sums = torch.zeros(bin_count, device=gray.device)
    counts = torch.zeros(bin_count, device=gray.device)
    sums.scatter_add_(0, indices.flatten(), gray.flatten())
    counts.scatter_add_(0, indices.flatten(), torch.ones_like(gray).flatten())
    valid = counts > 0
    profile = sums / counts.clamp_min(1)
    if not valid.all():
        profile[~valid] = profile[valid].mean()

    detrend_kernel = min(31, bin_count if bin_count % 2 == 1 else bin_count - 1)
    detrend_kernel = max(3, detrend_kernel)
    trend = F.avg_pool1d(
        profile[None, None],
        kernel_size=detrend_kernel,
        stride=1,
        padding=detrend_kernel // 2,
    )[0, 0]
    signal = profile - trend
    signal = signal - signal.mean()
    signal = signal / signal.std().clamp_min(1e-8)

    lag_min = max(2, int(round(min_spacing / bin_width)))
    lag_max = min(bin_count // 2, int(round(max_spacing / bin_width)))
    local_minimum = (
        (signal[1:-1] < signal[:-2])
        & (signal[1:-1] <= signal[2:])
        & (signal[1:-1] < -0.15)
    )
    candidates = torch.nonzero(local_minimum, as_tuple=False).flatten() + 1
    if candidates.numel() < 2:
        return 0.0, float("nan"), float("nan"), int(candidates.numel()), bin_count

    # Keep the deepest dark bands while preventing double-edge detections.
    ordered = candidates[torch.argsort(signal[candidates])]
    selected = []
    for candidate in ordered.tolist():
        if all(abs(candidate - existing) >= lag_min for existing in selected):
            selected.append(candidate)
    selected = sorted(selected)
    if len(selected) < 2:
        representative_radius = minimum + selected[0] * bin_width
        return 0.0, float("nan"), representative_radius, 1, bin_count

    selected_tensor = torch.tensor(selected, device=gray.device)
    differences = torch.diff(selected_tensor).float()
    valid_differences = differences[
        (differences >= lag_min) & (differences <= lag_max)
    ]
    if valid_differences.numel() == 0:
        representative_radius = minimum + float(selected_tensor.float().median()) * bin_width
        return 0.0, float("nan"), representative_radius, len(selected), bin_count

    spacing_bins = valid_differences.median()
    mad = (valid_differences - spacing_bins).abs().median()
    regularity = torch.exp(-1.4826 * mad / spacing_bins.clamp_min(1))
    contrast = (-signal[selected_tensor].mean() / 1.5).clamp(0, 1)
    count_factor = min(1.0, valid_differences.numel() / 3.0)
    periodicity = float((regularity * contrast * count_factor).item())
    spacing = float(spacing_bins.item() * bin_width)
    representative_radius = minimum + float(selected_tensor.float().median()) * bin_width
    return periodicity, spacing, representative_radius, len(selected), bin_count


def angular_coverage(radius, center_x, center_y, weights):
    height, width = radius.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=radius.device) * (512.0 / height),
        torch.arange(width, device=radius.device) * (512.0 / width),
        indexing="ij",
    )
    support = weights > torch.quantile(weights, 0.85)
    angles = torch.atan2(y[support] - center_y, x[support] - center_x)
    if angles.numel() < 8:
        return 0.0
    angles = torch.sort(torch.remainder(angles, 2 * math.pi)).values
    gaps = torch.diff(torch.cat((angles, angles[:1] + 2 * math.pi)))
    coverage = 2 * math.pi - float(gaps.max().item())
    return coverage


@torch.no_grad()
def analyze_batch(images, downsample, sigma, gradient_quantile, args):
    gray = images.mean(dim=1, keepdim=True)
    gray = F.interpolate(gray, size=(downsample, downsample), mode="area")
    gray = smooth_gray(gray, sigma)
    fit = weighted_circle_centers(gray, gradient_quantile)
    rows = []
    for index in range(images.shape[0]):
        periodicity, spacing, representative_radius, band_count, radial_bins = (
            radial_profile_metrics(
            fit["gray"][index],
            fit["radius"][index],
            fit["weights"][index],
            args.radial_bin_width,
            args.min_spacing,
            args.max_spacing,
            )
        )
        coverage = angular_coverage(
            fit["radius"][index],
            fit["center_x"][index],
            fit["center_y"][index],
            fit["weights"][index],
        )
        alignment = float(fit["radial_alignment"][index].item())
        gradient = float(fit["gradient_mean"][index].item())
        alignment_gate = min(1.0, max(0.0, (alignment - 0.62) / 0.28))
        periodicity_gate = min(1.0, max(0.0, (periodicity - 0.08) / 0.55))
        band_gate = min(1.0, max(0.0, (band_count - 1) / 4.0))
        gradient_gate = min(1.0, gradient / 0.08)
        ripple_score = (
            max(1e-8, alignment_gate)
            * max(1e-8, periodicity_gate)
            * max(1e-8, band_gate)
            * max(1e-8, gradient_gate)
        ) ** 0.25
        rows.append(
            {
                "ripple_score": ripple_score,
                "radial_alignment": alignment,
                "orientation_error_degrees": math.degrees(
                    math.acos(min(1.0, max(0.0, alignment)))
                ),
                "radial_periodicity": periodicity,
                "spacing_px": spacing,
                "radius_px": representative_radius,
                "ripple_band_count": band_count,
                "arc_coverage_degrees": math.degrees(coverage),
                "center_x": float(fit["center_x"][index].item()),
                "center_y": float(fit["center_y"][index].item()),
                "fit_condition_number": float(
                    fit["condition_number"][index].item()
                ),
                "lowpass_gradient_mean": gradient,
                "radial_bins": radial_bins,
            }
        )
    return rows


def process_id(path):
    return Path(path).stem.split(".", 1)[0]


def summarize(rows):
    numeric_keys = [
        "ripple_score",
        "radial_alignment",
        "radial_periodicity",
        "spacing_px",
        "radius_px",
        "arc_coverage_degrees",
        "orientation_error_degrees",
    ]
    result = {}
    for key in numeric_keys:
        values = torch.tensor(
            [row[key] for row in rows if math.isfinite(row[key])],
            dtype=torch.float32,
        )
        result[key] = {
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "q05": float(torch.quantile(values, 0.05).item()),
            "q95": float(torch.quantile(values, 0.95).item()),
        }
    return result


def save_montage(rows, path, count, reverse=True):
    selected = sorted(rows, key=lambda row: row["ripple_score"], reverse=reverse)[:count]
    columns = min(4, count)
    rows_count = int(math.ceil(count / columns))
    tile = 256
    header = 44
    canvas = Image.new("RGB", (columns * tile, rows_count * (tile + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        y_index, x_index = divmod(index, columns)
        image = Image.open(row["path"]).convert("RGB").resize((tile, tile))
        x = x_index * tile
        y = y_index * (tile + header)
        canvas.paste(image, (x, y + header))
        text = (
            f"s={row['ripple_score']:.2f} d={row['spacing_px']:.0f}px "
            f"R={row['radius_px']:.0f}px\n"
            f"a={row['radial_alignment']:.2f} p={row['radial_periodicity']:.2f} "
            f"cov={row['arc_coverage_degrees']:.0f}"
        )
        draw.text((x + 3, y + 3), text, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--downsample", type=int, default=128)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--gradient_quantile", type=float, default=0.80)
    parser.add_argument("--radial_bin_width", type=float, default=2.0)
    parser.add_argument("--min_spacing", type=float, default=8.0)
    parser.add_argument("--max_spacing", type=float, default=80.0)
    parser.add_argument("--montage_count", type=int, default=24)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Treat every image as one ungrouped collection (for generated samples).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Ripple analysis requires CUDA.")
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)
    paths = list_images(args.data_root)
    if args.plain:
        train, validation, ignored = paths, [], []
        source_counts = {}
        split_by_path = {path: "generated" for path in paths}
    else:
        train, validation, ignored, source_counts = build_spatial_split(paths)
        split_by_path = {path: "train" for path in train}
        split_by_path.update({path: "validation" for path in validation})
        split_by_path.update({path: "buffer" for path in ignored})

    loader = DataLoader(
        ImagePathDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    rows = []
    for images, batch_paths in loader:
        batch_rows = analyze_batch(
            images.to(device, non_blocking=True),
            args.downsample,
            args.sigma,
            args.gradient_quantile,
            args,
        )
        for row, path in zip(batch_rows, batch_paths):
            row["path"] = path
            row["split"] = split_by_path[path]
            row["process"] = "generated" if args.plain else process_id(path)
            rows.append(row)

    fieldnames = list(rows[0])
    with open(os.path.join(args.out_root, "ripple_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["process"]].append(row)
    report = {
        "method": {
            "downsample": args.downsample,
            "sigma": args.sigma,
            "gradient_quantile": args.gradient_quantile,
            "radial_bin_width": args.radial_bin_width,
            "spacing_range": [args.min_spacing, args.max_spacing],
            "note": "Provisional geometry detector; validate top-ranked montage before use.",
        },
        "counts": {
            "all": len(rows),
            "train": len(train),
            "validation": len(validation),
            "buffer": len(ignored),
        },
        "all": summarize(rows),
        "by_process": {
            process: {"count": len(values), **summarize(values)}
            for process, values in sorted(grouped.items())
        },
        "source_counts": source_counts,
    }
    with open(os.path.join(args.out_root, "ripple_summary.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    save_montage(
        rows,
        os.path.join(args.out_root, "ripple_top.png"),
        args.montage_count,
        reverse=True,
    )
    save_montage(
        rows,
        os.path.join(args.out_root, "ripple_bottom.png"),
        args.montage_count,
        reverse=False,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
