#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from microstructure_e1_cae import build_spatial_split, list_images  # noqa: E402


DEFAULT_DATA_ROOT = (
    "/Users/tangmaowen/Library/CloudStorage/GoogleDrive-tmwlxd@gmail.com/"
    "My Drive/Microstructure Image Generation/data/500um_p512_n500"
)


@dataclass(frozen=True)
class Collection:
    name: str
    experiment: str
    target_process: str
    pattern: str
    checkpoint: str = "last"


COLLECTIONS = [
    Collection("E8_P9_RGB_Sobel70", "E8", "9", "artifacts/e8_p9_sobel70/last_multiseed_40k/p9_cfg1.0_seed*_steps250.png"),
    Collection("E9_P4_RGB_Sobel70", "E9", "4", "artifacts/e9_process_experts/p4/last_multiseed_40k/p4_cfg1.0_seed*_steps250.png"),
    Collection("E9_P5_RGB_Sobel70", "E9", "5", "artifacts/e9_process_experts/p5/last_multiseed_40k/p5_cfg1.0_seed*_steps250.png"),
    Collection("E9_P7_RGB_Sobel70", "E9", "7", "artifacts/e9_process_experts/p7/last_multiseed_40k/p7_cfg1.0_seed*_steps250.png"),
    Collection("E10_P4_RGB_Full", "E10", "4", "artifacts/e10_process_full_experts/p4/last_multiseed_40k/p4_cfg1.0_seed*_steps250.png"),
    Collection("E10_P5_RGB_Full", "E10", "5", "artifacts/e10_process_full_experts/p5/last_multiseed_40k/p5_cfg1.0_seed*_steps250.png"),
    Collection("E10_P7_RGB_Full", "E10", "7", "artifacts/e10_process_full_experts/p7/last_multiseed_40k/p7_cfg1.0_seed*_steps250.png"),
    Collection("E11_P5P9_RGB_Sobel70_Shared", "E11", "5+9", "artifacts/e11_p5p9_sobel70/last_multiseed_40k/p5p9_cfg1.0_seed*_steps250.png"),
    Collection("E12_P5_Gray_Sobel50W2_Last", "E12", "5", "artifacts/e12_gray_strict_weighted/p5/last_multiseed_40k/p5_gray_sobel50_w2_cfg1.0_seed*_steps250.png"),
    Collection("E12_P9_Gray_Sobel50W2_Last", "E12", "9", "artifacts/e12_gray_strict_weighted/p9/last_multiseed_40k/p9_gray_sobel50_w2_cfg1.0_seed*_steps250.png"),
    Collection("E12_P5_Gray_Sobel50W2_Best", "E12", "5", "artifacts/e12_gray_strict_weighted/p5/best_multiseed/p5_gray_sobel50_w2_best_cfg1.0_seed*_steps250.png", "best"),
    Collection("E12_P9_Gray_Sobel50W2_Best", "E12", "9", "artifacts/e12_gray_strict_weighted/p9/best_multiseed/p9_gray_sobel50_w2_best_cfg1.0_seed*_steps250.png", "best"),
    Collection("E13_P5_RGB_Sobel50W2_Last", "E13", "5", "artifacts/e13_ablation/rgb_p5_sobel50_w2/last_multiseed_40k/p5_rgb_sobel50_w2_last_cfg1.0_seed*_steps250.png"),
    Collection("E13_P9_RGB_Sobel50W2_Last", "E13", "9", "artifacts/e13_ablation/rgb_p9_sobel50_w2/last_multiseed_40k/p9_rgb_sobel50_w2_last_cfg1.0_seed*_steps250.png"),
    Collection("E13_P5_Gray_Sobel70_Last", "E13", "5", "artifacts/e13_ablation/gray_p5_sobel70_unweighted/last_multiseed_40k/p5_gray_sobel70_unweighted_last_cfg1.0_seed*_steps250.png"),
    Collection("E13_P9_Gray_Sobel70_Last", "E13", "9", "artifacts/e13_ablation/gray_p9_sobel70_unweighted/last_multiseed_40k/p9_gray_sobel70_unweighted_last_cfg1.0_seed*_steps250.png"),
    Collection("E13_P5_RGB_Sobel50W2_Best", "E13", "5", "artifacts/e13_ablation/rgb_p5_sobel50_w2/best_multiseed/p5_rgb_sobel50_w2_best_cfg1.0_seed*_steps250.png", "best"),
    Collection("E13_P9_RGB_Sobel50W2_Best", "E13", "9", "artifacts/e13_ablation/rgb_p9_sobel50_w2/best_multiseed/p9_rgb_sobel50_w2_best_cfg1.0_seed*_steps250.png", "best"),
    Collection("E13_P5_Gray_Sobel70_Best", "E13", "5", "artifacts/e13_ablation/gray_p5_sobel70_unweighted/best_multiseed/p5_gray_sobel70_unweighted_best_cfg1.0_seed*_steps250.png", "best"),
    Collection("E13_P9_Gray_Sobel70_Best", "E13", "9", "artifacts/e13_ablation/gray_p9_sobel70_unweighted/best_multiseed/p9_gray_sobel70_unweighted_best_cfg1.0_seed*_steps250.png", "best"),
]


def process_from_path(path):
    return Path(path).stem.split(".", 1)[0].split("_", 1)[0]


def coarse_process(process):
    return process.split(".", 1)[0]


def load_rgb(path):
    return Image.open(path).convert("RGB")


def split_generated_tiles(path, tile_size=512):
    image = load_rgb(path)
    width, height = image.size
    if width == tile_size and height == tile_size:
        return [(0, image)]
    columns = max(1, width // tile_size)
    row_pitch = tile_size
    header = 0
    if height % tile_size != 0:
        rows = max(1, round(height / (tile_size + 30)))
        header = max(0, height // rows - tile_size)
        row_pitch = tile_size + header
    else:
        rows = height // tile_size
    tiles = []
    index = 0
    for row in range(rows):
        top = row * row_pitch + header
        if top + tile_size > height:
            continue
        for column in range(columns):
            left = column * tile_size
            if left + tile_size > width:
                continue
            tiles.append((index, image.crop((left, top, left + tile_size, top + tile_size))))
            index += 1
    if not tiles:
        raise RuntimeError(f"Could not split generated contact image: {path} size={image.size}")
    return tiles


def image_array(image, size=512):
    if image.size != (size, size):
        image = image.resize((size, size), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def box_blur(gray, kernel=31):
    pad = kernel // 2
    padded = np.pad(gray, pad, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    return (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    ) / float(kernel * kernel)


def sobel(gray):
    padded = np.pad(gray, 1, mode="reflect")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2]
        + 2 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return gx, gy


def spectral_spacing(gray, analysis_size=256, min_spacing=16.0, max_spacing=160.0):
    small = Image.fromarray(np.uint8(np.clip(gray, 0, 1) * 255)).resize(
        (analysis_size, analysis_size), Image.BICUBIC
    )
    signal = np.asarray(small, dtype=np.float32) / 255.0
    signal = signal - box_blur(signal, 31)
    window = np.hanning(analysis_size).astype(np.float32)
    signal = signal * window[:, None] * window[None, :]
    power = np.abs(np.fft.rfft2(signal, norm="ortho")) ** 2
    ky = np.fft.fftfreq(analysis_size) * analysis_size
    kx = np.fft.rfftfreq(analysis_size) * analysis_size
    radial = np.rint(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(np.int32)
    max_index = int(radial.max())
    sums = np.bincount(radial.ravel(), weights=power.ravel(), minlength=max_index + 1)
    counts = np.bincount(radial.ravel(), minlength=max_index + 1).clip(min=1)
    radial_power = sums / counts
    k = np.arange(radial_power.size, dtype=np.float32)
    whitened = radial_power * (k**2)
    minimum_k = max(1, int(math.ceil(512.0 / max_spacing)))
    maximum_k = min(max_index, int(math.floor(512.0 / min_spacing)))
    candidate = whitened[minimum_k : maximum_k + 1]
    if candidate.size < 3:
        return float("nan"), float("nan"), float("nan")
    candidate = np.convolve(candidate, np.ones(3) / 3.0, mode="same")
    peak_offset = int(np.argmax(candidate))
    peak_k = minimum_k + peak_offset
    spacing = 512.0 / peak_k
    confidence = float(candidate[peak_offset] / max(float(np.median(candidate)), 1e-12))
    low_band = radial_power[1 : max(2, minimum_k)].sum()
    target_band = radial_power[minimum_k : maximum_k + 1].sum()
    band_ratio = float(target_band / max(low_band + target_band, 1e-12))
    return float(spacing), confidence, band_ratio


def radial_ripple_score(gray):
    small = Image.fromarray(np.uint8(np.clip(gray, 0, 1) * 255)).resize((128, 128), Image.BICUBIC)
    signal = np.asarray(small, dtype=np.float32) / 255.0
    signal = box_blur(signal, 5)
    gx, gy = sobel(signal)
    mag = np.sqrt(gx * gx + gy * gy + 1e-12)
    threshold = np.quantile(mag, 0.80)
    weights = np.maximum(mag - threshold, 0.0)
    if weights.sum() <= 1e-8:
        return 0.0, float("nan"), float("nan")
    h, w = signal.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x *= 512.0 / w
    y *= 512.0 / h
    nx = gx / np.maximum(mag, 1e-8)
    ny = gy / np.maximum(mag, 1e-8)
    p00 = 1.0 - nx * nx
    p01 = -nx * ny
    p11 = 1.0 - ny * ny
    a00 = float((weights * p00).sum())
    a01 = float((weights * p01).sum())
    a11 = float((weights * p11).sum())
    b0 = float((weights * (p00 * x + p01 * y)).sum())
    b1 = float((weights * (p01 * x + p11 * y)).sum())
    ridge = 1e-4 * max(a00 + a11, 1e-6)
    a00 += ridge
    a11 += ridge
    det = max(a00 * a11 - a01 * a01, 1e-8)
    cx = (a11 * b0 - a01 * b1) / det
    cy = (a00 * b1 - a01 * b0) / det
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-8)
    radial_x = (x - cx) / radius
    radial_y = (y - cy) / radius
    alignment = float((weights * np.abs(nx * radial_x + ny * radial_y)).sum() / weights.sum())

    bin_width = 2.0
    indices = np.floor((radius - radius.min()) / bin_width).astype(np.int32)
    bin_count = int(indices.max()) + 1
    if bin_count < 12:
        return 0.0, float("nan"), alignment
    sums = np.bincount(indices.ravel(), weights=signal.ravel(), minlength=bin_count)
    counts = np.bincount(indices.ravel(), minlength=bin_count).clip(min=1)
    profile = sums / counts
    trend = np.convolve(profile, np.ones(min(31, bin_count)) / min(31, bin_count), mode="same")
    profile = profile - trend
    profile = (profile - profile.mean()) / max(profile.std(), 1e-8)
    minima = np.where((profile[1:-1] < profile[:-2]) & (profile[1:-1] <= profile[2:]) & (profile[1:-1] < -0.15))[0] + 1
    if minima.size < 2:
        return 0.0, float("nan"), alignment
    ordered = minima[np.argsort(profile[minima])]
    selected = []
    min_lag = int(round(8.0 / bin_width))
    max_lag = int(round(80.0 / bin_width))
    for candidate in ordered:
        if all(abs(int(candidate) - prior) >= min_lag for prior in selected):
            selected.append(int(candidate))
    selected = sorted(selected)
    if len(selected) < 2:
        return 0.0, float("nan"), alignment
    diffs = np.diff(selected)
    diffs = diffs[(diffs >= min_lag) & (diffs <= max_lag)]
    if diffs.size == 0:
        return 0.0, float("nan"), alignment
    spacing = float(np.median(diffs) * bin_width)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    regularity = math.exp(-1.4826 * mad / max(float(np.median(diffs)), 1.0))
    contrast = min(1.0, max(0.0, -float(profile[selected].mean()) / 1.5))
    count_factor = min(1.0, diffs.size / 3.0)
    alignment_gate = min(1.0, max(0.0, (alignment - 0.62) / 0.28))
    periodicity = regularity * contrast * count_factor
    periodicity_gate = min(1.0, max(0.0, (periodicity - 0.08) / 0.55))
    band_gate = min(1.0, max(0.0, (len(selected) - 1) / 4.0))
    gradient_gate = min(1.0, float(mag.mean()) / 0.08)
    score = (max(1e-8, alignment_gate) * max(1e-8, periodicity_gate) * max(1e-8, band_gate) * max(1e-8, gradient_gate)) ** 0.25
    return float(score), spacing, alignment


def metrics_for_array(rgb):
    gray = rgb.mean(axis=2)
    flat = gray.ravel()
    gx, gy = sobel(gray)
    grad = np.sqrt(gx * gx + gy * gy + 1e-12)
    jxx = float((gx * gx).mean())
    jyy = float((gy * gy).mean())
    jxy = float((gx * gy).mean())
    orientation_coherence = math.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / max(jxx + jyy, 1e-12)
    q01, q05, q25, q50, q75, q95, q99 = np.quantile(flat, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    block = gray.reshape(16, 32, 16, 32).mean(axis=(1, 3))
    rgb_mean = rgb.reshape(-1, 3).mean(axis=0)
    tint = float(np.std(rgb_mean))
    spectral, spectral_conf, spectral_band = spectral_spacing(gray)
    ripple, radial_spacing, radial_alignment = radial_ripple_score(gray)
    return {
        "intensity_mean": float(flat.mean()),
        "intensity_std": float(flat.std()),
        "intensity_iqr": float(q75 - q25),
        "dark_fraction": float((flat < 0.25).mean()),
        "bright_fraction": float((flat > 0.75).mean()),
        "extreme_fraction": float(((flat < 0.05) | (flat > 0.95)).mean()),
        "block_mean_std": float(block.std()),
        "block_mean_range": float(block.max() - block.min()),
        "rgb_tint_std": tint,
        "edge_mean": float(grad.mean()),
        "edge_p90": float(np.quantile(grad, 0.90)),
        "orientation_coherence": float(orientation_coherence),
        "spectral_spacing_px": spectral,
        "spectral_peak_confidence": spectral_conf,
        "spectral_target_band_fraction": spectral_band,
        "radial_ripple_score": ripple,
        "radial_spacing_px": radial_spacing,
        "radial_alignment": radial_alignment,
        "q01": float(q01),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "q99": float(q99),
    }


def thumbnail_feature(rgb, thumb=32):
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255))
    gray = np.asarray(image.convert("L").resize((thumb, thumb), Image.BICUBIC), dtype=np.float32) / 255.0
    gx, gy = sobel(gray)
    grad = np.sqrt(gx * gx + gy * gy + 1e-12)
    color = np.asarray(image.resize((16, 16), Image.BICUBIC), dtype=np.float32) / 255.0
    color_mean = color.reshape(-1, 3).mean(axis=0)
    color_std = color.reshape(-1, 3).std(axis=0)
    feature = np.concatenate(
        [
            gray.ravel(),
            grad.ravel(),
            np.asarray([gray.mean(), gray.std(), grad.mean(), grad.std()], dtype=np.float32),
            color_mean.astype(np.float32),
            color_std.astype(np.float32),
        ]
    ).astype(np.float32)
    return feature


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q05": float("nan"), "q95": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def nearest(query, reference, batch=64):
    best_dist = []
    best_index = []
    for start in range(0, query.shape[0], batch):
        q = query[start : start + batch]
        distances = np.sqrt(((q[:, None, :] - reference[None, :, :]) ** 2).mean(axis=2))
        idx = np.argmin(distances, axis=1)
        best_dist.extend(distances[np.arange(distances.shape[0]), idx].tolist())
        best_index.extend(idx.tolist())
    return np.asarray(best_dist), np.asarray(best_index)


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def fmt(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_root", default="artifacts/manufacturing_letters_evidence")
    parser.add_argument("--report", default="reports/manufacturing_letters_evidence.md")
    parser.add_argument(
        "--generated_only",
        action="store_true",
        help="Analyze local generated artifacts without reading original patch data.",
    )
    args = parser.parse_args()

    real_rows = []
    train_features = []
    val_features = []
    train_meta = []
    val_meta = []
    train_paths, val_paths, ignored_paths, source_counts = [], [], [], {}
    data_warning = None
    if not args.generated_only:
        data_paths = list_images(args.data_root)
        train_paths, val_paths, ignored_paths, source_counts = build_spatial_split(data_paths)
        print(f"data: train={len(train_paths)} val={len(val_paths)} ignored={len(ignored_paths)}", flush=True)
        try:
            for split, paths, feature_bucket, meta_bucket in (
                ("train", train_paths, train_features, train_meta),
                ("validation", val_paths, val_features, val_meta),
            ):
                for path in paths:
                    rgb = image_array(load_rgb(path))
                    metrics = metrics_for_array(rgb)
                    process = process_from_path(path)
                    row = {
                        "collection": f"Real_{split}_P{coarse_process(process)}",
                        "experiment": "Real",
                        "checkpoint": split,
                        "target_process": coarse_process(process),
                        "source_png": path,
                        "tile_index": 0,
                        "process": process,
                        **metrics,
                    }
                    real_rows.append(row)
                    feature_bucket.append(thumbnail_feature(rgb))
                    meta_bucket.append({"path": path, "process": process, "coarse_process": coarse_process(process)})
        except (OSError, TimeoutError) as exc:
            data_warning = (
                f"Original patch data could not be read locally: {type(exc).__name__}: {exc}. "
                "Run without --generated_only after downloading the Google Drive placeholders, "
                "or run this script on ACES where /scratch data is present."
            )
            print(f"warning: {data_warning}", flush=True)
            real_rows = []
            train_features = []
            val_features = []
            train_meta = []
            val_meta = []

    generated_rows = []
    generated_features = []
    generated_meta = []
    for collection in COLLECTIONS:
        paths = sorted(Path(".").glob(collection.pattern))
        if not paths:
            print(f"warning: no files for {collection.name}: {collection.pattern}", flush=True)
            continue
        for path in paths:
            for tile_index, tile in split_generated_tiles(path):
                rgb = image_array(tile)
                metrics = metrics_for_array(rgb)
                generated_rows.append(
                    {
                        "collection": collection.name,
                        "experiment": collection.experiment,
                        "checkpoint": collection.checkpoint,
                        "target_process": collection.target_process,
                        "source_png": str(path),
                        "tile_index": tile_index,
                        "process": collection.target_process,
                        **metrics,
                    }
                )
                generated_features.append(thumbnail_feature(rgb))
                generated_meta.append(
                    {
                        "collection": collection.name,
                        "experiment": collection.experiment,
                        "checkpoint": collection.checkpoint,
                        "target_process": collection.target_process,
                        "source_png": str(path),
                        "tile_index": tile_index,
                    }
                )

    metric_fields = [
        "collection",
        "experiment",
        "checkpoint",
        "target_process",
        "source_png",
        "tile_index",
        "process",
        "intensity_mean",
        "intensity_std",
        "intensity_iqr",
        "dark_fraction",
        "bright_fraction",
        "extreme_fraction",
        "block_mean_std",
        "block_mean_range",
        "rgb_tint_std",
        "edge_mean",
        "edge_p90",
        "orientation_coherence",
        "spectral_spacing_px",
        "spectral_peak_confidence",
        "spectral_target_band_fraction",
        "radial_ripple_score",
        "radial_spacing_px",
        "radial_alignment",
        "q01",
        "q05",
        "q50",
        "q95",
        "q99",
    ]
    write_csv(os.path.join(args.out_root, "image_metrics.csv"), real_rows + generated_rows, metric_fields)

    generated_features = np.vstack(generated_features).astype(np.float32)

    nearest_rows = []
    validation_median = float("nan")
    if train_features and val_features:
        train_features = np.vstack(train_features).astype(np.float32)
        val_features = np.vstack(val_features).astype(np.float32)
        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True).clip(min=1e-5)
        train_z = (train_features - mean) / std
        val_z = (val_features - mean) / std
        gen_z = (generated_features - mean) / std
        val_dist, val_idx = nearest(val_z, train_z)
        gen_dist, gen_idx = nearest(gen_z, train_z)
        validation_median = float(np.median(val_dist))

        for meta, distance, index in zip(val_meta, val_dist, val_idx):
            nearest_rows.append(
                {
                    "collection": "Real_validation",
                    "target_process": meta["coarse_process"],
                    "query_path": meta["path"],
                    "tile_index": 0,
                    "nearest_train_path": train_meta[int(index)]["path"],
                    "nearest_train_process": train_meta[int(index)]["process"],
                    "thumbnail_distance": float(distance),
                    "distance_over_validation_median": float(distance / validation_median),
                }
            )
        for meta, distance, index in zip(generated_meta, gen_dist, gen_idx):
            nearest_rows.append(
                {
                    "collection": meta["collection"],
                    "target_process": meta["target_process"],
                    "query_path": meta["source_png"],
                    "tile_index": meta["tile_index"],
                    "nearest_train_path": train_meta[int(index)]["path"],
                    "nearest_train_process": train_meta[int(index)]["process"],
                    "thumbnail_distance": float(distance),
                    "distance_over_validation_median": float(distance / validation_median),
                }
            )
    write_csv(
        os.path.join(args.out_root, "nearest_neighbors.csv"),
        nearest_rows,
        fieldnames=[
            "collection",
            "target_process",
            "query_path",
            "tile_index",
            "nearest_train_path",
            "nearest_train_process",
            "thumbnail_distance",
            "distance_over_validation_median",
        ],
    )

    generated_by_collection = {}
    for row in generated_rows:
        generated_by_collection.setdefault(row["collection"], []).append(row)
    real_by_collection = {}
    for row in real_rows:
        real_by_collection.setdefault(row["collection"], []).append(row)

    summary_rows = []
    all_collections = {**real_by_collection, **generated_by_collection}
    for name, rows in sorted(all_collections.items()):
        if not rows:
            continue
        distances = [
            row["thumbnail_distance"]
            for row in nearest_rows
            if row["collection"] == name
        ]
        target = rows[0]["target_process"]
        summary_rows.append(
            {
                "collection": name,
                "experiment": rows[0]["experiment"],
                "checkpoint": rows[0]["checkpoint"],
                "target_process": target,
                "n_tiles": len(rows),
                "ripple_score_median": summarize([row["radial_ripple_score"] for row in rows])["median"],
                "spectral_spacing_median": summarize([row["spectral_spacing_px"] for row in rows])["median"],
                "spectral_confidence_median": summarize([row["spectral_peak_confidence"] for row in rows])["median"],
                "edge_mean_median": summarize([row["edge_mean"] for row in rows])["median"],
                "intensity_std_median": summarize([row["intensity_std"] for row in rows])["median"],
                "block_range_median": summarize([row["block_mean_range"] for row in rows])["median"],
                "tint_median": summarize([row["rgb_tint_std"] for row in rows])["median"],
                "nearest_distance_median": summarize(distances)["median"] if distances else float("nan"),
                "nearest_ratio_vs_val": summarize([d / validation_median for d in distances])["median"] if distances else float("nan"),
            }
        )
    write_csv(os.path.join(args.out_root, "summary_by_collection.csv"), summary_rows)

    nearest_summary = []
    for name in sorted(set(row["collection"] for row in nearest_rows)):
        values = [row["thumbnail_distance"] for row in nearest_rows if row["collection"] == name]
        ratios = [row["distance_over_validation_median"] for row in nearest_rows if row["collection"] == name]
        nearest_summary.append(
            {
                "collection": name,
                "n_queries": len(values),
                "distance_median": summarize(values)["median"],
                "distance_q05": summarize(values)["q05"],
                "distance_q95": summarize(values)["q95"],
                "ratio_vs_validation_median": summarize(ratios)["median"],
            }
        )
    write_csv(os.path.join(args.out_root, "nonmemorization_summary.csv"), nearest_summary)

    json_report = {
        "data_root": args.data_root,
        "counts": {
            "train": len(train_paths),
            "validation": len(val_paths),
            "ignored_buffer": len(ignored_paths),
            "generated_tiles": len(generated_rows),
        },
        "data_warning": data_warning,
        "source_counts": source_counts,
        "feature_diagnostic": {
            "feature": "32x32 grayscale thumbnail + 32x32 Sobel thumbnail + color/intensity summary",
            "validation_nearest_train_median": validation_median,
            "note": "Generated values above 1.0 are farther from train than spatially held-out validation patches under this handcrafted feature.",
        },
    }
    with open(os.path.join(args.out_root, "evidence_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(json_report, handle, indent=2)

    manuscript_rows = [
        row
        for row in summary_rows
        if row["collection"]
        in {
            "Real_validation_P5",
            "Real_validation_P9",
            "E8_P9_RGB_Sobel70",
            "E9_P5_RGB_Sobel70",
            "E10_P5_RGB_Full",
            "E11_P5P9_RGB_Sobel70_Shared",
            "E12_P5_Gray_Sobel50W2_Last",
            "E12_P9_Gray_Sobel50W2_Last",
            "E13_P5_RGB_Sobel50W2_Last",
            "E13_P9_RGB_Sobel50W2_Last",
            "E13_P5_Gray_Sobel70_Last",
            "E13_P9_Gray_Sobel70_Last",
        }
    ]
    manuscript_rows = sorted(manuscript_rows, key=lambda row: (row["experiment"], row["collection"]))
    compact = []
    for row in manuscript_rows:
        compact.append(
            {
                "Collection": row["collection"],
                "n": row["n_tiles"],
                "Ripple": fmt(row["ripple_score_median"], 2),
                "Spec. spacing": fmt(row["spectral_spacing_median"], 1),
                "Spec. conf.": fmt(row["spectral_confidence_median"], 2),
                "Edge": fmt(row["edge_mean_median"], 3),
                "Block range": fmt(row["block_range_median"], 3),
                "NN ratio": fmt(row["nearest_ratio_vs_val"], 2),
            }
        )

    report = []
    report.append("# Manufacturing Letters evidence pack\n")
    report.append("Generated by `src/manufacturing_letters_evidence.py`.\n")
    report.append("## What was added\n")
    report.append("1. CPU morphology/spectral/ripple metrics for E8-E13 final generated tiles.\n")
    report.append("2. Leakage-aware handcrafted nearest-neighbor diagnostic against the spatial training split.\n")
    report.append("3. A four-item Manufacturing Letters figure/table plan.\n")
    report.append("## Data and split\n")
    report.append(f"- Data root: `{args.data_root}`\n")
    report.append(f"- Train/validation/buffer: {len(train_paths)}/{len(val_paths)}/{len(ignored_paths)} patches.\n")
    if data_warning:
        report.append(f"- Warning: {data_warning}\n")
    report.append(f"- Generated tiles analyzed: {len(generated_rows)} after splitting contact strips into 512 x 512 tiles.\n")
    report.append("## Compact manuscript table candidate\n")
    report.append(markdown_table(compact, ["Collection", "n", "Ripple", "Spec. spacing", "Spec. conf.", "Edge", "Block range", "NN ratio"]))
    report.append("\nNotes: `NN ratio` is generated nearest-training distance divided by the real validation nearest-training median. Values above 1.0 are not closer to training patches than held-out real patches under the thumbnail/Sobel feature.\n")
    report.append("## Recommended four figures/tables for Manufacturing Letters\n")
    report.append("1. **Fig. 1 - Data scarcity and protocol:** eight-source dataset, patch overlap, spatial split, deterministic CAE -> latent diffusion pipeline.\n")
    report.append("2. **Fig. 2 - Representation gate:** collapsed VAE vs deterministic CAE/gray CAE reconstruction, with PSNR/SSIM/latent-std callouts.\n")
    report.append("3. **Fig. 3 - Generative evidence panel:** E8 P9, E9 P5, E10 P5 full, E11 P5+P9 shared, E12 P5/P9 gray weighted, E13 ablations.\n")
    report.append("4. **Table 1 - Compact evidence table:** use the metrics table above, possibly reduced to Real P5/P9 + E8/E9/E10/E11/E12/E13 rows.\n")
    report.append("## Files\n")
    report.append(f"- Per-image/tile metrics: `{args.out_root}/image_metrics.csv`\n")
    report.append(f"- Collection summary: `{args.out_root}/summary_by_collection.csv`\n")
    report.append(f"- Nearest-neighbor rows: `{args.out_root}/nearest_neighbors.csv`\n")
    report.append(f"- Non-memorization summary: `{args.out_root}/nonmemorization_summary.csv`\n")
    report.append(f"- Metadata: `{args.out_root}/evidence_metadata.json`\n")
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))

    print(f"wrote {args.report}", flush=True)
    print(f"wrote {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
