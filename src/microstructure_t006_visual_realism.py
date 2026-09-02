#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


FEATURE_GROUPS = (
    "appearance",
    "block_texture",
    "gradient_hist",
    "lbp_texture",
    "spectrum",
    "full",
)


def read_csv(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_image(path):
    image = Image.open(path).convert("RGB")
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.Resampling.BICUBIC)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    return image, rgb, gray


def sobel(gray):
    padded = np.pad(gray, 1, mode="edge")
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
    return np.sqrt(gx**2 + gy**2 + 1e-12)


def hist(values, bins, value_range):
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    counts = counts.astype(np.float32)
    total = counts.sum()
    if total <= 0:
        return counts
    return counts / total


def block_stats(gray, gradient, blocks=8):
    h, w = gray.shape
    gray = gray[: h - h % blocks, : w - w % blocks]
    gradient = gradient[: h - h % blocks, : w - w % blocks]
    bh = gray.shape[0] // blocks
    bw = gray.shape[1] // blocks
    g_blocks = gray.reshape(blocks, bh, blocks, bw)
    e_blocks = gradient.reshape(blocks, bh, blocks, bw)
    means = g_blocks.mean(axis=(1, 3)).ravel()
    stds = g_blocks.std(axis=(1, 3)).ravel()
    edges = e_blocks.mean(axis=(1, 3)).ravel()
    return np.concatenate([means, stds, edges]).astype(np.float32)


def lbp_histogram(gray):
    center = gray[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    neighbors = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    return hist(code.ravel(), bins=256, value_range=(0, 256))


def spectrum_features(gray, bins=32):
    small = Image.fromarray(np.uint8(np.clip(gray * 255, 0, 255))).resize(
        (256, 256), Image.Resampling.BICUBIC
    )
    array = np.asarray(small, dtype=np.float32) / 255.0
    array = array - array.mean()
    window = np.hanning(array.shape[0])
    power = np.abs(np.fft.rfft2(array * window[:, None] * window[None, :])) ** 2
    ky = np.fft.fftfreq(array.shape[0]) * array.shape[0]
    kx = np.fft.rfftfreq(array.shape[1]) * array.shape[1]
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    edges = np.linspace(0, radius.max(), bins + 1)
    values = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (radius >= left) & (radius < right)
        values.append(float(np.log1p(power[mask].mean() if mask.any() else 0.0)))
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values)
    return values / norm if norm > 1e-8 else values


def feature_groups(path):
    _, rgb, gray = read_image(path)
    gradient = sobel(gray)
    appearance = []
    for channel in range(3):
        values = rgb[:, :, channel]
        appearance.extend(
            [
                float(values.mean()),
                float(values.std()),
                float(np.quantile(values, 0.05)),
                float(np.quantile(values, 0.50)),
                float(np.quantile(values, 0.95)),
            ]
        )
    appearance.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float((gray < 0.25).mean()),
            float((gray > 0.75).mean()),
        ]
    )
    groups = {
        "appearance": np.asarray(appearance, dtype=np.float32),
        "block_texture": block_stats(gray, gradient),
        "gradient_hist": hist(gradient.ravel(), bins=32, value_range=(0.0, 2.0)),
        "lbp_texture": lbp_histogram(gray),
        "spectrum": spectrum_features(gray),
    }
    groups["full"] = np.concatenate([groups[name] for name in FEATURE_GROUPS if name != "full"])
    return groups


def choose_real_rows(rows, process, max_count):
    process_rows = [row for row in rows if row["process"] == process]
    process_rows = sorted(process_rows, key=lambda row: (row.get("source", ""), row.get("patch_index", ""), row["path"]))
    if len(process_rows) <= max_count:
        return process_rows
    indices = np.linspace(0, len(process_rows) - 1, max_count, dtype=int)
    return [process_rows[int(index)] for index in indices]


def split_reference_holdout(rows):
    reference = []
    holdout = []
    for index, row in enumerate(rows):
        if index % 5 == 0:
            holdout.append(row)
        else:
            reference.append(row)
    if not holdout and reference:
        holdout = reference[-1:]
        reference = reference[:-1]
    return reference, holdout


def stack_features(records, group):
    return np.stack([record["features"][group] for record in records]).astype(np.float32)


def nearest(query, reference):
    distances = np.sqrt(((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    index = np.argmin(distances, axis=1)
    value = distances[np.arange(distances.shape[0]), index]
    return value, index


def standardize(reference, *arrays):
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std < 1e-6] = 1.0
    return [((array - mean) / std).astype(np.float32) for array in arrays]


def make_nearest_grid(rows, path, max_rows=12):
    pairs = rows[:max_rows]
    cell = 160
    header = 42
    canvas = Image.new("RGB", (2 * cell, len(pairs) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(pairs):
        y = i * (cell + header)
        draw.text((4, y + 6), f"P{row['process']} {row['group']} ratio={float(row['distance_ratio']):.2f}", fill="black")
        gen = Image.open(row["generated_path"]).convert("RGB").resize((cell, cell))
        real = Image.open(row["nearest_real_path"]).convert("RGB").resize((cell, cell))
        canvas.paste(gen, (0, y + header))
        canvas.paste(real, (cell, y + header))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t005_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--max_real_per_process", type=int, default=500)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    real_rows = read_csv(args.t002_root / "patch_descriptors.csv")
    generated_rows = read_csv(args.t005_root / "ldm_selected_samples.csv")

    metrics = []
    nearest_rows = []
    summary_rows = []
    report_processes = {}

    for process in sorted({row["process"] for row in generated_rows}, key=int):
        real_subset = choose_real_rows(real_rows, process, args.max_real_per_process)
        reference_rows, holdout_rows = split_reference_holdout(real_subset)
        generated_subset = [row for row in generated_rows if row["process"] == process]

        reference_records = [
            {"path": row["path"], "features": feature_groups(row["path"])}
            for row in reference_rows
        ]
        holdout_records = [
            {"path": row["path"], "features": feature_groups(row["path"])}
            for row in holdout_rows
        ]
        generated_records = []
        for row in generated_subset:
            generated_records.append(
                {
                    "path": row["selected_path"],
                    "candidate": row["candidate"],
                    "features": feature_groups(row["selected_path"]),
                }
            )

        process_group_ratios = {}
        process_leakage_flags = {}
        for group in FEATURE_GROUPS:
            reference = stack_features(reference_records, group)
            holdout = stack_features(holdout_records, group)
            generated = stack_features(generated_records, group)
            reference_z, holdout_z, generated_z = standardize(reference, reference, holdout, generated)
            holdout_dist, holdout_nn = nearest(holdout_z, reference_z)
            generated_dist, generated_nn = nearest(generated_z, reference_z)

            holdout_median = float(np.median(holdout_dist))
            generated_median = float(np.median(generated_dist))
            ratio = generated_median / max(holdout_median, 1e-8)
            leakage_threshold = 0.35 * holdout_median
            leakage_fraction = float((generated_dist < leakage_threshold).mean())
            supported = bool(0.35 <= ratio <= 2.0 and leakage_fraction <= 0.25)
            process_group_ratios[group] = ratio
            process_leakage_flags[group] = leakage_fraction
            metrics.append(
                {
                    "process": process,
                    "group": group,
                    "real_reference_count": len(reference_records),
                    "real_holdout_count": len(holdout_records),
                    "generated_count": len(generated_records),
                    "real_holdout_nn_median": holdout_median,
                    "generated_nn_median": generated_median,
                    "generated_over_real_ratio": ratio,
                    "leakage_fraction": leakage_fraction,
                    "visual_group_supported": supported,
                }
            )
            for idx, record in enumerate(generated_records):
                nn_record = reference_records[int(generated_nn[idx])]
                nearest_rows.append(
                    {
                        "process": process,
                        "group": group,
                        "candidate": record["candidate"],
                        "generated_path": record["path"],
                        "nearest_real_path": nn_record["path"],
                        "generated_nn_distance": float(generated_dist[idx]),
                        "real_holdout_nn_median": holdout_median,
                        "distance_ratio": float(generated_dist[idx] / max(holdout_median, 1e-8)),
                        "potential_leakage": bool(generated_dist[idx] < leakage_threshold),
                    }
                )

        critical_groups = ("full", "lbp_texture", "spectrum", "block_texture")
        process_supported = all(0.35 <= process_group_ratios[group] <= 2.0 for group in critical_groups)
        process_supported = process_supported and all(
            process_leakage_flags[group] <= 0.25 for group in critical_groups
        )
        report_processes[process] = {
            "visual_realism_supported": process_supported,
            "ratios": process_group_ratios,
            "leakage_fractions": process_leakage_flags,
        }
        summary_rows.append(
            {
                "process": process,
                "visual_realism_supported": process_supported,
                "full_ratio": process_group_ratios["full"],
                "lbp_texture_ratio": process_group_ratios["lbp_texture"],
                "spectrum_ratio": process_group_ratios["spectrum"],
                "block_texture_ratio": process_group_ratios["block_texture"],
                "appearance_ratio": process_group_ratios["appearance"],
                "full_leakage_fraction": process_leakage_flags["full"],
            }
        )

    p9_supported = report_processes.get("9", {}).get("visual_realism_supported", False)
    supported_processes = [
        process for process, values in report_processes.items() if values["visual_realism_supported"]
    ]
    multi_supported = len(supported_processes) >= 3
    gate = {
        "p9_visual_realism_supported": p9_supported,
        "multi_process_visual_realism_supported": multi_supported,
        "supported_processes": supported_processes,
        "minimum_supported_processes": 3,
        "recommendation": (
            "Visual realism channel supports the current P9/multi-process claim."
            if p9_supported and multi_supported
            else "STOP: visual realism channel does not support strengthening the manuscript claim."
        ),
    }
    report = {
        "feature_groups": list(FEATURE_GROUPS),
        "processes": report_processes,
        "gate": gate,
        "notes": [
            "This is a handcrafted visual-realism channel, not a StyleGAN2-ADA discriminator.",
            "Distance ratios compare generated nearest-real distances against real holdout nearest-real distances.",
            "Ratios far above 1 indicate off-manifold visual texture; ratios far below 1 indicate possible memorization/leakage.",
        ],
    }

    write_csv(
        args.out_root / "feature_group_metrics.csv",
        metrics,
        [
            "process",
            "group",
            "real_reference_count",
            "real_holdout_count",
            "generated_count",
            "real_holdout_nn_median",
            "generated_nn_median",
            "generated_over_real_ratio",
            "leakage_fraction",
            "visual_group_supported",
        ],
    )
    write_csv(
        args.out_root / "nearest_neighbors.csv",
        nearest_rows,
        [
            "process",
            "group",
            "candidate",
            "generated_path",
            "nearest_real_path",
            "generated_nn_distance",
            "real_holdout_nn_median",
            "distance_ratio",
            "potential_leakage",
        ],
    )
    write_csv(
        args.out_root / "visual_realism_summary.csv",
        summary_rows,
        [
            "process",
            "visual_realism_supported",
            "full_ratio",
            "lbp_texture_ratio",
            "spectrum_ratio",
            "block_texture_ratio",
            "appearance_ratio",
            "full_leakage_fraction",
        ],
    )
    nearest_for_grid = [row for row in nearest_rows if row["group"] == "full"]
    make_nearest_grid(nearest_for_grid, args.out_root / "visual_nearest_grid.png")
    (args.out_root / "visual_realism_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
