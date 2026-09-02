#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microstructure_e1_cae import build_spatial_split, list_images


PROCESS_PARAMETERS = {
    "1": {"laser_power": 487.5, "scan_speed": 6.56, "dwell_time": 35.0},
    "2": {"laser_power": 525.0, "scan_speed": 3.70, "dwell_time": 0.0},
    "3": {"laser_power": 562.5, "scan_speed": 8.46, "dwell_time": 15.0},
    "4": {"laser_power": 600.0, "scan_speed": 1.80, "dwell_time": 30.0},
    "5": {"laser_power": 450.0, "scan_speed": 4.65, "dwell_time": 20.0},
    "6": {"laser_power": 412.5, "scan_speed": 2.75, "dwell_time": 5.0},
    "7": {"laser_power": 375.0, "scan_speed": 5.60, "dwell_time": 40.0},
    "8": {"laser_power": 337.0, "scan_speed": 0.84, "dwell_time": 25.0},
    "9": {"laser_power": 300.0, "scan_speed": 7.40, "dwell_time": 10.0},
}

DESCRIPTOR_NAMES = (
    "intensity_mean",
    "intensity_std",
    "dark_fraction",
    "bright_fraction",
    "low_frequency_contrast",
    "gradient_mean",
    "gradient_p90",
    "sobel_top10_mean",
    "orientation_coherence",
    "block_intensity_range",
    "spectral_spacing_px",
    "spectral_peak_confidence",
)

TARGET_DESCRIPTOR_NAMES = (
    "spectral_spacing_px",
    "spectral_peak_confidence",
    "gradient_mean",
    "sobel_top10_mean",
    "dark_fraction",
    "low_frequency_contrast",
)

PATCH_RE = re.compile(
    r"^(?P<source>(?P<process>\d+)\.[^_]+)_p(?P<patch>\d+)_x(?P<x>\d+)_y(?P<y>\d+)$"
)


def process_parameters(process):
    values = dict(PROCESS_PARAMETERS.get(process, {}))
    if values:
        values["linear_energy"] = values["laser_power"] / values["scan_speed"]
    return values


def parse_patch_path(path):
    match = PATCH_RE.match(Path(path).stem)
    if not match:
        source = Path(path).stem.split("_", 1)[0]
        process = source.split(".", 1)[0]
        return {
            "process": process,
            "source": source,
            "patch_index": "",
            "x": "",
            "y": "",
        }
    return {
        "process": match.group("process"),
        "source": match.group("source"),
        "patch_index": int(match.group("patch")),
        "x": int(match.group("x")),
        "y": int(match.group("y")),
    }


def read_image(path, analysis_size):
    image = Image.open(path).convert("RGB")
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.Resampling.BICUBIC)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    small = image.resize((analysis_size, analysis_size), Image.Resampling.BICUBIC)
    small_gray = np.asarray(small, dtype=np.float32).mean(axis=2) / 255.0
    return rgb, gray, small_gray


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
    return gx, gy


def low_frequency_contrast(gray):
    image = Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255)))
    low = image.resize((32, 32), Image.Resampling.BICUBIC).resize(
        gray.shape[::-1], Image.Resampling.BICUBIC
    )
    low_array = np.asarray(low, dtype=np.float32) / 255.0
    return float(low_array.std())


def block_intensity_range(gray, blocks=8):
    height, width = gray.shape
    cropped = gray[: height - height % blocks, : width - width % blocks]
    block_h = cropped.shape[0] // blocks
    block_w = cropped.shape[1] // blocks
    means = cropped.reshape(blocks, block_h, blocks, block_w).mean(axis=(1, 3))
    return float(means.max() - means.min())


def spectral_spacing(gray, original_size, min_spacing, max_spacing):
    analysis_size = gray.shape[0]
    kernel = np.ones(31, dtype=np.float32) / 31.0
    trend = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, gray)
    trend = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, trend)
    detrended = gray - trend
    window = np.hanning(analysis_size)
    power = np.abs(np.fft.rfft2(detrended * window[:, None] * window[None, :])) ** 2

    ky = np.fft.fftfreq(analysis_size) * analysis_size
    kx = np.fft.rfftfreq(analysis_size) * analysis_size
    radial_index = np.rint(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(int)
    maximum_index = int(radial_index.max())
    sums = np.bincount(radial_index.ravel(), weights=power.ravel(), minlength=maximum_index + 1)
    counts = np.bincount(radial_index.ravel(), minlength=maximum_index + 1)
    radial_power = sums / np.maximum(counts, 1)
    k = np.arange(radial_power.size, dtype=np.float32)
    whitened = radial_power * (k**2)

    minimum_k = max(1, int(math.ceil(original_size / max_spacing)))
    maximum_k = min(maximum_index, int(math.floor(original_size / min_spacing)))
    if maximum_k <= minimum_k:
        return float("nan"), 0, 0.0
    candidate = whitened[minimum_k : maximum_k + 1].astype(np.float32)
    if candidate.size >= 3:
        candidate = np.convolve(candidate, np.array([1 / 3, 1 / 3, 1 / 3]), mode="same")
    peak_offset = int(candidate.argmax())
    peak_k = minimum_k + peak_offset
    spacing = original_size / peak_k
    confidence = float(candidate[peak_offset] / max(float(np.median(candidate)), 1e-12))
    return float(spacing), int(peak_k), confidence


def compute_descriptors(path, args):
    rgb, gray, small_gray = read_image(path, args.analysis_size)
    gx, gy = sobel(gray)
    gradient = np.sqrt(gx**2 + gy**2 + 1e-12)
    top10_threshold = np.quantile(gradient, 0.90)
    jxx = float((gx**2).mean())
    jyy = float((gy**2).mean())
    jxy = float((gx * gy).mean())
    denominator = jxx + jyy + 1e-12
    orientation_coherence = math.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / denominator
    spacing, peak_k, confidence = spectral_spacing(
        small_gray,
        original_size=512.0,
        min_spacing=args.min_spacing,
        max_spacing=args.max_spacing,
    )
    metadata = parse_patch_path(path)
    return {
        "path": str(path),
        **metadata,
        "split": "",
        "spectral_peak_k": peak_k,
        "intensity_mean": float(gray.mean()),
        "intensity_std": float(gray.std()),
        "dark_fraction": float((gray < 0.25).mean()),
        "bright_fraction": float((gray > 0.75).mean()),
        "low_frequency_contrast": low_frequency_contrast(gray),
        "gradient_mean": float(gradient.mean()),
        "gradient_p90": float(np.quantile(gradient, 0.90)),
        "sobel_top10_mean": float(gradient[gradient >= top10_threshold].mean()),
        "orientation_coherence": float(orientation_coherence),
        "block_intensity_range": block_intensity_range(gray),
        "spectral_spacing_px": spacing,
        "spectral_peak_confidence": confidence,
    }


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "std": None, "median": None, "q05": None, "q95": None}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def eta_squared(rows, descriptor):
    values = np.asarray([row[descriptor] for row in rows], dtype=np.float64)
    labels = np.asarray([row["process"] for row in rows])
    valid = np.isfinite(values)
    values = values[valid]
    labels = labels[valid]
    if values.size < 2:
        return 0.0
    grand = values.mean()
    total = np.square(values - grand).sum()
    if total <= 1e-12:
        return 0.0
    between = 0.0
    for label in sorted(set(labels)):
        group = values[labels == label]
        between += group.size * (group.mean() - grand) ** 2
    return float(between / total)


def process_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["process"]].append(row)
    result = {}
    for process, process_rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        sources = sorted({row["source"] for row in process_rows})
        result[process] = {
            "count": len(process_rows),
            "source_count": len(sources),
            "sources": sources,
            "process_parameters": process_parameters(process),
            "descriptors": {
                name: summarize([row[name] for row in process_rows])
                for name in DESCRIPTOR_NAMES
            },
        }
    return result


def source_holdout_accuracy(rows, descriptors):
    sources = sorted({row["source"] for row in rows})
    records = []
    correct = 0
    total = 0
    for source in sources:
        train = [row for row in rows if row["source"] != source]
        test = [row for row in rows if row["source"] == source]
        train_processes = sorted({row["process"] for row in train}, key=int)
        if test[0]["process"] not in train_processes:
            continue
        train_matrix = np.asarray([[row[name] for name in descriptors] for row in train])
        mean = train_matrix.mean(axis=0)
        std = train_matrix.std(axis=0)
        std[std < 1e-8] = 1.0
        centroids = {}
        for process in train_processes:
            matrix = np.asarray(
                [[row[name] for name in descriptors] for row in train if row["process"] == process]
            )
            centroids[process] = ((matrix - mean) / std).mean(axis=0)
        source_correct = 0
        for row in test:
            vector = (np.asarray([row[name] for name in descriptors]) - mean) / std
            predicted = min(
                centroids,
                key=lambda process: float(np.linalg.norm(vector - centroids[process])),
            )
            is_correct = predicted == row["process"]
            source_correct += int(is_correct)
            correct += int(is_correct)
            total += 1
        records.append(
            {
                "source": source,
                "process": test[0]["process"],
                "count": len(test),
                "accuracy": source_correct / len(test),
            }
        )
    return {
        "overall_accuracy": correct / total if total else None,
        "evaluated_count": total,
        "source_records": records,
        "descriptors": list(descriptors),
    }


def centroid_distances(rows, descriptors):
    matrix = np.asarray([[row[name] for name in descriptors] for row in rows], dtype=np.float64)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (matrix - mean) / std
    labels = [row["process"] for row in rows]
    centroids = {}
    for process in sorted(set(labels), key=int):
        centroids[process] = standardized[[label == process for label in labels]].mean(axis=0)
    distances = {}
    processes = sorted(centroids, key=int)
    for left_index, left in enumerate(processes):
        for right in processes[left_index + 1 :]:
            distances[f"{left}-{right}"] = float(
                np.linalg.norm(centroids[left] - centroids[right])
            )
    return distances


def write_patch_csv(path, rows):
    fieldnames = (
        "path",
        "process",
        "source",
        "patch_index",
        "x",
        "y",
        "split",
        *DESCRIPTOR_NAMES,
        "spectral_peak_k",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path, summary):
    fieldnames = (
        "process",
        "count",
        "source_count",
        "sources",
        "laser_power",
        "scan_speed",
        "dwell_time",
        "linear_energy",
    )
    for name in DESCRIPTOR_NAMES:
        fieldnames += (f"{name}_median", f"{name}_q05", f"{name}_q95")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for process, values in summary.items():
            params = values["process_parameters"]
            row = {
                "process": process,
                "count": values["count"],
                "source_count": values["source_count"],
                "sources": ";".join(values["sources"]),
                "laser_power": params.get("laser_power", ""),
                "scan_speed": params.get("scan_speed", ""),
                "dwell_time": params.get("dwell_time", ""),
                "linear_energy": params.get("linear_energy", ""),
            }
            for name in DESCRIPTOR_NAMES:
                descriptor = values["descriptors"][name]
                row[f"{name}_median"] = descriptor["median"]
                row[f"{name}_q05"] = descriptor["q05"]
                row[f"{name}_q95"] = descriptor["q95"]
            writer.writerow(row)


def load_trusted_spectral_anchors(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    anchors = {}
    for process, values in payload.get("by_process", {}).items():
        anchors[process] = {
            "spectral_spacing_px": values["spectral_spacing_px"]["median"],
            "spectral_peak_confidence": values["spectral_peak_confidence"]["median"],
            "count": values.get("count"),
            "source": str(path),
        }
    return anchors


def write_target_csv(path, summary, trusted_spectral):
    fieldnames = (
        "process",
        "laser_power",
        "scan_speed",
        "dwell_time",
        "linear_energy",
        *[f"target_{name}" for name in TARGET_DESCRIPTOR_NAMES],
        "trusted_full_spectral_spacing_px",
        "trusted_full_spectral_peak_confidence",
        "trusted_full_spectral_count",
        "spectral_target_note",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for process, values in summary.items():
            params = values["process_parameters"]
            row = {
                "process": process,
                "laser_power": params.get("laser_power", ""),
                "scan_speed": params.get("scan_speed", ""),
                "dwell_time": params.get("dwell_time", ""),
                "linear_energy": params.get("linear_energy", ""),
            }
            for name in TARGET_DESCRIPTOR_NAMES:
                row[f"target_{name}"] = values["descriptors"][name]["median"]
            trusted = trusted_spectral.get(process)
            if trusted:
                row["trusted_full_spectral_spacing_px"] = trusted["spectral_spacing_px"]
                row["trusted_full_spectral_peak_confidence"] = trusted[
                    "spectral_peak_confidence"
                ]
                row["trusted_full_spectral_count"] = trusted["count"]
                row["spectral_target_note"] = (
                    "Use trusted_full_spectral_* for process-to-descriptor mapping "
                    "when the local patch subset is incomplete."
                )
            else:
                row["trusted_full_spectral_spacing_px"] = ""
                row["trusted_full_spectral_peak_confidence"] = ""
                row["trusted_full_spectral_count"] = ""
                row["spectral_target_note"] = "No trusted full-data spectral anchor found."
            writer.writerow(row)


def draw_summary_figure(path, summary):
    panels = [
        ("spectral_spacing_px", "spacing px"),
        ("spectral_peak_confidence", "spectral confidence"),
        ("gradient_mean", "gradient mean"),
        ("low_frequency_contrast", "low-freq contrast"),
    ]
    processes = sorted(summary, key=int)
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
        label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    draw.text((40, 28), "E14 process-wise morphology descriptor audit", fill=(20, 20, 20), font=title_font)
    colors = {
        "4": (110, 110, 110),
        "5": (45, 110, 180),
        "7": (80, 150, 90),
        "9": (190, 95, 55),
    }
    for index, (descriptor, label) in enumerate(panels):
        x0 = 70 + (index % 2) * 650
        y0 = 110 + (index // 2) * 360
        plot_w, plot_h = 520, 230
        draw.text((x0, y0 - 34), label, fill=(20, 20, 20), font=title_font)
        medians = [summary[p]["descriptors"][descriptor]["median"] for p in processes]
        q05 = [summary[p]["descriptors"][descriptor]["q05"] for p in processes]
        q95 = [summary[p]["descriptors"][descriptor]["q95"] for p in processes]
        finite = [v for trio in zip(medians, q05, q95) for v in trio if v is not None and math.isfinite(v)]
        minimum = min(finite)
        maximum = max(finite)
        if abs(maximum - minimum) < 1e-9:
            maximum = minimum + 1.0
        draw.rectangle((x0, y0, x0 + plot_w, y0 + plot_h), outline=(80, 80, 80), width=2)
        for tick in range(5):
            y = y0 + plot_h - tick * plot_h / 4
            value = minimum + tick * (maximum - minimum) / 4
            draw.line((x0, y, x0 + plot_w, y), fill=(230, 230, 230))
            draw.text((x0 + plot_w + 8, y - 11), f"{value:.2g}", fill=(80, 80, 80), font=label_font)
        bar_w = 72
        gap = 45
        for p_index, process in enumerate(processes):
            x = x0 + 58 + p_index * (bar_w + gap)
            median = medians[p_index]
            low = q05[p_index]
            high = q95[p_index]

            def scale(value):
                return y0 + plot_h - (value - minimum) / (maximum - minimum) * plot_h

            bar_top = scale(median)
            draw.rectangle((x, bar_top, x + bar_w, y0 + plot_h), fill=colors.get(process, (120, 120, 120)))
            draw.line((x + bar_w / 2, scale(low), x + bar_w / 2, scale(high)), fill=(20, 20, 20), width=3)
            draw.text((x + 18, y0 + plot_h + 12), f"P{process}", fill=(20, 20, 20), font=label_font)
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--analysis_size", type=int, default=256)
    parser.add_argument("--min_spacing", type=float, default=16.0)
    parser.add_argument("--max_spacing", type=float, default=160.0)
    parser.add_argument(
        "--trusted_spectral_summary",
        default="artifacts/ripple_spectral/real/spectral_summary.json",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = list_images(args.data_root)
    train_paths, val_paths, ignored_paths, source_split_counts = build_spatial_split(paths)
    split_by_path = {path: "train" for path in train_paths}
    split_by_path.update({path: "validation" for path in val_paths})
    split_by_path.update({path: "ignored_buffer" for path in ignored_paths})

    rows = []
    for index, path in enumerate(paths, start=1):
        row = compute_descriptors(path, args)
        row["split"] = split_by_path.get(path, "")
        rows.append(row)
        if index % 250 == 0:
            print(f"[E14] processed {index}/{len(paths)} patches", flush=True)

    summary = process_summary(rows)
    trusted_spectral = load_trusted_spectral_anchors(args.trusted_spectral_summary)
    eta = {name: eta_squared(rows, name) for name in DESCRIPTOR_NAMES}
    ranked_eta = sorted(eta.items(), key=lambda item: item[1], reverse=True)
    separability = {
        "eta_squared": eta,
        "ranked_descriptors": [
            {"descriptor": name, "eta_squared": value} for name, value in ranked_eta
        ],
        "centroid_distances": centroid_distances(rows, TARGET_DESCRIPTOR_NAMES),
        "source_holdout": source_holdout_accuracy(rows, TARGET_DESCRIPTOR_NAMES),
        "source_split_counts": source_split_counts,
    }
    report = {
        "data_root": str(Path(args.data_root).resolve()),
        "count": len(rows),
        "observed_processes": sorted(summary, key=int),
        "target_descriptors": list(TARGET_DESCRIPTOR_NAMES),
        "process_summary": summary,
        "trusted_spectral_anchors": trusted_spectral,
        "separability": separability,
    }

    write_patch_csv(out_root / "patch_descriptors.csv", rows)
    write_summary_csv(out_root / "process_summary.csv", summary)
    write_target_csv(out_root / "process_descriptor_targets.csv", summary, trusted_spectral)
    with open(out_root / "process_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_root / "separability.json", "w", encoding="utf-8") as handle:
        json.dump(separability, handle, indent=2)
    with open(out_root / "e14_descriptor_audit_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    draw_summary_figure(out_root / "descriptor_process_summary.png", summary)

    print(json.dumps({
        "count": len(rows),
        "observed_processes": sorted(summary, key=int),
        "top_eta_squared": ranked_eta[:5],
        "source_holdout_accuracy": separability["source_holdout"]["overall_accuracy"],
        "out_root": str(out_root),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
