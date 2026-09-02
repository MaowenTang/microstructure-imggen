#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from microstructure_e14_descriptor_audit import compute_descriptors


RADIUS_BY_PROCESS = {"4": 360.0, "5": 230.0, "7": 290.0, "9": 190.0}

TARGET_COLUMNS = {
    "spectral_spacing_px": "target_spectral_spacing_px",
    "spectral_peak_confidence": "target_spectral_peak_confidence",
    "gradient_mean": "target_gradient_mean",
    "sobel_top10_mean": "target_sobel_top10_mean",
    "dark_fraction": "target_dark_fraction",
    "low_frequency_contrast": "target_low_frequency_contrast",
}


class DescriptorArgs:
    analysis_size = 256
    min_spacing = 16.0
    max_spacing = 160.0


def read_csv(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def target_for_process(target_row, summary_row):
    target = {name: parse_float(target_row[column]) for name, column in TARGET_COLUMNS.items()}
    target["intensity_mean"] = parse_float(summary_row["intensity_mean_median"])
    target["intensity_std"] = parse_float(summary_row["intensity_std_median"])
    target["bright_fraction"] = parse_float(summary_row["bright_fraction_median"])
    return target


def procedural_image(process, target, seed, contrast, width_scale, low_scale, noise_scale):
    size = 512
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    spacing = max(8.0, float(target["spectral_spacing_px"]))
    radius = RADIUS_BY_PROCESS.get(process, 220.0)
    angle = math.radians(float(rng.uniform(-8.0, 8.0)))
    center_x = 256.0 + float(rng.uniform(-12.0, 12.0))
    center_y = 256.0 - radius + float(rng.uniform(-10.0, 10.0))
    dx = x - center_x
    dy = y - center_y
    rx = math.cos(angle) * dx + math.sin(angle) * dy
    ry = -math.sin(angle) * dx + math.cos(angle) * dy
    aspect = 1.05 + float(rng.uniform(-0.04, 0.04))
    elliptical_radius = np.sqrt((rx / aspect) ** 2 + ry**2 + 1e-8)

    width = max(2.5, width_scale * spacing)
    phase = np.remainder(elliptical_radius - radius + 0.5 * spacing, spacing) - 0.5 * spacing
    bands = np.exp(-0.5 * (phase / width) ** 2)
    radial_window = 1.0 / (1.0 + np.exp(-(elliptical_radius - (radius - 3.7 * spacing)) / (0.25 * spacing)))
    radial_window *= 1.0 / (1.0 + np.exp(-((radius + 3.7 * spacing) - elliptical_radius) / (0.25 * spacing)))
    sector = 1.0 / (1.0 + np.exp(-(ry + 0.15 * radius) / 12.0))
    bands = bands * radial_window * sector

    low_target = max(0.0, float(target["low_frequency_contrast"]))
    low_field = (
        np.sin(2 * math.pi * (x / size) * 0.70 + rng.uniform(0, 2 * math.pi))
        + 0.55 * np.sin(2 * math.pi * (y / size) * 0.55 + rng.uniform(0, 2 * math.pi))
    )
    low_field = low_field / max(float(low_field.std()), 1e-8)
    low_field *= low_scale * low_target

    base = float(target["intensity_mean"])
    gray = base + low_field - contrast * bands
    gray += rng.normal(0.0, noise_scale, size=(size, size)).astype(np.float32)
    gray += base - float(gray.mean())
    gray = np.clip(gray, 0.0, 1.0)
    rgb = np.dstack([gray, gray, gray])
    return Image.fromarray(np.uint8(np.round(rgb * 255.0)))


def descriptor_error(descriptor_row, target):
    weights = {
        "spectral_spacing_px": 1.3,
        "gradient_mean": 1.0,
        "sobel_top10_mean": 1.0,
        "low_frequency_contrast": 1.0,
        "dark_fraction": 0.6,
        "intensity_mean": 0.5,
    }
    errors = {}
    score = 0.0
    weight_total = 0.0
    for name, weight in weights.items():
        observed = float(descriptor_row[name])
        desired = float(target[name])
        scale = max(abs(desired), 0.05)
        relative = abs(observed - desired) / scale
        errors[f"{name}_relative_error"] = relative
        score += weight * relative
        weight_total += weight
    errors["weighted_relative_error"] = score / weight_total
    return errors


def generate_for_process(process, target, generated_root, samples_per_process):
    process_root = generated_root / f"process_{process}"
    process_root.mkdir(parents=True, exist_ok=True)
    candidates = []
    contrast_values = np.linspace(0.04, 0.70, 9)
    width_values = (0.14, 0.20, 0.28, 0.36)
    low_values = (0.7, 1.0, 1.3)
    noise_values = (0.0, 0.012)
    index = 0
    for contrast in contrast_values:
        for width_scale in width_values:
            for low_scale in low_values:
                for noise_scale in noise_values:
                    path = process_root / f"{process}.t004_candidate_{index:04d}.png"
                    image = procedural_image(
                        process,
                        target,
                        seed=1000 * int(process) + index,
                        contrast=float(contrast),
                        width_scale=float(width_scale),
                        low_scale=float(low_scale),
                        noise_scale=float(noise_scale),
                    )
                    image.save(path)
                    descriptors = compute_descriptors(path, DescriptorArgs)
                    errors = descriptor_error(descriptors, target)
                    candidates.append(
                        {
                            "path": str(path),
                            "process": process,
                            "contrast": float(contrast),
                            "width_scale": float(width_scale),
                            "low_scale": float(low_scale),
                            "noise_scale": float(noise_scale),
                            **descriptors,
                            **errors,
                        }
                    )
                    index += 1
    candidates = sorted(candidates, key=lambda row: row["weighted_relative_error"])
    selected = candidates[:samples_per_process]
    selected_root = process_root / "selected"
    selected_root.mkdir(exist_ok=True)
    for rank, row in enumerate(selected):
        original = Path(row["path"])
        selected_path = selected_root / f"{process}.t004_selected_{rank:03d}.png"
        Image.open(original).save(selected_path)
        row["selected_path"] = str(selected_path)
    return selected, candidates


def make_grid(selected_rows, path):
    images = []
    labels = []
    for row in selected_rows:
        images.append(Image.open(row["selected_path"]).convert("RGB").resize((192, 192)))
        labels.append(f"P{row['process']} err={row['weighted_relative_error']:.2f}")
    columns = 3
    rows = int(math.ceil(len(images) / columns))
    header = 34
    canvas = Image.new("RGB", (columns * 192, rows * (192 + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images):
        x = (idx % columns) * 192
        y = (idx // columns) * (192 + header)
        draw.text((x + 5, y + 8), labels[idx], fill="black")
        canvas.paste(image, (x, y + header))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t003_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--samples_per_process", type=int, default=3)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    generated_root = args.out_root / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    mapper_report = json.loads((args.t003_root / "process_descriptor_mapper.json").read_text())
    if not mapper_report["gate"]["observed_regime_descriptor_routing_supported"]:
        raise RuntimeError("T-003 did not support observed-regime descriptor routing.")

    target_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_descriptor_targets.csv")}
    summary_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_summary.csv")}
    all_selected = []
    all_candidates = []
    adherence_rows = []
    for process in sorted(target_rows, key=int):
        target = target_for_process(target_rows[process], summary_rows[process])
        selected, candidates = generate_for_process(process, target, generated_root, args.samples_per_process)
        all_selected.extend(selected)
        all_candidates.extend(candidates)
        best = selected[0]
        adherence_rows.append(
            {
                "process": process,
                "selected_count": len(selected),
                "best_weighted_relative_error": best["weighted_relative_error"],
                "best_spectral_spacing_px": best["spectral_spacing_px"],
                "target_spectral_spacing_px": target["spectral_spacing_px"],
                "best_spectral_spacing_relative_error": best["spectral_spacing_px_relative_error"],
                "best_sobel_top10_mean": best["sobel_top10_mean"],
                "target_sobel_top10_mean": target["sobel_top10_mean"],
                "best_low_frequency_contrast": best["low_frequency_contrast"],
                "target_low_frequency_contrast": target["low_frequency_contrast"],
            }
        )

    candidate_fields = [
        "process",
        "selected_path",
        "path",
        "weighted_relative_error",
        "spectral_spacing_px",
        "spectral_spacing_px_relative_error",
        "gradient_mean",
        "gradient_mean_relative_error",
        "sobel_top10_mean",
        "sobel_top10_mean_relative_error",
        "low_frequency_contrast",
        "low_frequency_contrast_relative_error",
        "dark_fraction",
        "dark_fraction_relative_error",
        "intensity_mean",
        "intensity_mean_relative_error",
        "contrast",
        "width_scale",
        "low_scale",
        "noise_scale",
    ]
    write_csv(args.out_root / "generated_descriptors.csv", all_selected, candidate_fields)
    write_csv(
        args.out_root / "target_adherence.csv",
        adherence_rows,
        [
            "process",
            "selected_count",
            "best_weighted_relative_error",
            "best_spectral_spacing_px",
            "target_spectral_spacing_px",
            "best_spectral_spacing_relative_error",
            "best_sobel_top10_mean",
            "target_sobel_top10_mean",
            "best_low_frequency_contrast",
            "target_low_frequency_contrast",
        ],
    )
    make_grid(all_selected, args.out_root / "process_grid.png")

    median_best_error = float(np.median([row["best_weighted_relative_error"] for row in adherence_rows]))
    spectral_pass_count = sum(
        float(row["best_spectral_spacing_relative_error"]) <= 0.35 for row in adherence_rows
    )
    proxy_supported = median_best_error <= 0.85 and spectral_pass_count >= 3
    report = {
        "n_processes": len(adherence_rows),
        "samples_per_process": args.samples_per_process,
        "candidate_count": len(all_candidates),
        "selected_count": len(all_selected),
        "gate": {
            "descriptor_proxy_generation_supported": proxy_supported,
            "median_best_weighted_relative_error": median_best_error,
            "spectral_spacing_pass_count_at_35pct": spectral_pass_count,
            "required_spectral_spacing_pass_count": 3,
            "recommendation": (
                "Proceed to T-005 LDM pilot under observed-regime scope."
                if proxy_supported
                else "STOP: descriptor proxy generation did not preserve target descriptors well enough."
            ),
        },
        "adherence": adherence_rows,
        "t003_gate": mapper_report["gate"],
        "claim_boundary": "observed-regime descriptor-mediated proxy generation only",
    }
    (args.out_root / "t004_descriptor_generation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
