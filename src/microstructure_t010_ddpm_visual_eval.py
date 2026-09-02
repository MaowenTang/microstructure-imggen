#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from microstructure_t006_visual_realism import (
    FEATURE_GROUPS,
    choose_real_rows,
    feature_groups,
    nearest,
    read_csv,
    split_reference_holdout,
    stack_features,
    standardize,
    write_csv,
)


CRITICAL_GROUPS = ("full", "lbp_texture", "spectrum", "block_texture")


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def crop_grid(path, method, out_root, cell_size=512):
    image = Image.open(path).convert("RGB")
    columns = image.width // cell_size
    rows = image.height // cell_size
    if columns <= 0 or rows <= 0:
        raise RuntimeError(f"Grid is smaller than one {cell_size} cell: {path} size={image.size}")
    sample_root = out_root / "samples" / method
    sample_root.mkdir(parents=True, exist_ok=True)
    records = []
    index = 0
    for row in range(rows):
        for column in range(columns):
            crop = image.crop(
                (
                    column * cell_size,
                    row * cell_size,
                    (column + 1) * cell_size,
                    (row + 1) * cell_size,
                )
            )
            out_path = sample_root / f"{method}_sample_{index:02d}.png"
            crop.save(out_path)
            records.append(
                {
                    "method": method,
                    "sample_index": index,
                    "path": str(out_path),
                }
            )
            index += 1
    return records


def build_pooled_real_records(real_rows, max_real_per_process):
    reference_rows = []
    holdout_rows = []
    for process in sorted({row["process"] for row in real_rows}, key=int):
        subset = choose_real_rows(real_rows, process, max_real_per_process)
        reference, holdout = split_reference_holdout(subset)
        reference_rows.extend(reference)
        holdout_rows.extend(holdout)
    reference_records = [{"path": row["path"], "features": feature_groups(row["path"])} for row in reference_rows]
    holdout_records = [{"path": row["path"], "features": feature_groups(row["path"])} for row in holdout_rows]
    return reference_records, holdout_records


def score_samples(sample_rows, reference_records, holdout_records):
    sample_records = [
        {"path": row["path"], "features": feature_groups(row["path"]), "row": row}
        for row in sample_rows
    ]
    for group in FEATURE_GROUPS:
        reference = stack_features(reference_records, group)
        holdout = stack_features(holdout_records, group)
        generated = np.stack([record["features"][group] for record in sample_records]).astype(np.float32)
        reference_z, holdout_z, generated_z = standardize(reference, reference, holdout, generated)
        holdout_dist, _ = nearest(holdout_z, reference_z)
        generated_dist, generated_nn = nearest(generated_z, reference_z)
        holdout_median = float(np.median(holdout_dist))
        leakage_threshold = 0.35 * holdout_median
        for index, record in enumerate(sample_records):
            row = record["row"]
            distance = float(generated_dist[index])
            row[f"{group}_nn_distance"] = distance
            row[f"{group}_ratio"] = float(distance / max(holdout_median, 1e-8))
            row[f"{group}_potential_leakage"] = bool(distance < leakage_threshold)
            row[f"{group}_real_holdout_nn_median"] = holdout_median
            row[f"nearest_real_path_{group}"] = reference_records[int(generated_nn[index])]["path"]
    for row in sample_rows:
        row["critical_leakage"] = bool(any(row[f"{group}_potential_leakage"] for group in CRITICAL_GROUPS))
    return sample_rows


def median(rows, key):
    values = [parse_float(row[key]) for row in rows if key in row]
    values = [value for value in values if math.isfinite(value)]
    return float(np.median(values)) if values else float("nan")


def method_summary(method, rows, baseline_full):
    full = median(rows, "full_ratio")
    lbp = median(rows, "lbp_texture_ratio")
    spectrum = median(rows, "spectrum_ratio")
    block = median(rows, "block_texture_ratio")
    appearance = median(rows, "appearance_ratio")
    leakage = float(np.mean([bool(row["critical_leakage"]) for row in rows]))
    improvement = (baseline_full - full) / baseline_full if baseline_full > 0 else 0.0
    return {
        "method": method,
        "sample_count": len(rows),
        "full_ratio_median": full,
        "lbp_texture_ratio_median": lbp,
        "spectrum_ratio_median": spectrum,
        "block_texture_ratio_median": block,
        "appearance_ratio_median": appearance,
        "leakage_fraction": leakage,
        "baseline_t006_full_ratio_median": baseline_full,
        "full_ratio_improvement_fraction": improvement,
        "relative_visual_improved": bool(improvement >= 0.20 and leakage <= 0.25),
        "absolute_visual_realism_supported": bool(
            full <= 2.0 and lbp <= 2.0 and spectrum <= 2.0 and block <= 2.0 and leakage <= 0.25
        ),
    }


def make_samples_grid(rows, path):
    columns = 6
    cell = 150
    header = 42
    canvas_rows = int(math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * cell, canvas_rows * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        x = (index % columns) * cell
        y = (index // columns) * (cell + header)
        label = f"{row['method']} f={parse_float(row['full_ratio']):.1f}"
        draw.text((x + 4, y + 7), label[:24], fill="black")
        image = Image.open(row["path"]).convert("RGB").resize((cell, cell))
        canvas.paste(image, (x, y + header))
    canvas.save(path)


def make_nearest_grid(rows, path, max_rows=18):
    rows = sorted(rows, key=lambda row: parse_float(row["full_ratio"]))[:max_rows]
    cell = 160
    header = 48
    canvas = Image.new("RGB", (2 * cell, len(rows) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        y = index * (cell + header)
        label = (
            f"{row['method']} full={parse_float(row['full_ratio']):.1f} "
            f"lbp={parse_float(row['lbp_texture_ratio']):.1f}"
        )
        draw.text((4, y + 6), label[:45], fill="black")
        sample = Image.open(row["path"]).convert("RGB").resize((cell, cell))
        real = Image.open(row["nearest_real_path_full"]).convert("RGB").resize((cell, cell))
        canvas.paste(sample, (0, y + header))
        canvas.paste(real, (cell, y + header))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t006_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--ddpm_1ch_grid", required=True, type=Path)
    parser.add_argument("--ddpm_rgb_grid", required=True, type=Path)
    parser.add_argument("--max_real_per_process", type=int, default=500)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    real_rows = read_csv(args.t002_root / "patch_descriptors.csv")
    baseline_rows = read_csv(args.t006_root / "visual_realism_summary.csv")
    baseline_full = float(np.median([parse_float(row["full_ratio"]) for row in baseline_rows]))

    sample_rows = []
    sample_rows.extend(crop_grid(args.ddpm_1ch_grid, "ddpm_1ch_40k", args.out_root))
    sample_rows.extend(crop_grid(args.ddpm_rgb_grid, "ddpm_rgb_20k", args.out_root))
    reference_records, holdout_records = build_pooled_real_records(real_rows, args.max_real_per_process)
    scored_rows = score_samples(sample_rows, reference_records, holdout_records)

    summaries = []
    report_methods = {}
    for method in sorted({row["method"] for row in scored_rows}):
        rows = [row for row in scored_rows if row["method"] == method]
        summary = method_summary(method, rows, baseline_full)
        summaries.append(summary)
        report_methods[method] = summary

    improved = [row["method"] for row in summaries if row["relative_visual_improved"]]
    absolute = [row["method"] for row in summaries if row["absolute_visual_realism_supported"]]
    gate = {
        "pooled_baseline_t006_full_ratio_median": baseline_full,
        "relative_visual_improved_methods": improved,
        "absolute_visual_supported_methods": absolute,
        "any_relative_visual_improvement_supported": bool(improved),
        "any_absolute_visual_realism_supported": bool(absolute),
        "recommendation": (
            "Existing DDPM samples improve visual realism; next branch should add process/descriptor conditioning."
            if improved and not absolute
            else (
                "Existing DDPM samples support pooled visual realism; next branch should add process/descriptor conditioning."
                if absolute
                else "Existing DDPM samples do not improve visual realism; prioritize normalization and stronger training."
            )
        ),
    }
    report = {
        "method": "Evaluate existing unconditional DDPM sample grids with pooled real-patch visual calibration.",
        "inputs": {
            "ddpm_1ch_grid": str(args.ddpm_1ch_grid),
            "ddpm_rgb_grid": str(args.ddpm_rgb_grid),
        },
        "methods": report_methods,
        "gate": gate,
        "notes": [
            "Existing DDPM samples are unconditional, so this task does not claim process or descriptor control.",
            "Pooled visual calibration uses real patches from all observed processes.",
            "Absolute visual support requires full, LBP, spectrum, and block texture ratios <= 2.0.",
        ],
    }

    score_fields = [
        "method",
        "sample_index",
        "path",
        "appearance_ratio",
        "block_texture_ratio",
        "gradient_hist_ratio",
        "lbp_texture_ratio",
        "spectrum_ratio",
        "full_ratio",
        "critical_leakage",
        "nearest_real_path_full",
    ]
    summary_fields = [
        "method",
        "sample_count",
        "full_ratio_median",
        "lbp_texture_ratio_median",
        "spectrum_ratio_median",
        "block_texture_ratio_median",
        "appearance_ratio_median",
        "leakage_fraction",
        "baseline_t006_full_ratio_median",
        "full_ratio_improvement_fraction",
        "relative_visual_improved",
        "absolute_visual_realism_supported",
    ]
    write_csv(args.out_root / "ddpm_sample_scores.csv", scored_rows, score_fields)
    write_csv(args.out_root / "ddpm_method_summary.csv", summaries, summary_fields)
    make_samples_grid(scored_rows, args.out_root / "t010_ddpm_samples_grid.png")
    make_nearest_grid(scored_rows, args.out_root / "t010_ddpm_nearest_grid.png")
    (args.out_root / "t010_ddpm_visual_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
