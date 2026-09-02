#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from microstructure_e14_descriptor_audit import compute_descriptors


CANDIDATES = {
    "4": {
        "e10_full": "runs/e10_p4_full_ldm/last_multiseed_40k",
        "e9_sobel70": "runs/e9_p4_sobel70_ldm/last_multiseed_40k",
    },
    "5": {
        "e10_full": "runs/e10_p5_full_ldm/last_multiseed_40k",
        "e13_rgb_sobel50_w2": "runs/e13_rgb_p5_sobel50_w2_ldm/last_multiseed_40k",
        "e12_gray_sobel50_w2": "runs/e12_gray_p5_sobel50_w2_ldm/last_multiseed_40k",
        "e9_sobel70": "runs/e9_p5_sobel70_ldm/last_multiseed_40k",
    },
    "7": {
        "e10_full": "runs/e10_p7_full_ldm/last_multiseed_40k",
        "e9_sobel70": "runs/e9_p7_sobel70_ldm/last_multiseed_40k",
    },
    "9": {
        "e13_rgb_sobel50_w2": "runs/e13_rgb_p9_sobel50_w2_ldm/last_multiseed_40k",
        "e12_gray_sobel50_w2": "runs/e12_gray_p9_sobel50_w2_ldm/last_multiseed_40k",
        "e13_gray_sobel70_unweighted": "runs/e13_gray_p9_sobel70_unweighted_ldm/last_multiseed_40k",
        "e8_sobel70": "runs/e8_p9_sobel70_ldm/last_multiseed_40k",
    },
}

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


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def target_for_process(target_row, summary_row):
    target = {name: parse_float(target_row[column]) for name, column in TARGET_COLUMNS.items()}
    target["intensity_mean"] = parse_float(summary_row["intensity_mean_median"])
    target["intensity_std"] = parse_float(summary_row["intensity_std_median"])
    target["bright_fraction"] = parse_float(summary_row["bright_fraction_median"])
    return target


def crop_grid(path, process, candidate, out_root, cells_per_grid):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    crops = []
    if width == 512 and height == 512:
        crop_boxes = [(0, 0, 512, 512)]
    else:
        y0 = 30 if height >= 542 else 0
        columns = max(1, min(cells_per_grid, width // 512))
        crop_boxes = [(column * 512, y0, column * 512 + 512, y0 + 512) for column in range(columns)]
    candidate_root = out_root / "crops" / f"process_{process}" / sanitize(candidate)
    candidate_root.mkdir(parents=True, exist_ok=True)
    for index, box in enumerate(crop_boxes):
        crop = image.crop(box)
        out_path = candidate_root / f"{process}.t005_{sanitize(candidate)}_{sanitize(path.stem)}_cell{index:02d}.png"
        crop.save(out_path)
        crops.append(out_path)
    return crops


def descriptor_error(row, target):
    weights = {
        "spectral_spacing_px": 1.3,
        "gradient_mean": 1.0,
        "sobel_top10_mean": 1.0,
        "low_frequency_contrast": 1.0,
        "dark_fraction": 0.6,
        "intensity_mean": 0.5,
    }
    result = {}
    score = 0.0
    total = 0.0
    for name, weight in weights.items():
        observed = float(row[name])
        desired = float(target[name])
        scale = max(abs(desired), 0.05)
        relative = abs(observed - desired) / scale
        result[f"{name}_relative_error"] = relative
        score += weight * relative
        total += weight
    result["weighted_relative_error"] = score / total
    return result


def summarize_candidate(process, candidate, rows, target):
    if not rows:
        return None
    summary = {
        "process": process,
        "candidate": candidate,
        "count": len(rows),
    }
    for name in (
        "weighted_relative_error",
        "spectral_spacing_px_relative_error",
        "gradient_mean_relative_error",
        "sobel_top10_mean_relative_error",
        "low_frequency_contrast_relative_error",
        "dark_fraction_relative_error",
        "intensity_mean_relative_error",
        "spectral_spacing_px",
        "gradient_mean",
        "sobel_top10_mean",
        "low_frequency_contrast",
        "dark_fraction",
        "intensity_mean",
    ):
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        summary[f"{name}_median"] = float(np.median(values))
        summary[f"{name}_min"] = float(np.min(values))
    summary["target_spectral_spacing_px"] = target["spectral_spacing_px"]
    summary["target_sobel_top10_mean"] = target["sobel_top10_mean"]
    summary["target_low_frequency_contrast"] = target["low_frequency_contrast"]
    summary["process_pass"] = (
        summary["weighted_relative_error_median"] <= 1.0
        and summary["spectral_spacing_px_relative_error_median"] <= 0.50
    )
    return summary


def make_grid(selected_rows, path):
    images = []
    labels = []
    for row in selected_rows:
        images.append(Image.open(row["selected_path"]).convert("RGB").resize((180, 180)))
        labels.append(
            f"P{row['process']} {row['candidate']} err={float(row['weighted_relative_error']):.2f}"
        )
    columns = 4
    rows = int(math.ceil(len(images) / columns))
    header = 36
    canvas = Image.new("RGB", (columns * 180, rows * (180 + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(images):
        x = (index % columns) * 180
        y = (index // columns) * (180 + header)
        draw.text((x + 4, y + 8), labels[index][:30], fill="black")
        canvas.paste(image, (x, y + header))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True, type=Path)
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t003_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--max_grids_per_candidate", type=int, default=8)
    parser.add_argument("--cells_per_grid", type=int, default=4)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    selected_root = args.out_root / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)

    t003 = json.loads((args.t003_root / "process_descriptor_mapper.json").read_text())
    if not t003["gate"]["observed_regime_descriptor_routing_supported"]:
        raise RuntimeError("T-003 observed-regime routing gate did not pass.")

    target_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_descriptor_targets.csv")}
    summary_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_summary.csv")}

    descriptor_rows = []
    candidate_summary = []
    selected_rows = []
    for process, candidates in CANDIDATES.items():
        target = target_for_process(target_rows[process], summary_rows[process])
        process_candidate_rows = []
        for candidate, relative_root in candidates.items():
            candidate_root = args.project_root / relative_root
            grid_paths = sorted(candidate_root.glob("*.png"))[: args.max_grids_per_candidate]
            candidate_rows = []
            for grid_path in grid_paths:
                crop_paths = crop_grid(grid_path, process, candidate, args.out_root, args.cells_per_grid)
                for crop_path in crop_paths:
                    descriptors = compute_descriptors(crop_path, DescriptorArgs)
                    errors = descriptor_error(descriptors, target)
                    row = {
                        "process": process,
                        "candidate": candidate,
                        "grid_path": str(grid_path),
                        "crop_path": str(crop_path),
                        **descriptors,
                        **errors,
                    }
                    descriptor_rows.append(row)
                    candidate_rows.append(row)
            summary = summarize_candidate(process, candidate, candidate_rows, target)
            if summary:
                candidate_summary.append(summary)
                process_candidate_rows.extend(candidate_rows)
        best_candidates = [row for row in candidate_summary if row["process"] == process]
        if not best_candidates:
            continue
        best = min(best_candidates, key=lambda row: float(row["weighted_relative_error_median"]))
        best_rows = [row for row in process_candidate_rows if row["candidate"] == best["candidate"]]
        best_rows = sorted(best_rows, key=lambda row: float(row["weighted_relative_error"]))[:4]
        selected_process_root = selected_root / f"process_{process}"
        selected_process_root.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(best_rows):
            selected_path = selected_process_root / f"{process}.t005_selected_{best['candidate']}_{rank:02d}.png"
            shutil.copyfile(row["crop_path"], selected_path)
            selected = {
                **row,
                "selected_path": str(selected_path),
                "best_candidate_weighted_relative_error_median": best["weighted_relative_error_median"],
                "best_candidate_spectral_spacing_relative_error_median": best[
                    "spectral_spacing_px_relative_error_median"
                ],
            }
            selected_rows.append(selected)

    descriptor_fields = [
        "process",
        "candidate",
        "grid_path",
        "crop_path",
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
    ]
    write_csv(args.out_root / "ldm_generated_descriptors.csv", descriptor_rows, descriptor_fields)
    summary_fields = [
        "process",
        "candidate",
        "count",
        "process_pass",
        "weighted_relative_error_median",
        "weighted_relative_error_min",
        "spectral_spacing_px_relative_error_median",
        "spectral_spacing_px_relative_error_min",
        "sobel_top10_mean_relative_error_median",
        "low_frequency_contrast_relative_error_median",
        "target_spectral_spacing_px",
        "spectral_spacing_px_median",
        "target_sobel_top10_mean",
        "sobel_top10_mean_median",
        "target_low_frequency_contrast",
        "low_frequency_contrast_median",
    ]
    write_csv(args.out_root / "ldm_candidate_summary.csv", candidate_summary, summary_fields)
    selected_fields = descriptor_fields + [
        "selected_path",
        "best_candidate_weighted_relative_error_median",
        "best_candidate_spectral_spacing_relative_error_median",
    ]
    write_csv(args.out_root / "ldm_selected_samples.csv", selected_rows, selected_fields)
    make_grid(selected_rows, args.out_root / "ldm_selected_grid.png")

    best_by_process = {}
    for process in sorted(CANDIDATES, key=int):
        rows = [row for row in candidate_summary if row["process"] == process]
        if rows:
            best_by_process[process] = min(rows, key=lambda row: float(row["weighted_relative_error_median"]))

    p9_best = best_by_process.get("9")
    p9_anchor_supported = bool(
        p9_best
        and float(p9_best["weighted_relative_error_median"]) <= 1.0
        and float(p9_best["spectral_spacing_px_relative_error_median"]) <= 0.50
    )
    passed_processes = [
        process for process, row in best_by_process.items() if row.get("process_pass")
    ]
    multi_process_supported = len(passed_processes) >= 3
    report = {
        "candidate_count": len(candidate_summary),
        "descriptor_row_count": len(descriptor_rows),
        "selected_count": len(selected_rows),
        "best_by_process": best_by_process,
        "gate": {
            "p9_ripple_anchor_supported": p9_anchor_supported,
            "multi_process_descriptor_routing_supported": multi_process_supported,
            "passed_processes": passed_processes,
            "minimum_passed_processes_for_multi_process_claim": 3,
            "recommendation": (
                "Proceed with T-005 as the LDM evidence path."
                if p9_anchor_supported
                else "STOP: P9 ripple-bearing anchor is not supported by descriptor validation."
            ),
        },
        "claim_boundary": {
            "allowed": "observed-regime descriptor-routed LDM generation",
            "not_allowed": "arbitrary unseen process-parameter generation",
        },
        "dependencies": {
            "uses_t002_descriptor_targets": True,
            "uses_t003_observed_regime_gate": True,
            "uses_t004_proxy_outputs": False,
        },
    }
    (args.out_root / "t005_ldm_descriptor_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
