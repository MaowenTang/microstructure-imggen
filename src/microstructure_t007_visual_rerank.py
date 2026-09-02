#!/usr/bin/env python3

import argparse
import csv
import json
import math
import shutil
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
VISUAL_WEIGHTS = {
    "full": 0.30,
    "lbp_texture": 0.30,
    "spectrum": 0.25,
    "appearance": 0.10,
    "block_texture": 0.05,
}


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def percentile_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    if len(values) <= 1:
        ranks[:] = 0.0
        return ranks
    ranks[order] = np.arange(len(values), dtype=float) / float(len(values) - 1)
    return ranks


def median(rows, key):
    values = [parse_float(row[key]) for row in rows if key in row and row[key] != ""]
    return float(np.median(values)) if values else float("nan")


def safe_ratio_improvement(before, after):
    before = float(before)
    after = float(after)
    if not math.isfinite(before) or abs(before) < 1e-12:
        return 0.0
    return float((before - after) / before)


def image_label(row, prefix):
    return (
        f"{prefix} P{row['process']} {row['candidate']} "
        f"d={parse_float(row['weighted_relative_error']):.2f} "
        f"f={parse_float(row['full_ratio']):.2f}"
    )


def copy_selected(row, selected_root, selected_index):
    process_root = selected_root / f"process_{row['process']}"
    process_root.mkdir(parents=True, exist_ok=True)
    src = Path(row["crop_path"])
    name = (
        f"{row['process']}.t007_selected_"
        f"{row['candidate']}_{selected_index:02d}_{src.stem}.png"
    )
    dst = process_root / name
    shutil.copy2(src, dst)
    return str(dst)


def make_selected_grid(rows, path):
    columns = 4
    cell = 180
    header = 44
    canvas_rows = int(math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * cell, canvas_rows * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        x = (idx % columns) * cell
        y = (idx // columns) * (cell + header)
        label = (
            f"P{row['process']} d={parse_float(row['weighted_relative_error']):.2f} "
            f"f={parse_float(row['full_ratio']):.1f}"
        )
        draw.text((x + 4, y + 7), label, fill="black")
        image = Image.open(row["selected_path"]).convert("RGB").resize((cell, cell))
        canvas.paste(image, (x, y + header))
    canvas.save(path)


def make_before_after_grid(baseline_rows, selected_rows, path):
    pairs = []
    for process in sorted({row["process"] for row in selected_rows}, key=int):
        base = [row for row in baseline_rows if row["process"] == process]
        new = [row for row in selected_rows if row["process"] == process]
        for index in range(min(len(base), len(new))):
            pairs.append((base[index], new[index]))

    cell = 170
    header = 50
    columns = 2
    canvas = Image.new("RGB", (columns * cell, len(pairs) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (base, new) in enumerate(pairs):
        y = idx * (cell + header)
        draw.text((4, y + 6), image_label(base, "T005")[:45], fill="black")
        draw.text((cell + 4, y + 6), image_label(new, "T007")[:45], fill="black")
        base_path = base.get("selected_path") or base["crop_path"]
        base_img = Image.open(base_path).convert("RGB").resize((cell, cell))
        new_img = Image.open(new["selected_path"]).convert("RGB").resize((cell, cell))
        canvas.paste(base_img, (0, y + header))
        canvas.paste(new_img, (cell, y + header))
    canvas.save(path)


def make_nearest_grid(rows, path, max_rows=16):
    rows = rows[:max_rows]
    cell = 170
    header = 48
    canvas = Image.new("RGB", (2 * cell, len(rows) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        y = idx * (cell + header)
        draw.text(
            (4, y + 6),
            f"P{row['process']} full={parse_float(row['full_ratio']):.2f} "
            f"lbp={parse_float(row['lbp_texture_ratio']):.2f}",
            fill="black",
        )
        gen = Image.open(row["selected_path"]).convert("RGB").resize((cell, cell))
        real = Image.open(row["nearest_real_path_full"]).convert("RGB").resize((cell, cell))
        canvas.paste(gen, (0, y + header))
        canvas.paste(real, (cell, y + header))
    canvas.save(path)


def build_real_records(rows):
    return [{"path": row["path"], "features": feature_groups(row["path"])} for row in rows]


def add_visual_scores(process, candidate_rows, real_rows, max_real_per_process):
    real_subset = choose_real_rows(real_rows, process, max_real_per_process)
    reference_rows, holdout_rows = split_reference_holdout(real_subset)
    reference_records = build_real_records(reference_rows)
    holdout_records = build_real_records(holdout_rows)

    candidate_records = []
    for row in candidate_rows:
        candidate_records.append(
            {
                "row": row,
                "path": row["crop_path"],
                "features": feature_groups(row["crop_path"]),
            }
        )

    holdout_medians = {}
    leakage_thresholds = {}
    for group in FEATURE_GROUPS:
        reference = stack_features(reference_records, group)
        holdout = stack_features(holdout_records, group)
        generated = np.stack([record["features"][group] for record in candidate_records]).astype(np.float32)
        reference_z, holdout_z, generated_z = standardize(reference, reference, holdout, generated)
        holdout_dist, _ = nearest(holdout_z, reference_z)
        generated_dist, generated_nn = nearest(generated_z, reference_z)
        holdout_median = float(np.median(holdout_dist))
        holdout_medians[group] = holdout_median
        leakage_thresholds[group] = 0.35 * holdout_median

        for idx, record in enumerate(candidate_records):
            row = record["row"]
            dist = float(generated_dist[idx])
            ratio = float(dist / max(holdout_median, 1e-8))
            row[f"{group}_nn_distance"] = dist
            row[f"{group}_ratio"] = ratio
            row[f"{group}_potential_leakage"] = bool(dist < leakage_thresholds[group])
            row[f"nearest_real_path_{group}"] = reference_records[int(generated_nn[idx])]["path"]

    for record in candidate_records:
        row = record["row"]
        row["visual_raw_score"] = float(
            sum(VISUAL_WEIGHTS[group] * math.log(max(parse_float(row[f"{group}_ratio"]), 1e-8))
                for group in VISUAL_WEIGHTS)
        )
        row["critical_leakage"] = bool(
            any(row[f"{group}_potential_leakage"] for group in CRITICAL_GROUPS)
        )
        row["descriptor_eligible"] = bool(
            parse_float(row["weighted_relative_error"]) <= 0.50
            and parse_float(row["spectral_spacing_px_relative_error"]) <= 0.50
        )
        row["descriptor_relaxed_eligible"] = bool(
            parse_float(row["weighted_relative_error"]) <= 0.75
            and parse_float(row["spectral_spacing_px_relative_error"]) <= 0.75
        )

    descriptor_ranks = percentile_ranks([parse_float(row["weighted_relative_error"]) for row in candidate_rows])
    visual_ranks = percentile_ranks([parse_float(row["visual_raw_score"]) for row in candidate_rows])
    for idx, row in enumerate(candidate_rows):
        leakage_penalty = 2.0 if row["critical_leakage"] else 0.0
        descriptor_penalty = 0.4 if not row["descriptor_relaxed_eligible"] else 0.0
        row["descriptor_rank"] = float(descriptor_ranks[idx])
        row["visual_rank"] = float(visual_ranks[idx])
        row["rerank_score"] = float(
            0.45 * row["descriptor_rank"] + 0.55 * row["visual_rank"] + leakage_penalty + descriptor_penalty
        )
        row["real_reference_count"] = len(reference_records)
        row["real_holdout_count"] = len(holdout_records)

    return candidate_rows


def select_rows(process_rows, samples_per_process):
    pools = [
        [row for row in process_rows if row["descriptor_eligible"] and not row["critical_leakage"]],
        [row for row in process_rows if row["descriptor_relaxed_eligible"] and not row["critical_leakage"]],
        [row for row in process_rows if not row["critical_leakage"]],
        list(process_rows),
    ]
    pool = next((rows for rows in pools if len(rows) >= samples_per_process), pools[-1])
    ordered = sorted(pool, key=lambda row: (parse_float(row["rerank_score"]), parse_float(row["visual_raw_score"])))

    selected = []
    used_grids = set()
    for row in ordered:
        if row["grid_path"] in used_grids:
            continue
        selected.append(row)
        used_grids.add(row["grid_path"])
        if len(selected) == samples_per_process:
            break
    if len(selected) < samples_per_process:
        for row in ordered:
            if row in selected:
                continue
            selected.append(row)
            if len(selected) == samples_per_process:
                break
    return selected


def attach_baseline_visual_rows(baseline_rows, scored_by_crop):
    rows = []
    for row in baseline_rows:
        scored = dict(scored_by_crop[row["crop_path"]])
        scored["selected_path"] = row["selected_path"]
        scored["baseline_selected_path"] = row["selected_path"]
        rows.append(scored)
    return rows


def summarize_process(process, baseline_rows, selected_rows):
    baseline_desc = median(baseline_rows, "weighted_relative_error")
    selected_desc = median(selected_rows, "weighted_relative_error")
    baseline_full = median(baseline_rows, "full_ratio")
    selected_full = median(selected_rows, "full_ratio")
    baseline_lbp = median(baseline_rows, "lbp_texture_ratio")
    selected_lbp = median(selected_rows, "lbp_texture_ratio")
    baseline_spectrum = median(baseline_rows, "spectrum_ratio")
    selected_spectrum = median(selected_rows, "spectrum_ratio")
    baseline_appearance = median(baseline_rows, "appearance_ratio")
    selected_appearance = median(selected_rows, "appearance_ratio")
    selected_block = median(selected_rows, "block_texture_ratio")
    selected_leakage = float(np.mean([bool(row["critical_leakage"]) for row in selected_rows]))

    full_improvement = safe_ratio_improvement(baseline_full, selected_full)
    lbp_improvement = safe_ratio_improvement(baseline_lbp, selected_lbp)
    spectrum_improvement = safe_ratio_improvement(baseline_spectrum, selected_spectrum)
    descriptor_ok = selected_desc <= max(0.50, baseline_desc * 1.50)
    relative_visual_improved = bool(
        full_improvement >= 0.20
        and (lbp_improvement >= 0.10 or spectrum_improvement >= 0.10)
        and descriptor_ok
        and selected_leakage <= 0.25
    )
    absolute_visual_supported = bool(
        selected_full <= 2.0
        and selected_lbp <= 2.0
        and selected_spectrum <= 2.0
        and selected_block <= 2.0
        and selected_leakage <= 0.25
    )

    return {
        "process": process,
        "baseline_weighted_relative_error_median": baseline_desc,
        "t007_weighted_relative_error_median": selected_desc,
        "baseline_full_ratio_median": baseline_full,
        "t007_full_ratio_median": selected_full,
        "full_ratio_improvement_fraction": full_improvement,
        "baseline_lbp_texture_ratio_median": baseline_lbp,
        "t007_lbp_texture_ratio_median": selected_lbp,
        "lbp_texture_improvement_fraction": lbp_improvement,
        "baseline_spectrum_ratio_median": baseline_spectrum,
        "t007_spectrum_ratio_median": selected_spectrum,
        "spectrum_improvement_fraction": spectrum_improvement,
        "baseline_appearance_ratio_median": baseline_appearance,
        "t007_appearance_ratio_median": selected_appearance,
        "t007_block_texture_ratio_median": selected_block,
        "t007_leakage_fraction": selected_leakage,
        "relative_visual_improved": relative_visual_improved,
        "absolute_visual_realism_supported": absolute_visual_supported,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t005_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--samples_per_process", type=int, default=4)
    parser.add_argument("--max_real_per_process", type=int, default=500)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    selected_root = args.out_root / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)

    real_rows = read_csv(args.t002_root / "patch_descriptors.csv")
    all_rows = read_csv(args.t005_root / "ldm_generated_descriptors.csv")
    baseline_rows = read_csv(args.t005_root / "ldm_selected_samples.csv")

    scored_rows = []
    selected_rows = []
    baseline_scored_rows = []
    process_rows_for_report = {}

    for process in sorted({row["process"] for row in all_rows}, key=int):
        process_candidates = [dict(row) for row in all_rows if row["process"] == process]
        process_scored = add_visual_scores(
            process,
            process_candidates,
            real_rows,
            args.max_real_per_process,
        )
        scored_rows.extend(process_scored)
        selected = select_rows(process_scored, args.samples_per_process)
        for idx, row in enumerate(selected):
            row["selected_path"] = copy_selected(row, selected_root, idx)
            row["selection_rank"] = idx
            selected_rows.append(row)
        process_rows_for_report[process] = process_scored

    scored_by_crop = {row["crop_path"]: row for row in scored_rows}
    baseline_scored_rows = attach_baseline_visual_rows(baseline_rows, scored_by_crop)

    comparison_rows = []
    report_processes = {}
    for process in sorted({row["process"] for row in selected_rows}, key=int):
        baseline_process = [row for row in baseline_scored_rows if row["process"] == process]
        selected_process = [row for row in selected_rows if row["process"] == process]
        summary = summarize_process(process, baseline_process, selected_process)
        comparison_rows.append(summary)
        report_processes[process] = summary

    relative_improved_processes = [
        row["process"] for row in comparison_rows if row["relative_visual_improved"]
    ]
    absolute_supported_processes = [
        row["process"] for row in comparison_rows if row["absolute_visual_realism_supported"]
    ]
    p9_relative_improved = bool(report_processes.get("9", {}).get("relative_visual_improved", False))
    p9_absolute_supported = bool(
        report_processes.get("9", {}).get("absolute_visual_realism_supported", False)
    )
    multi_relative_improved = len(relative_improved_processes) >= 3
    multi_absolute_supported = len(absolute_supported_processes) >= 3

    gate = {
        "p9_visual_improvement_supported": p9_relative_improved,
        "multi_process_visual_improvement_supported": multi_relative_improved,
        "relative_improved_processes": relative_improved_processes,
        "p9_visual_realism_claim_supported": p9_absolute_supported,
        "multi_process_visual_realism_claim_supported": multi_absolute_supported,
        "absolute_supported_processes": absolute_supported_processes,
        "recommendation": (
            "T-007 supports a stronger visual selection figure, but not a full visual-realism claim."
            if (p9_relative_improved or multi_relative_improved) and not (p9_absolute_supported and multi_absolute_supported)
            else (
                "T-007 supports strengthening the visual-realism claim."
                if p9_absolute_supported and multi_absolute_supported
                else "T-007 does not sufficiently improve visual realism; open a training/discriminator branch."
            )
        ),
    }

    report = {
        "method": "Visual-realism-aware reranking of the existing T-005 LDM candidate pool.",
        "selection": {
            "samples_per_process": args.samples_per_process,
            "max_real_per_process": args.max_real_per_process,
            "visual_weights": VISUAL_WEIGHTS,
            "critical_groups": CRITICAL_GROUPS,
            "descriptor_eligible": "weighted_relative_error <= 0.50 and spectral_spacing_px_relative_error <= 0.50",
            "relative_improvement_gate": "full improvement >= 20%, LBP or spectrum improvement >= 10%, descriptor median not worse than max(0.50, 1.5x baseline), leakage <= 25%",
            "absolute_visual_realism_gate": "full, LBP, spectrum, and block texture ratios <= 2.0 with leakage <= 25%",
        },
        "processes": report_processes,
        "gate": gate,
        "notes": [
            "This task reranks existing generated crops only; it does not train a new generator.",
            "A relative improvement can justify choosing better-looking images for figures, but does not by itself prove visual realism.",
            "Absolute visual-realism support requires the selected generated images to be close to real holdout patches in texture and spectrum.",
        ],
    }

    score_fields = [
        "process",
        "candidate",
        "grid_path",
        "crop_path",
        "weighted_relative_error",
        "spectral_spacing_px_relative_error",
        "descriptor_eligible",
        "descriptor_relaxed_eligible",
        "descriptor_rank",
        "visual_raw_score",
        "visual_rank",
        "rerank_score",
        "critical_leakage",
        "appearance_ratio",
        "block_texture_ratio",
        "gradient_hist_ratio",
        "lbp_texture_ratio",
        "spectrum_ratio",
        "full_ratio",
        "nearest_real_path_full",
    ]
    selected_fields = score_fields + [
        "selection_rank",
        "selected_path",
        "gradient_mean_relative_error",
        "sobel_top10_mean_relative_error",
        "low_frequency_contrast_relative_error",
        "dark_fraction_relative_error",
        "intensity_mean_relative_error",
    ]
    comparison_fields = [
        "process",
        "baseline_weighted_relative_error_median",
        "t007_weighted_relative_error_median",
        "baseline_full_ratio_median",
        "t007_full_ratio_median",
        "full_ratio_improvement_fraction",
        "baseline_lbp_texture_ratio_median",
        "t007_lbp_texture_ratio_median",
        "lbp_texture_improvement_fraction",
        "baseline_spectrum_ratio_median",
        "t007_spectrum_ratio_median",
        "spectrum_improvement_fraction",
        "baseline_appearance_ratio_median",
        "t007_appearance_ratio_median",
        "t007_block_texture_ratio_median",
        "t007_leakage_fraction",
        "relative_visual_improved",
        "absolute_visual_realism_supported",
    ]

    write_csv(args.out_root / "all_candidate_visual_scores.csv", scored_rows, score_fields)
    write_csv(args.out_root / "t007_selected_samples.csv", selected_rows, selected_fields)
    write_csv(args.out_root / "process_comparison.csv", comparison_rows, comparison_fields)
    make_selected_grid(selected_rows, args.out_root / "t007_selected_grid.png")
    make_before_after_grid(baseline_scored_rows, selected_rows, args.out_root / "t007_before_after_grid.png")
    make_nearest_grid(selected_rows, args.out_root / "t007_nearest_grid.png")
    (args.out_root / "t007_visual_rerank_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
