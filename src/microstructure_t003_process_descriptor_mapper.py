#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


FEATURE_COLUMNS = ("laser_power", "scan_speed", "dwell_time", "linear_energy")

MORPHOLOGY_DESCRIPTORS = {
    "spectral_spacing_px": "target_spectral_spacing_px",
    "spectral_peak_confidence": "target_spectral_peak_confidence",
    "gradient_mean": "target_gradient_mean",
    "sobel_top10_mean": "target_sobel_top10_mean",
    "low_frequency_contrast": "target_low_frequency_contrast",
}

APPEARANCE_DESCRIPTORS = {
    "dark_fraction": "target_dark_fraction",
    "intensity_mean": "intensity_mean_median",
    "intensity_std": "intensity_std_median",
    "bright_fraction": "bright_fraction_median",
}


def read_csv(path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_descriptor_names(rows, descriptor_map):
    names = []
    for name, column in descriptor_map.items():
        values = [parse_float(row.get(column)) for row in rows]
        if all(math.isfinite(v) for v in values):
            names.append(name)
    return names


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def fit_ridge(train_x, train_y, test_x, alpha=1.0):
    train_xs, test_xs = standardize(train_x, test_x)
    design = np.concatenate([np.ones((train_xs.shape[0], 1)), train_xs], axis=1)
    test_design = np.concatenate([np.ones((test_xs.shape[0], 1)), test_xs], axis=1)
    reg = np.eye(design.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ train_y)
    return test_design @ beta


def nearest_neighbor(train_x, train_y, test_x):
    train_xs, test_xs = standardize(train_x, test_x)
    predictions = []
    neighbor_indices = []
    for row in test_xs:
        distances = np.sqrt(((train_xs - row) ** 2).sum(axis=1))
        index = int(np.argmin(distances))
        predictions.append(train_y[index])
        neighbor_indices.append(index)
    return np.vstack(predictions), neighbor_indices


def leave_one_out(rows, descriptor_names, descriptor_columns):
    x = np.array([[parse_float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.array(
        [[parse_float(row[descriptor_columns[name]]) for name in descriptor_names] for row in rows],
        dtype=float,
    )
    processes = [row["process"] for row in rows]

    prediction_rows = []
    predictions = {"ridge": np.zeros_like(y), "nearest_parameter_neighbor": np.zeros_like(y), "mean_baseline": np.zeros_like(y)}
    neighbor_processes = []

    for held_index in range(len(rows)):
        mask = np.ones(len(rows), dtype=bool)
        mask[held_index] = False
        train_x, test_x = x[mask], x[~mask]
        train_y = y[mask]

        predictions["ridge"][held_index : held_index + 1] = fit_ridge(train_x, train_y, test_x)
        nn_pred, nn_indices = nearest_neighbor(train_x, train_y, test_x)
        predictions["nearest_parameter_neighbor"][held_index : held_index + 1] = nn_pred
        predictions["mean_baseline"][held_index : held_index + 1] = train_y.mean(axis=0, keepdims=True)
        train_processes = [p for j, p in enumerate(processes) if j != held_index]
        neighbor_processes.append(train_processes[nn_indices[0]])

    for method, pred in predictions.items():
        for i, process in enumerate(processes):
            for j, descriptor in enumerate(descriptor_names):
                prediction_rows.append(
                    {
                        "method": method,
                        "heldout_process": process,
                        "nearest_process": neighbor_processes[i] if method == "nearest_parameter_neighbor" else "",
                        "descriptor": descriptor,
                        "observed": y[i, j],
                        "predicted": pred[i, j],
                        "absolute_error": abs(pred[i, j] - y[i, j]),
                    }
                )

    return x, y, processes, predictions, prediction_rows


def metric_rows(y, predictions, descriptor_names):
    rows = []
    descriptor_ranges = np.maximum(y.max(axis=0) - y.min(axis=0), 1e-12)
    for method, pred in predictions.items():
        for j, descriptor in enumerate(descriptor_names):
            errors = pred[:, j] - y[:, j]
            rmse = float(np.sqrt(np.mean(errors**2)))
            mae = float(np.mean(np.abs(errors)))
            rows.append(
                {
                    "method": method,
                    "descriptor": descriptor,
                    "rmse": rmse,
                    "mae": mae,
                    "range_normalized_rmse": rmse / float(descriptor_ranges[j]),
                }
            )
    return rows


def summarize_method(metrics, descriptor_group):
    by_method = {}
    for row in metrics:
        if row["descriptor"] not in descriptor_group:
            continue
        by_method.setdefault(row["method"], []).append(float(row["range_normalized_rmse"]))
    return {method: float(np.median(values)) for method, values in by_method.items() if values}


def load_eta_squared(t002_root):
    path = t002_root / "separability.json"
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    return data.get("eta_squared", {})


def make_summary_figure(path, rows, descriptor_names, descriptor_columns):
    width, height = 1200, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), "T-003 process-to-descriptor target assessment", fill="black")

    processes = [row["process"] for row in rows]
    colors = ["#6f6f6f", "#2f70b7", "#50975b", "#c56338", "#9467bd", "#8c564b"]
    panel_w, panel_h = 500, 230
    x0s = [70, 680]
    y0s = [110, 430]
    selected = descriptor_names[:4]
    for k, descriptor in enumerate(selected):
        x0 = x0s[k % 2]
        y0 = y0s[k // 2]
        draw.text((x0, y0 - 32), descriptor, fill="black")
        values = np.array([parse_float(row[descriptor_columns[descriptor]]) for row in rows], dtype=float)
        vmin, vmax = float(values.min()), float(values.max())
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline="#222222")
        for i, (process, value) in enumerate(zip(processes, values)):
            bar_w = 70
            gap = 40
            base_x = x0 + 45 + i * (bar_w + gap)
            scaled = (float(value) - vmin) / (vmax - vmin)
            bar_h = int(scaled * (panel_h - 70))
            y1 = y0 + panel_h - 35
            draw.rectangle((base_x, y1 - bar_h, base_x + bar_w, y1), fill=colors[i % len(colors)])
            draw.text((base_x + 18, y1 + 8), f"P{process}", fill="black")
            draw.text((base_x - 5, y1 - bar_h - 18), f"{value:.3g}", fill="black")
    image.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    target_rows = read_csv(args.t002_root / "process_descriptor_targets.csv")
    summary_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_summary.csv")}
    rows = []
    for row in target_rows:
        merged = dict(row)
        merged.update(summary_rows.get(row["process"], {}))
        rows.append(merged)
    rows = sorted(rows, key=lambda row: int(row["process"]))

    morphology_names = finite_descriptor_names(rows, MORPHOLOGY_DESCRIPTORS)
    appearance_names = finite_descriptor_names(rows, APPEARANCE_DESCRIPTORS)
    descriptor_columns = {}
    descriptor_columns.update({name: MORPHOLOGY_DESCRIPTORS[name] for name in morphology_names})
    descriptor_columns.update({name: APPEARANCE_DESCRIPTORS[name] for name in appearance_names})
    descriptor_names = morphology_names + appearance_names

    x, y, processes, predictions, prediction_rows = leave_one_out(rows, descriptor_names, descriptor_columns)
    metrics = metric_rows(y, predictions, descriptor_names)
    morphology_method_summary = summarize_method(metrics, morphology_names)
    appearance_method_summary = summarize_method(metrics, appearance_names)
    eta_squared = load_eta_squared(args.t002_root)

    max_morphology_eta = max((float(eta_squared.get(name, 0.0)) for name in morphology_names), default=0.0)
    observed_regime_supported = len(rows) >= 3 and max_morphology_eta > 0.10 and all(
        math.isfinite(parse_float(row[descriptor_columns[morphology_names[0]]])) for row in rows
    )
    ridge_morphology_nrmse = morphology_method_summary.get("ridge", float("inf"))
    nn_morphology_nrmse = morphology_method_summary.get("nearest_parameter_neighbor", float("inf"))
    mean_morphology_nrmse = morphology_method_summary.get("mean_baseline", float("inf"))
    best_predictive_nrmse = min(ridge_morphology_nrmse, nn_morphology_nrmse)

    continuous_supported = (
        len(rows) >= 8
        and best_predictive_nrmse <= 0.35
        and best_predictive_nrmse < 0.80 * mean_morphology_nrmse
    )

    if not observed_regime_supported:
        recommendation = "STOP: observed-regime descriptor routing is not supported."
    elif continuous_supported:
        recommendation = "Proceed: mapper evidence supports a cautious continuous parameter pilot."
    else:
        recommendation = (
            "Proceed only with observed-regime descriptor routing; do not claim arbitrary "
            "unseen process-parameter prediction."
        )

    report = {
        "n_processes": len(rows),
        "processes": processes,
        "feature_columns": list(FEATURE_COLUMNS),
        "descriptor_groups": {
            "morphology": morphology_names,
            "appearance": appearance_names,
        },
        "gate": {
            "observed_regime_descriptor_routing_supported": observed_regime_supported,
            "continuous_unseen_process_regression_supported": continuous_supported,
            "minimum_process_count_for_continuous_gate": 8,
            "max_morphology_eta_squared": max_morphology_eta,
            "ridge_morphology_median_range_normalized_rmse": ridge_morphology_nrmse,
            "nearest_neighbor_morphology_median_range_normalized_rmse": nn_morphology_nrmse,
            "mean_baseline_morphology_median_range_normalized_rmse": mean_morphology_nrmse,
            "recommendation": recommendation,
        },
        "method_summary": {
            "morphology_median_range_normalized_rmse": morphology_method_summary,
            "appearance_median_range_normalized_rmse": appearance_method_summary,
        },
        "eta_squared": eta_squared,
        "notes": [
            "Only four observed process regimes are available.",
            "Observed-regime routing can use the descriptor targets directly.",
            "Continuous unseen-parameter prediction requires more unique process regimes before it can be claimed.",
        ],
    }

    with (args.out_root / "process_descriptor_mapper.json").open("w") as f:
        json.dump(report, f, indent=2)
    with (args.out_root / "descriptor_target_groups.json").open("w") as f:
        json.dump(report["descriptor_groups"], f, indent=2)

    write_csv(
        args.out_root / "descriptor_mapping_metrics.csv",
        metrics,
        ["method", "descriptor", "rmse", "mae", "range_normalized_rmse"],
    )
    write_csv(
        args.out_root / "loo_predictions.csv",
        prediction_rows,
        ["method", "heldout_process", "nearest_process", "descriptor", "observed", "predicted", "absolute_error"],
    )
    make_summary_figure(args.out_root / "process_descriptor_mapper.png", rows, descriptor_names, descriptor_columns)

    print(
        json.dumps(
            {
                "n_processes": report["n_processes"],
                "observed_regime_descriptor_routing_supported": observed_regime_supported,
                "continuous_unseen_process_regression_supported": continuous_supported,
                "recommendation": recommendation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
