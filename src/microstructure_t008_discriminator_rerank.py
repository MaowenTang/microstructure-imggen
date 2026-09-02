#!/usr/bin/env python3

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from microstructure_t006_visual_realism import choose_real_rows, read_csv, split_reference_holdout, write_csv
from microstructure_t007_visual_rerank import median, parse_float, percentile_ranks, safe_ratio_improvement


CRITICAL_GROUPS = ("full", "lbp_texture", "spectrum", "block_texture")


class SmallDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_image(path, image_size):
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def auc_score(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bool_value(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def deterministic_fake_split(rows):
    train = []
    valid = []
    for idx, row in enumerate(sorted(rows, key=lambda item: (item["process"], item["candidate"], item["crop_path"]))):
        if idx % 5 == 0:
            valid.append(row)
        else:
            train.append(row)
    return train, valid


def build_real_split(real_rows, processes, max_real_per_process):
    train = []
    valid = []
    for process in processes:
        subset = choose_real_rows(real_rows, process, max_real_per_process)
        reference, holdout = split_reference_holdout(subset)
        train.extend(reference)
        valid.extend(holdout)
    return train, valid


def make_arrays(real_train, real_valid, fake_train, fake_valid, image_size):
    train_paths = [row["path"] for row in real_train] + [row["crop_path"] for row in fake_train]
    valid_paths = [row["path"] for row in real_valid] + [row["crop_path"] for row in fake_valid]
    train_labels = np.asarray([1] * len(real_train) + [0] * len(fake_train), dtype=np.float32)
    valid_labels = np.asarray([1] * len(real_valid) + [0] * len(fake_valid), dtype=np.float32)

    train_images = np.stack([load_image(path, image_size) for path in train_paths]).astype(np.float32)
    valid_images = np.stack([load_image(path, image_size) for path in valid_paths]).astype(np.float32)
    mean = train_images.mean(axis=(0, 2, 3), keepdims=True)
    std = train_images.std(axis=(0, 2, 3), keepdims=True)
    std[std < 1e-6] = 1.0
    train_images = (train_images - mean) / std
    valid_images = (valid_images - mean) / std
    return train_images, train_labels, valid_images, valid_labels, mean, std


def train_model(train_images, train_labels, valid_images, valid_labels, epochs, batch_size, lr):
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = SmallDiscriminator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dataset = TensorDataset(torch.from_numpy(train_images), torch.from_numpy(train_labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    real_count = float((train_labels == 1).sum())
    fake_count = float((train_labels == 0).sum())
    class_weights = {
        1.0: 0.5 / max(real_count, 1.0),
        0.0: 0.5 / max(fake_count, 1.0),
    }
    history = []
    valid_tensor = torch.from_numpy(valid_images).to(device)
    valid_label_tensor = torch.from_numpy(valid_labels).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss_values = nn.functional.binary_cross_entropy_with_logits(logits, batch_y, reduction="none")
            weights = torch.where(
                batch_y > 0.5,
                torch.full_like(batch_y, class_weights[1.0]),
                torch.full_like(batch_y, class_weights[0.0]),
            )
            loss = (loss_values * weights).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_tensor)
            valid_probs = torch.sigmoid(valid_logits)
            valid_loss = float(
                nn.functional.binary_cross_entropy(valid_probs, valid_label_tensor).detach().cpu()
            )
            valid_scores = valid_probs.detach().cpu().numpy()
        valid_auc = auc_score(valid_labels, valid_scores)
        valid_acc = float(((valid_scores >= 0.5).astype(np.float32) == valid_labels).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "valid_loss": valid_loss,
                "valid_auc": valid_auc,
                "valid_accuracy": valid_acc,
            }
        )
    return model, history


def score_images(model, rows, mean, std, image_size, batch_size):
    device = torch.device("cpu")
    model.eval()
    probs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = np.stack([load_image(row["crop_path"], image_size) for row in batch_rows]).astype(np.float32)
        images = (images - mean) / std
        with torch.no_grad():
            logits = model(torch.from_numpy(images).to(device))
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(float(value) for value in batch_probs)
    return probs


def copy_selected(row, selected_root, selected_index):
    process_root = selected_root / f"process_{row['process']}"
    process_root.mkdir(parents=True, exist_ok=True)
    src = Path(row["crop_path"])
    dst = process_root / (
        f"{row['process']}.t008_selected_{row['candidate']}_{selected_index:02d}_{src.stem}.png"
    )
    shutil.copy2(src, dst)
    return str(dst)


def select_rows(process_rows, samples_per_process):
    pools = [
        [
            row
            for row in process_rows
            if bool_value(row["descriptor_eligible"]) and not bool_value(row["critical_leakage"])
        ],
        [
            row
            for row in process_rows
            if bool_value(row["descriptor_relaxed_eligible"]) and not bool_value(row["critical_leakage"])
        ],
        [row for row in process_rows if not bool_value(row["critical_leakage"])],
        list(process_rows),
    ]
    pool = next((rows for rows in pools if len(rows) >= samples_per_process), pools[-1])
    ordered = sorted(
        pool,
        key=lambda row: (
            parse_float(row["discriminator_rerank_score"]),
            -parse_float(row["discriminator_real_probability"]),
        ),
    )
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


def attach_visual_scores(all_rows, visual_rows):
    visual_by_crop = {row["crop_path"]: row for row in visual_rows}
    merged = []
    for row in all_rows:
        out = dict(visual_by_crop[row["crop_path"]])
        out["discriminator_real_probability"] = row["discriminator_real_probability"]
        merged.append(out)
    return merged


def make_selected_grid(rows, path):
    columns = 4
    cell = 180
    header = 48
    canvas_rows = int(math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * cell, canvas_rows * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        x = (idx % columns) * cell
        y = (idx // columns) * (cell + header)
        label = (
            f"P{row['process']} p={parse_float(row['discriminator_real_probability']):.2f} "
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
    header = 54
    canvas = Image.new("RGB", (2 * cell, len(pairs) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (base, new) in enumerate(pairs):
        y = idx * (cell + header)
        draw.text((4, y + 6), f"T005 P{base['process']} f={parse_float(base['full_ratio']):.1f}", fill="black")
        draw.text(
            (cell + 4, y + 6),
            f"T008 P{new['process']} p={parse_float(new['discriminator_real_probability']):.2f} f={parse_float(new['full_ratio']):.1f}",
            fill="black",
        )
        base_path = base.get("selected_path") or base["crop_path"]
        base_img = Image.open(base_path).convert("RGB").resize((cell, cell))
        new_img = Image.open(new["selected_path"]).convert("RGB").resize((cell, cell))
        canvas.paste(base_img, (0, y + header))
        canvas.paste(new_img, (cell, y + header))
    canvas.save(path)


def make_nearest_grid(rows, path, max_rows=16):
    rows = rows[:max_rows]
    cell = 170
    header = 50
    canvas = Image.new("RGB", (2 * cell, len(rows) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, row in enumerate(rows):
        y = idx * (cell + header)
        draw.text(
            (4, y + 6),
            f"P{row['process']} p={parse_float(row['discriminator_real_probability']):.2f} full={parse_float(row['full_ratio']):.1f}",
            fill="black",
        )
        gen = Image.open(row["selected_path"]).convert("RGB").resize((cell, cell))
        real = Image.open(row["nearest_real_path_full"]).convert("RGB").resize((cell, cell))
        canvas.paste(gen, (0, y + header))
        canvas.paste(real, (cell, y + header))
    canvas.save(path)


def summarize_process(process, baseline_rows, selected_rows):
    baseline_desc = median(baseline_rows, "weighted_relative_error")
    selected_desc = median(selected_rows, "weighted_relative_error")
    baseline_full = median(baseline_rows, "full_ratio")
    selected_full = median(selected_rows, "full_ratio")
    baseline_lbp = median(baseline_rows, "lbp_texture_ratio")
    selected_lbp = median(selected_rows, "lbp_texture_ratio")
    baseline_spectrum = median(baseline_rows, "spectrum_ratio")
    selected_spectrum = median(selected_rows, "spectrum_ratio")
    selected_block = median(selected_rows, "block_texture_ratio")
    selected_prob = median(selected_rows, "discriminator_real_probability")
    leakage = float(np.mean([bool_value(row["critical_leakage"]) for row in selected_rows]))
    full_improvement = safe_ratio_improvement(baseline_full, selected_full)
    lbp_improvement = safe_ratio_improvement(baseline_lbp, selected_lbp)
    spectrum_improvement = safe_ratio_improvement(baseline_spectrum, selected_spectrum)
    descriptor_ok = selected_desc <= max(0.50, baseline_desc * 1.50)
    relative_visual_improved = bool(
        full_improvement >= 0.20
        and (lbp_improvement >= 0.10 or spectrum_improvement >= 0.10)
        and descriptor_ok
        and leakage <= 0.25
    )
    absolute_visual_supported = bool(
        selected_full <= 2.0
        and selected_lbp <= 2.0
        and selected_spectrum <= 2.0
        and selected_block <= 2.0
        and leakage <= 0.25
    )
    return {
        "process": process,
        "baseline_weighted_relative_error_median": baseline_desc,
        "t008_weighted_relative_error_median": selected_desc,
        "baseline_full_ratio_median": baseline_full,
        "t008_full_ratio_median": selected_full,
        "full_ratio_improvement_fraction": full_improvement,
        "baseline_lbp_texture_ratio_median": baseline_lbp,
        "t008_lbp_texture_ratio_median": selected_lbp,
        "lbp_texture_improvement_fraction": lbp_improvement,
        "baseline_spectrum_ratio_median": baseline_spectrum,
        "t008_spectrum_ratio_median": selected_spectrum,
        "spectrum_improvement_fraction": spectrum_improvement,
        "t008_block_texture_ratio_median": selected_block,
        "t008_discriminator_real_probability_median": selected_prob,
        "t008_leakage_fraction": leakage,
        "relative_visual_improved": relative_visual_improved,
        "absolute_visual_realism_supported": absolute_visual_supported,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t005_root", required=True, type=Path)
    parser.add_argument("--t007_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--samples_per_process", type=int, default=4)
    parser.add_argument("--max_real_per_process", type=int, default=500)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    selected_root = args.out_root / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(7)
    np.random.seed(7)

    real_rows = read_csv(args.t002_root / "patch_descriptors.csv")
    generated_rows = read_csv(args.t005_root / "ldm_generated_descriptors.csv")
    baseline_rows = read_csv(args.t005_root / "ldm_selected_samples.csv")
    visual_rows = read_csv(args.t007_root / "all_candidate_visual_scores.csv")

    processes = sorted({row["process"] for row in generated_rows}, key=int)
    real_train, real_valid = build_real_split(real_rows, processes, args.max_real_per_process)
    fake_train, fake_valid = deterministic_fake_split(generated_rows)
    train_images, train_labels, valid_images, valid_labels, mean, std = make_arrays(
        real_train, real_valid, fake_train, fake_valid, args.image_size
    )
    model, history = train_model(
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    probabilities = score_images(model, generated_rows, mean, std, args.image_size, args.batch_size)
    for row, probability in zip(generated_rows, probabilities):
        row["discriminator_real_probability"] = probability

    merged_rows = attach_visual_scores(generated_rows, visual_rows)
    for process in processes:
        process_rows = [row for row in merged_rows if row["process"] == process]
        descriptor_ranks = percentile_ranks([parse_float(row["weighted_relative_error"]) for row in process_rows])
        visual_ranks = percentile_ranks([parse_float(row["visual_raw_score"]) for row in process_rows])
        realness_ranks = percentile_ranks([-parse_float(row["discriminator_real_probability"]) for row in process_rows])
        for idx, row in enumerate(process_rows):
            row["discriminator_rank"] = float(realness_ranks[idx])
            leakage_penalty = 2.0 if bool_value(row["critical_leakage"]) else 0.0
            descriptor_penalty = 0.4 if not bool_value(row["descriptor_relaxed_eligible"]) else 0.0
            row["discriminator_rerank_score"] = float(
                0.50 * realness_ranks[idx]
                + 0.30 * descriptor_ranks[idx]
                + 0.20 * visual_ranks[idx]
                + leakage_penalty
                + descriptor_penalty
            )

    selected_rows = []
    for process in processes:
        process_rows = [row for row in merged_rows if row["process"] == process]
        selected = select_rows(process_rows, args.samples_per_process)
        for idx, row in enumerate(selected):
            row["selected_path"] = copy_selected(row, selected_root, idx)
            row["selection_rank"] = idx
            selected_rows.append(row)

    scored_by_crop = {row["crop_path"]: row for row in merged_rows}
    baseline_scored_rows = []
    for row in baseline_rows:
        scored = dict(scored_by_crop[row["crop_path"]])
        scored["selected_path"] = row["selected_path"]
        baseline_scored_rows.append(scored)

    comparison_rows = []
    report_processes = {}
    for process in processes:
        base = [row for row in baseline_scored_rows if row["process"] == process]
        selected = [row for row in selected_rows if row["process"] == process]
        summary = summarize_process(process, base, selected)
        comparison_rows.append(summary)
        report_processes[process] = summary

    valid_scores = []
    model.eval()
    with torch.no_grad():
        valid_scores = torch.sigmoid(model(torch.from_numpy(valid_images))).detach().cpu().numpy()
    validation = {
        "auc": auc_score(valid_labels, valid_scores),
        "accuracy": float(((valid_scores >= 0.5).astype(np.float32) == valid_labels).mean()),
        "real_train_count": len(real_train),
        "real_valid_count": len(real_valid),
        "fake_train_count": len(fake_train),
        "fake_valid_count": len(fake_valid),
        "image_size": args.image_size,
        "epochs": args.epochs,
    }

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
    generated_separable = bool(validation["auc"] >= 0.80)
    gate = {
        "discriminator_validation_auc": validation["auc"],
        "discriminator_validation_accuracy": validation["accuracy"],
        "generated_pool_visually_separable_from_real": generated_separable,
        "p9_visual_improvement_supported": p9_relative_improved,
        "multi_process_visual_improvement_supported": multi_relative_improved,
        "relative_improved_processes": relative_improved_processes,
        "p9_visual_realism_claim_supported": p9_absolute_supported,
        "multi_process_visual_realism_claim_supported": multi_absolute_supported,
        "absolute_supported_processes": absolute_supported_processes,
        "recommendation": (
            "T-008 supports a stronger visual selection figure, but not a full visual-realism claim."
            if (p9_relative_improved or multi_relative_improved) and not (p9_absolute_supported and multi_absolute_supported)
            else (
                "T-008 supports strengthening the visual-realism claim."
                if p9_absolute_supported and multi_absolute_supported
                else "T-008 does not rescue visual realism; generator-side training or discriminator-guided generation is required."
            )
        ),
    }

    report = {
        "method": "Small CNN discriminator reranking of the existing T-005 LDM candidate pool.",
        "validation": validation,
        "processes": report_processes,
        "gate": gate,
        "notes": [
            "The discriminator score is used only for candidate selection, not as sole evidence of visual realism.",
            "Handcrafted T-007 texture/spectrum ratios remain the guardrail for manuscript claims.",
            "If the discriminator can easily separate generated from real images, the current candidate pool is visually off-manifold.",
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
        "visual_raw_score",
        "discriminator_real_probability",
        "discriminator_rank",
        "discriminator_rerank_score",
        "critical_leakage",
        "appearance_ratio",
        "block_texture_ratio",
        "gradient_hist_ratio",
        "lbp_texture_ratio",
        "spectrum_ratio",
        "full_ratio",
        "nearest_real_path_full",
    ]
    selected_fields = score_fields + ["selection_rank", "selected_path"]
    comparison_fields = [
        "process",
        "baseline_weighted_relative_error_median",
        "t008_weighted_relative_error_median",
        "baseline_full_ratio_median",
        "t008_full_ratio_median",
        "full_ratio_improvement_fraction",
        "baseline_lbp_texture_ratio_median",
        "t008_lbp_texture_ratio_median",
        "lbp_texture_improvement_fraction",
        "baseline_spectrum_ratio_median",
        "t008_spectrum_ratio_median",
        "spectrum_improvement_fraction",
        "t008_block_texture_ratio_median",
        "t008_discriminator_real_probability_median",
        "t008_leakage_fraction",
        "relative_visual_improved",
        "absolute_visual_realism_supported",
    ]
    history_fields = ["epoch", "train_loss", "valid_loss", "valid_auc", "valid_accuracy"]

    write_csv(args.out_root / "discriminator_scores.csv", merged_rows, score_fields)
    write_csv(args.out_root / "t008_selected_samples.csv", selected_rows, selected_fields)
    write_csv(args.out_root / "process_comparison.csv", comparison_rows, comparison_fields)
    write_csv(args.out_root / "training_history.csv", history, history_fields)
    make_selected_grid(selected_rows, args.out_root / "t008_selected_grid.png")
    make_before_after_grid(baseline_scored_rows, selected_rows, args.out_root / "t008_before_after_grid.png")
    make_nearest_grid(selected_rows, args.out_root / "t008_nearest_grid.png")
    torch.save(model.state_dict(), args.out_root / "small_discriminator_state.pt")
    (args.out_root / "t008_discriminator_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
