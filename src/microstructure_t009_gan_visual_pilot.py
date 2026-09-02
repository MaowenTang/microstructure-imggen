#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset

from microstructure_e14_descriptor_audit import compute_descriptors
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
DESCRIPTOR_VALUE_FIELDS = (
    "spectral_peak_k",
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


class DescriptorArgs:
    analysis_size = 256
    min_spacing = 16.0
    max_spacing = 160.0


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def median(rows, key):
    values = [parse_float(row[key]) for row in rows if key in row and row[key] != ""]
    return float(np.median(values)) if values else float("nan")


def safe_improvement(before, after):
    if not math.isfinite(before) or abs(before) < 1e-12:
        return 0.0
    return float((before - after) / before)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RealPatchDataset(Dataset):
    def __init__(self, rows, process_to_index, image_size, flip=True):
        self.rows = list(rows)
        self.process_to_index = dict(process_to_index)
        self.image_size = image_size
        self.flip = flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["path"]).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        if self.flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        label = self.process_to_index[row["process"]]
        return tensor, torch.tensor(label, dtype=torch.long)


def g_channels(resolution, base):
    if resolution <= 8:
        return base * 16
    if resolution == 16:
        return base * 8
    if resolution == 32:
        return base * 4
    if resolution == 64:
        return base * 2
    return base


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalGenerator(nn.Module):
    def __init__(self, image_size, latent_dim, process_count, base_channels):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(process_count, latent_dim)
        channels_4 = g_channels(4, base_channels)
        self.fc = nn.Linear(latent_dim * 2, channels_4 * 4 * 4)
        blocks = []
        in_channels = channels_4
        resolution = 4
        while resolution < image_size:
            out_channels = g_channels(resolution * 2, base_channels)
            blocks.append(GenBlock(in_channels, out_channels))
            in_channels = out_channels
            resolution *= 2
        self.blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        conditioning = self.label_embedding(labels)
        x = torch.cat([z, conditioning], dim=1)
        x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.blocks(x)
        return self.to_rgb(x)


def d_channels(resolution, base):
    if resolution >= 256:
        return base
    if resolution == 128:
        return base * 2
    if resolution == 64:
        return base * 4
    if resolution == 32:
        return base * 8
    return base * 16


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionDiscriminator(nn.Module):
    def __init__(self, image_size, process_count, base_channels):
        super().__init__()
        blocks = []
        in_channels = 3
        resolution = image_size
        while resolution > 4:
            out_channels = d_channels(resolution, base_channels)
            blocks.append(DiscBlock(in_channels, out_channels))
            in_channels = out_channels
            resolution //= 2
        self.blocks = nn.Sequential(*blocks)
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Flatten(),
        )
        feature_dim = in_channels * 4 * 4
        self.linear = spectral_norm(nn.Linear(feature_dim, 1))
        self.label_embedding = nn.Embedding(process_count, feature_dim)

    def forward(self, x, labels):
        features = self.final(self.blocks(x))
        unconditional = self.linear(features).squeeze(1)
        projection = (features * self.label_embedding(labels)).sum(dim=1) / math.sqrt(features.shape[1])
        return unconditional + projection


def tensor_to_image(tensor, output_size):
    array = ((tensor.detach().clamp(-1, 1).cpu() + 1.0) * 127.5).to(torch.uint8)
    image = Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")
    if image.size != (output_size, output_size):
        image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)
    return image


@torch.no_grad()
def generate_samples(generator, processes, process_to_index, latent_dim, device, count_per_process, seed, output_size, out_root):
    generator.eval()
    generator_root = out_root / "generated"
    generator_root.mkdir(parents=True, exist_ok=True)
    rows = []
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    for process in processes:
        label_value = process_to_index[process]
        labels = torch.full((count_per_process,), label_value, dtype=torch.long, device=device)
        z = torch.randn((count_per_process, latent_dim), generator=torch_generator, device=device)
        images = generator(z, labels)
        process_root = generator_root / f"process_{process}"
        process_root.mkdir(parents=True, exist_ok=True)
        for index in range(count_per_process):
            path = process_root / f"t009_p{process}_sample_{index:02d}.png"
            tensor_to_image(images[index], output_size).save(path)
            rows.append(
                {
                    "process": process,
                    "sample_index": index,
                    "path": str(path),
                }
            )
    generator.train()
    return rows


def target_for_process(target_row, summary_row):
    return {
        "spectral_spacing_px": parse_float(target_row["target_spectral_spacing_px"]),
        "gradient_mean": parse_float(target_row["target_gradient_mean"]),
        "sobel_top10_mean": parse_float(target_row["target_sobel_top10_mean"]),
        "low_frequency_contrast": parse_float(target_row["target_low_frequency_contrast"]),
        "dark_fraction": parse_float(target_row["target_dark_fraction"]),
        "intensity_mean": parse_float(summary_row["intensity_mean_median"]),
    }


def descriptor_error(descriptor, target):
    weights = {
        "spectral_spacing_px": 1.3,
        "gradient_mean": 1.0,
        "sobel_top10_mean": 1.0,
        "low_frequency_contrast": 1.0,
        "dark_fraction": 0.6,
        "intensity_mean": 0.5,
    }
    output = {}
    total = 0.0
    score = 0.0
    for name, weight in weights.items():
        observed = float(descriptor[name])
        desired = float(target[name])
        scale = max(abs(desired), 0.05)
        relative = abs(observed - desired) / scale
        output[f"{name}_relative_error"] = relative
        score += weight * relative
        total += weight
    output["weighted_relative_error"] = score / total
    return output


def build_real_records(rows):
    return [{"path": row["path"], "features": feature_groups(row["path"])} for row in rows]


def visual_metrics_for_process(process, generated_rows, real_rows, max_real_per_process):
    real_subset = choose_real_rows(real_rows, process, max_real_per_process)
    reference_rows, holdout_rows = split_reference_holdout(real_subset)
    reference_records = build_real_records(reference_rows)
    holdout_records = build_real_records(holdout_rows)
    generated_records = [
        {"path": row["path"], "features": feature_groups(row["path"]), "row": row}
        for row in generated_rows
    ]

    for group in FEATURE_GROUPS:
        reference = stack_features(reference_records, group)
        holdout = stack_features(holdout_records, group)
        generated = np.stack([record["features"][group] for record in generated_records]).astype(np.float32)
        reference_z, holdout_z, generated_z = standardize(reference, reference, holdout, generated)
        holdout_dist, _ = nearest(holdout_z, reference_z)
        generated_dist, generated_nn = nearest(generated_z, reference_z)
        holdout_median = float(np.median(holdout_dist))
        leakage_threshold = 0.35 * holdout_median
        for index, record in enumerate(generated_records):
            row = record["row"]
            distance = float(generated_dist[index])
            row[f"{group}_nn_distance"] = distance
            row[f"{group}_ratio"] = float(distance / max(holdout_median, 1e-8))
            row[f"{group}_potential_leakage"] = bool(distance < leakage_threshold)
            row[f"nearest_real_path_{group}"] = reference_records[int(generated_nn[index])]["path"]
            row[f"{group}_real_holdout_nn_median"] = holdout_median
    for row in generated_rows:
        row["critical_leakage"] = bool(any(row[f"{group}_potential_leakage"] for group in CRITICAL_GROUPS))
    return generated_rows


def summarize_process(process, rows, baseline):
    summary = {
        "process": process,
        "generated_count": len(rows),
        "weighted_relative_error_median": median(rows, "weighted_relative_error"),
        "spectral_spacing_relative_error_median": median(rows, "spectral_spacing_px_relative_error"),
        "full_ratio_median": median(rows, "full_ratio"),
        "lbp_texture_ratio_median": median(rows, "lbp_texture_ratio"),
        "spectrum_ratio_median": median(rows, "spectrum_ratio"),
        "block_texture_ratio_median": median(rows, "block_texture_ratio"),
        "appearance_ratio_median": median(rows, "appearance_ratio"),
        "leakage_fraction": float(np.mean([bool(row["critical_leakage"]) for row in rows])),
    }
    baseline_full = parse_float(baseline.get("full_ratio", "nan"))
    baseline_lbp = parse_float(baseline.get("lbp_texture_ratio", "nan"))
    baseline_spectrum = parse_float(baseline.get("spectrum_ratio", "nan"))
    summary["baseline_full_ratio"] = baseline_full
    summary["baseline_lbp_texture_ratio"] = baseline_lbp
    summary["baseline_spectrum_ratio"] = baseline_spectrum
    summary["full_ratio_improvement_fraction"] = safe_improvement(baseline_full, summary["full_ratio_median"])
    summary["lbp_texture_improvement_fraction"] = safe_improvement(
        baseline_lbp, summary["lbp_texture_ratio_median"]
    )
    summary["spectrum_improvement_fraction"] = safe_improvement(
        baseline_spectrum, summary["spectrum_ratio_median"]
    )
    summary["relative_visual_improved"] = bool(
        summary["full_ratio_improvement_fraction"] >= 0.20
        and (
            summary["lbp_texture_improvement_fraction"] >= 0.10
            or summary["spectrum_improvement_fraction"] >= 0.10
        )
        and summary["leakage_fraction"] <= 0.25
    )
    summary["absolute_visual_realism_supported"] = bool(
        summary["full_ratio_median"] <= 2.0
        and summary["lbp_texture_ratio_median"] <= 2.0
        and summary["spectrum_ratio_median"] <= 2.0
        and summary["block_texture_ratio_median"] <= 2.0
        and summary["leakage_fraction"] <= 0.25
    )
    summary["descriptor_alignment_supported"] = bool(
        summary["weighted_relative_error_median"] <= 0.75
        and summary["spectral_spacing_relative_error_median"] <= 0.75
    )
    return summary


def make_generated_grid(rows, path):
    columns = 4
    cell = 180
    header = 48
    canvas_rows = int(math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * cell, canvas_rows * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        x = (index % columns) * cell
        y = (index // columns) * (cell + header)
        label = (
            f"P{row['process']} d={parse_float(row['weighted_relative_error']):.2f} "
            f"f={parse_float(row['full_ratio']):.1f}"
        )
        draw.text((x + 4, y + 7), label, fill="black")
        image = Image.open(row["path"]).convert("RGB").resize((cell, cell))
        canvas.paste(image, (x, y + header))
    canvas.save(path)


def make_nearest_grid(rows, path, max_rows=16):
    rows = rows[:max_rows]
    cell = 170
    header = 50
    canvas = Image.new("RGB", (2 * cell, len(rows) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        y = index * (cell + header)
        label = (
            f"P{row['process']} full={parse_float(row['full_ratio']):.1f} "
            f"lbp={parse_float(row['lbp_texture_ratio']):.1f}"
        )
        draw.text((4, y + 6), label, fill="black")
        generated = Image.open(row["path"]).convert("RGB").resize((cell, cell))
        real = Image.open(row["nearest_real_path_full"]).convert("RGB").resize((cell, cell))
        canvas.paste(generated, (0, y + header))
        canvas.paste(real, (cell, y + header))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t006_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--samples_per_process", type=int, default=8)
    parser.add_argument("--max_real_per_process", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    real_rows = read_csv(args.t002_root / "patch_descriptors.csv")
    target_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_descriptor_targets.csv")}
    summary_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_summary.csv")}
    baseline_rows = {row["process"]: row for row in read_csv(args.t006_root / "visual_realism_summary.csv")}
    processes = sorted({row["process"] for row in real_rows}, key=int)
    process_to_index = {process: index for index, process in enumerate(processes)}

    dataset = RealPatchDataset(real_rows, process_to_index, args.image_size, flip=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    iterator = iter(loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = ConditionalGenerator(
        args.image_size,
        args.latent_dim,
        len(processes),
        args.base_channels,
    ).to(device)
    discriminator = ProjectionDiscriminator(
        args.image_size,
        len(processes),
        args.base_channels,
    ).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.999))
    scaler_g = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    scaler_d = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    history = []
    for step in range(1, args.max_steps + 1):
        try:
            real, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            real, labels = next(iterator)
        real = real.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        z = torch.randn((real.shape[0], args.latent_dim), device=device)

        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            fake = generator(z, labels).detach()
            real_score = discriminator(real, labels)
            fake_score = discriminator(fake, labels)
            loss_d = torch.relu(1.0 - real_score).mean() + torch.relu(1.0 + fake_score).mean()
        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        z = torch.randn((real.shape[0], args.latent_dim), device=device)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            fake = generator(z, labels)
            fake_score = discriminator(fake, labels)
            loss_g = -fake_score.mean()
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            record = {
                "step": step,
                "loss_d": float(loss_d.detach().cpu()),
                "loss_g": float(loss_g.detach().cpu()),
                "real_score_mean": float(real_score.detach().mean().cpu()),
                "fake_score_mean": float(fake_score.detach().mean().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    generated_rows = generate_samples(
        generator,
        processes,
        process_to_index,
        args.latent_dim,
        device,
        args.samples_per_process,
        args.seed + 1000,
        args.output_size,
        args.out_root,
    )

    descriptor_rows = []
    descriptor_args = DescriptorArgs()
    for row in generated_rows:
        raw_descriptor = compute_descriptors(row["path"], descriptor_args)
        descriptor = {field: raw_descriptor[field] for field in DESCRIPTOR_VALUE_FIELDS}
        target = target_for_process(target_rows[row["process"]], summary_rows[row["process"]])
        errors = descriptor_error(descriptor, target)
        row.update(descriptor)
        row.update(errors)
        descriptor_rows.append(dict(row))

    scored_rows = []
    for process in processes:
        process_generated = [row for row in generated_rows if row["process"] == process]
        scored_rows.extend(
            visual_metrics_for_process(process, process_generated, real_rows, args.max_real_per_process)
        )

    process_rows = []
    report_processes = {}
    for process in processes:
        rows = [row for row in scored_rows if row["process"] == process]
        summary = summarize_process(process, rows, baseline_rows[process])
        process_rows.append(summary)
        report_processes[process] = summary

    relative_improved = [row["process"] for row in process_rows if row["relative_visual_improved"]]
    absolute_supported = [row["process"] for row in process_rows if row["absolute_visual_realism_supported"]]
    descriptor_supported = [row["process"] for row in process_rows if row["descriptor_alignment_supported"]]
    p9_visual_supported = bool(report_processes.get("9", {}).get("absolute_visual_realism_supported", False))
    p9_descriptor_supported = bool(report_processes.get("9", {}).get("descriptor_alignment_supported", False))
    multi_visual_supported = len(absolute_supported) >= 3
    multi_descriptor_supported = len(descriptor_supported) >= 3
    gate = {
        "p9_relative_visual_improvement_supported": bool(
            report_processes.get("9", {}).get("relative_visual_improved", False)
        ),
        "multi_process_relative_visual_improvement_supported": len(relative_improved) >= 3,
        "relative_improved_processes": relative_improved,
        "p9_visual_realism_claim_supported": p9_visual_supported,
        "multi_process_visual_realism_claim_supported": multi_visual_supported,
        "absolute_visual_supported_processes": absolute_supported,
        "p9_descriptor_alignment_supported": p9_descriptor_supported,
        "multi_process_descriptor_alignment_supported": multi_descriptor_supported,
        "descriptor_supported_processes": descriptor_supported,
        "recommendation": (
            "T-009 improves visual realism but needs descriptor control before manuscript use."
            if (relative_improved or absolute_supported) and not (p9_descriptor_supported and multi_descriptor_supported)
            else (
                "T-009 supports a stronger generator-side visual branch."
                if p9_visual_supported and multi_visual_supported and p9_descriptor_supported and multi_descriptor_supported
                else "T-009 does not yet support visual-realism claim; continue with stronger StyleGAN2/diffusion training and normalization."
            )
        ),
    }
    report = {
        "method": "Small process-conditioned hinge-GAN trained on real T-002 DED patches.",
        "training": {
            "image_size": args.image_size,
            "output_size": args.output_size,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
            "amp": bool(args.amp),
            "device": str(device),
            "real_patch_count": len(real_rows),
            "processes": processes,
        },
        "processes": report_processes,
        "gate": gate,
        "notes": [
            "This is a generator-side pilot, not a final StyleGAN2-ADA baseline.",
            "Visual support is evaluated with the same T-006 texture/spectrum/leakage guardrails.",
            "Descriptor support is checked separately because adversarial training alone does not guarantee morphology controllability.",
        ],
    }

    descriptor_fields = [
        "process",
        "sample_index",
        "path",
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
    ]
    visual_fields = [
        "process",
        "sample_index",
        "path",
        "weighted_relative_error",
        "spectral_spacing_px_relative_error",
        "appearance_ratio",
        "block_texture_ratio",
        "gradient_hist_ratio",
        "lbp_texture_ratio",
        "spectrum_ratio",
        "full_ratio",
        "critical_leakage",
        "nearest_real_path_full",
    ]
    comparison_fields = [
        "process",
        "generated_count",
        "weighted_relative_error_median",
        "spectral_spacing_relative_error_median",
        "full_ratio_median",
        "lbp_texture_ratio_median",
        "spectrum_ratio_median",
        "block_texture_ratio_median",
        "appearance_ratio_median",
        "leakage_fraction",
        "baseline_full_ratio",
        "baseline_lbp_texture_ratio",
        "baseline_spectrum_ratio",
        "full_ratio_improvement_fraction",
        "lbp_texture_improvement_fraction",
        "spectrum_improvement_fraction",
        "relative_visual_improved",
        "absolute_visual_realism_supported",
        "descriptor_alignment_supported",
    ]
    history_fields = ["step", "loss_d", "loss_g", "real_score_mean", "fake_score_mean"]
    write_csv(args.out_root / "generated_descriptors.csv", descriptor_rows, descriptor_fields)
    write_csv(args.out_root / "generated_visual_summary.csv", scored_rows, visual_fields)
    write_csv(args.out_root / "process_comparison.csv", process_rows, comparison_fields)
    write_csv(args.out_root / "training_history.csv", history, history_fields)
    make_generated_grid(scored_rows, args.out_root / "t009_generated_grid.png")
    make_nearest_grid(scored_rows, args.out_root / "t009_nearest_grid.png")
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "process_to_index": process_to_index,
            "args": vars(args),
        },
        args.out_root / "t009_gan_state.pt",
    )
    (args.out_root / "t009_gan_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
