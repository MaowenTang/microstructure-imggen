#!/usr/bin/env python3

import argparse
import copy
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from microstructure_e14_descriptor_audit import PROCESS_PARAMETERS, compute_descriptors
from microstructure_t006_visual_realism import read_csv, write_csv
from microstructure_t009_gan_visual_pilot import (
    DESCRIPTOR_VALUE_FIELDS,
    DescriptorArgs,
    descriptor_error,
    make_generated_grid,
    make_nearest_grid,
    summarize_process,
    target_for_process,
    visual_metrics_for_process,
)


PROCESS_FEATURE_NAMES = ("laser_power", "scan_speed", "dwell_time", "linear_energy")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def process_feature_row(process):
    values = PROCESS_PARAMETERS[str(process)]
    laser_power = float(values["laser_power"])
    scan_speed = float(values["scan_speed"])
    dwell_time = float(values["dwell_time"])
    return np.asarray(
        [laser_power, scan_speed, dwell_time, laser_power / scan_speed],
        dtype=np.float32,
    )


def process_feature_stats(processes):
    rows = np.stack([process_feature_row(process) for process in processes]).astype(np.float32)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalized_process_features(processes, mean, std, device):
    rows = np.stack([process_feature_row(process) for process in processes]).astype(np.float32)
    rows = (rows - mean[None, :]) / std[None, :]
    return torch.from_numpy(rows).to(device=device, dtype=torch.float32)


def tensor_to_image(tensor):
    array = ((tensor.detach().clamp(-1, 1).cpu() + 1.0) * 127.5).to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")


def load_image(path, resolution):
    image = Image.open(path).convert("RGB")
    if image.size != (resolution, resolution):
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((resolution, resolution), Image.Resampling.BICUBIC)
    return image


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def select_training_rows(real_rows, processes, top_ratio):
    rows = [row for row in real_rows if row["process"] in processes]
    if top_ratio >= 0.999:
        return rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["process"], row.get("source", ""))].append(row)
    selected = []
    for key in sorted(grouped):
        group = sorted(
            grouped[key],
            key=lambda row: parse_float(row.get("sobel_top10_mean", "0")),
            reverse=True,
        )
        keep = max(1, int(round(len(group) * top_ratio)))
        selected.extend(group[:keep])
    return sorted(selected, key=lambda row: (row["process"], row.get("source", ""), row["path"]))


class ProcessBalancedPatchDataset(Dataset):
    def __init__(self, rows, processes, epoch_size, resolution, horizontal_flip_prob):
        self.rows = list(rows)
        self.processes = list(processes)
        self.epoch_size = int(epoch_size)
        self.resolution = int(resolution)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.by_process = defaultdict(list)
        for row in self.rows:
            self.by_process[row["process"]].append(row)
        missing = [process for process in self.processes if not self.by_process[process]]
        if missing:
            raise RuntimeError(f"Missing training rows for processes: {missing}")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        process = self.processes[index % len(self.processes)]
        row = random.choice(self.by_process[process])
        image = load_image(row["path"], self.resolution)
        if random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image_to_tensor(image), process


class ConditionedUNet(nn.Module):
    def __init__(self, base_model, condition_dim):
        super().__init__()
        cfg = base_model.config
        self.backbone = UNet2DModel(
            sample_size=cfg.sample_size,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            layers_per_block=cfg.layers_per_block,
            block_out_channels=tuple(cfg.block_out_channels),
            down_block_types=tuple(cfg.down_block_types),
            up_block_types=tuple(cfg.up_block_types),
            attention_head_dim=cfg.attention_head_dim,
            norm_num_groups=cfg.norm_num_groups,
            act_fn=cfg.act_fn,
            dropout=cfg.dropout,
            class_embed_type="identity",
        )
        missing, unexpected = self.backbone.load_state_dict(base_model.state_dict(), strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while warm-starting conditioned UNet: {unexpected}")
        self.warm_start_missing = list(missing)
        time_embedding_dim = int(cfg.block_out_channels[0]) * 4
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim + 1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        nn.init.zeros_(self.condition_projection[-1].weight)
        nn.init.zeros_(self.condition_projection[-1].bias)

    def forward(self, sample, timestep, condition, present):
        labels = self.condition_projection(torch.cat((condition, present[:, None]), dim=1))
        return self.backbone(sample, timestep, class_labels=labels).sample


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def save_process_sample_grid(rows, path, cell=180):
    rows = sorted(rows, key=lambda row: (int(row["process"]), int(row["sample_index"])))
    columns = max(1, max(int(row["sample_index"]) for row in rows) + 1)
    processes = sorted({row["process"] for row in rows}, key=int)
    header = 34
    canvas = Image.new("RGB", (columns * cell, len(processes) * (cell + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, process in enumerate(processes):
        draw.text((6, r * (cell + header) + 8), f"Process {process}", fill="black")
    for row in rows:
        r = processes.index(row["process"])
        c = int(row["sample_index"])
        y = r * (cell + header)
        image = Image.open(row["path"]).convert("RGB").resize((cell, cell))
        canvas.paste(image, (c * cell, y + header))
    canvas.save(path)


@torch.no_grad()
def generate_samples(
    model,
    scheduler,
    processes,
    feature_mean,
    feature_std,
    device,
    resolution,
    samples_per_process,
    sample_steps,
    guidance_scale,
    seed,
    out_root,
    amp,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    labels = [process for process in processes for _ in range(samples_per_process)]
    x = torch.randn((len(labels), 3, resolution, resolution), generator=generator, device=device)
    condition = normalized_process_features(labels, feature_mean, feature_std, device)
    present = torch.ones(len(labels), device=device)
    null_condition = torch.zeros_like(condition)
    absent = torch.zeros(len(labels), device=device)

    sampler = DDPMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(sample_steps, device=device)
    amp_enabled = bool(amp) and device.type == "cuda"
    for timestep in sampler.timesteps:
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            if guidance_scale == 1.0:
                prediction = model(x, timestep, condition, present)
            else:
                uncond = model(x, timestep, null_condition, absent)
                cond = model(x, timestep, condition, present)
                prediction = uncond + guidance_scale * (cond - uncond)
        x = sampler.step(prediction, timestep, x, generator=generator).prev_sample

    generated_root = out_root / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, process in enumerate(labels):
        sample_index = index % samples_per_process
        process_root = generated_root / f"process_{process}"
        process_root.mkdir(parents=True, exist_ok=True)
        path = process_root / f"t011_p{process}_sample_{sample_index:02d}.png"
        tensor_to_image(x[index]).save(path)
        rows.append({"process": process, "sample_index": sample_index, "path": str(path)})
    model.train()
    return rows


def save_checkpoint(path, model, ema_model, optimizer, lr_scheduler, scaler, step, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def evaluate_generated(args, generated_rows, training_summary):
    real_rows = read_csv(args.t002_root / "patch_descriptors.csv")
    target_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_descriptor_targets.csv")}
    summary_rows = {row["process"]: row for row in read_csv(args.t002_root / "process_summary.csv")}
    baseline_rows = {row["process"]: row for row in read_csv(args.t006_root / "visual_realism_summary.csv")}
    processes = sorted({row["process"] for row in generated_rows}, key=int)

    descriptor_args = DescriptorArgs()
    descriptor_rows = []
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
            visual_metrics_for_process(
                process,
                process_generated,
                real_rows,
                args.max_real_per_process,
            )
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
    p9 = report_processes.get("9", {})
    gate = {
        "p9_relative_visual_improvement_supported": bool(p9.get("relative_visual_improved", False)),
        "multi_process_relative_visual_improvement_supported": len(relative_improved) >= 3,
        "relative_improved_processes": relative_improved,
        "p9_visual_realism_claim_supported": bool(p9.get("absolute_visual_realism_supported", False)),
        "multi_process_visual_realism_claim_supported": len(absolute_supported) >= 3,
        "absolute_visual_supported_processes": absolute_supported,
        "p9_descriptor_alignment_supported": bool(p9.get("descriptor_alignment_supported", False)),
        "multi_process_descriptor_alignment_supported": len(descriptor_supported) >= 3,
        "descriptor_supported_processes": descriptor_supported,
        "recommendation": (
            "T-011 is a promising conditioned diffusion branch; extend training and tune texture loss/selection."
            if relative_improved
            else "T-011 does not improve visual realism; revise conditioning/data normalization before scaling."
        ),
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
    write_csv(args.out_root / "generated_descriptors.csv", descriptor_rows, descriptor_fields)
    write_csv(args.out_root / "generated_visual_summary.csv", scored_rows, visual_fields)
    write_csv(args.out_root / "process_comparison.csv", process_rows, comparison_fields)
    write_csv(
        args.out_root / "training_history.csv",
        training_summary["history"],
        ["step", "loss", "lr", "time_min"],
    )
    make_generated_grid(scored_rows, args.out_root / "t011_generated_grid.png")
    make_nearest_grid(scored_rows, args.out_root / "t011_nearest_grid.png")
    report = {
        "method": "Warm-started RGB DDPM fine-tuned with processing-parameter conditioning.",
        "training": training_summary,
        "processes": report_processes,
        "gate": gate,
        "notes": [
            "Conditioning uses observed processing-parameter vectors; this does not claim interpolation to unseen settings.",
            "The model is warm-started from the stronger unconditional RGB DDPM evaluated in T-010.",
            "Visual support uses the same T-006 texture/spectrum/leakage guardrails.",
        ],
    }
    (args.out_root / "t011_ddpm_report.json").write_text(json.dumps(report, indent=2))
    return report


def train_and_eval(args):
    if not torch.cuda.is_available():
        raise RuntimeError("T-011 conditioned DDPM training requires CUDA on ACES/HPC.")
    args.out_root.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda")

    real_rows = read_csv(args.t002_root / "patch_descriptors.csv")
    processes = sorted({row["process"] for row in real_rows}, key=int)
    if args.processes.lower() != "all":
        requested = sorted({item.strip() for item in args.processes.split(",") if item.strip()}, key=int)
        processes = [process for process in processes if process in requested]
    feature_mean, feature_std = process_feature_stats(processes)
    train_rows = select_training_rows(real_rows, processes, args.top_ratio)
    dataset = ProcessBalancedPatchDataset(
        train_rows,
        processes,
        args.epoch_size,
        args.resolution,
        args.horizontal_flip_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    iterator = iter(loader)

    base_model = UNet2DModel.from_pretrained(args.init_checkpoint)
    model = ConditionedUNet(base_model, condition_dim=len(PROCESS_FEATURE_NAMES)).to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(args.init_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    amp_enabled = bool(args.amp)
    history = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    print(
        json.dumps(
            {
                "event": "start",
                "processes": processes,
                "real_rows": len(real_rows),
                "train_rows": len(train_rows),
                "top_ratio": args.top_ratio,
                "init_checkpoint": str(args.init_checkpoint),
                "warm_start_missing": model.warm_start_missing,
            }
        ),
        flush=True,
    )

    model.train()
    for step in range(1, args.max_steps + 1):
        total_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                images, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                images, labels = next(iterator)
            images = images.to(device, non_blocking=True)
            labels = list(labels)
            condition = normalized_process_features(labels, feature_mean, feature_std, device)
            if args.condition_dropout > 0:
                present = (torch.rand(condition.shape[0], device=device) >= args.condition_dropout).float()
                condition = condition * present[:, None]
            else:
                present = torch.ones(condition.shape[0], device=device)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
            ).long()
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, timesteps)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(noisy, timesteps, condition, present)
                loss = F.mse_loss(prediction.float(), noise.float()) / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        update_ema(ema_model, model, args.ema_decay)

        if step == 1 or step % args.log_every == 0 or step == args.max_steps:
            record = {
                "step": step,
                "loss": total_loss * args.grad_accum,
                "lr": lr_scheduler.get_last_lr()[0],
                "time_min": (time.time() - started) / 60.0,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(args.out_root / "t011_conditioned_ddpm_state.pt", model, ema_model, optimizer, lr_scheduler, scaler, step, args)

    generated_rows = generate_samples(
        ema_model,
        scheduler,
        processes,
        feature_mean,
        feature_std,
        device,
        args.resolution,
        args.samples_per_process,
        args.sample_steps,
        args.guidance_scale,
        args.seed + 1100,
        args.out_root,
        args.amp,
    )
    save_process_sample_grid(generated_rows, args.out_root / "t011_process_samples_grid.png")
    training_summary = {
        "init_checkpoint": str(args.init_checkpoint),
        "resolution": args.resolution,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "condition_dropout": args.condition_dropout,
        "ema_decay": args.ema_decay,
        "top_ratio": args.top_ratio,
        "train_rows": len(train_rows),
        "all_real_rows": len(real_rows),
        "processes": processes,
        "process_feature_names": PROCESS_FEATURE_NAMES,
        "process_feature_mean": feature_mean.tolist(),
        "process_feature_std": feature_std.tolist(),
        "samples_per_process": args.samples_per_process,
        "sample_steps": args.sample_steps,
        "guidance_scale": args.guidance_scale,
        "history": history,
    }
    report = evaluate_generated(args, generated_rows, training_summary)
    print(json.dumps(report["gate"], indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t002_root", required=True, type=Path)
    parser.add_argument("--t006_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--init_checkpoint", required=True, type=Path)
    parser.add_argument("--processes", default="all")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--top_ratio", type=float, default=0.70)
    parser.add_argument("--epoch_size", type=int, default=2800)
    parser.add_argument("--horizontal_flip_prob", type=float, default=0.50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--condition_dropout", type=float, default=0.15)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--samples_per_process", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=150)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--max_real_per_process", type=int, default=500)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main():
    train_and_eval(build_parser().parse_args())


if __name__ == "__main__":
    main()
