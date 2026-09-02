#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from diffusers import DDIMScheduler, UNet2DModel

from microstructure_e1_cae import build_spatial_split, list_images
from microstructure_e2_diagnostics import DESCRIPTOR_NAMES, morphology_descriptors
from microstructure_e2_ldm import (
    build_noise_scheduler,
    denormalize_latent,
    load_cae,
    normalize_latent,
    seed_all,
    update_ema,
)


PROCESS_PARAMETERS = {
    "4": (600.0, 1.80, 30.0),
    "5": (450.0, 4.65, 20.0),
    "7": (375.0, 5.60, 40.0),
    "9": (300.0, 7.40, 10.0),
}
PROCESS_FEATURE_NAMES = ("laser_power", "scan_speed", "time", "linear_energy")


def process_id(path):
    value = Path(path).stem.split(".", 1)[0]
    if value not in PROCESS_PARAMETERS:
        raise ValueError(f"No process parameters defined for patch: {path}")
    return value


def process_features(paths, device):
    rows = []
    for path in paths:
        power, speed, duration = PROCESS_PARAMETERS[process_id(path)]
        rows.append((power, speed, duration, power / speed))
    return torch.tensor(rows, dtype=torch.float32, device=device)


class ConditionalPatchDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor / 127.5 - 1.0, path


class ConditionedUNet(nn.Module):
    def __init__(self, latent_channels, condition_dim):
        super().__init__()
        self.backbone = UNet2DModel(
            sample_size=64,
            in_channels=latent_channels,
            out_channels=latent_channels,
            layers_per_block=2,
            block_out_channels=(128, 192, 256, 320),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            attention_head_dim=8,
            norm_num_groups=32,
            class_embed_type="identity",
        )
        time_embedding_dim = 128 * 4
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dim + 1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        nn.init.zeros_(self.condition_projection[-1].weight)
        nn.init.zeros_(self.condition_projection[-1].bias)

    def forward(self, sample, timestep, condition, present):
        projection_input = torch.cat((condition, present[:, None]), dim=1)
        class_embedding = self.condition_projection(projection_input)
        return self.backbone(
            sample,
            timestep,
            class_labels=class_embedding,
        ).sample


def tensor_to_pil(x):
    array = ((x.detach().clamp(-1, 1) + 1.0) * 127.5)
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_process_grid(images, labels, path, columns_per_process):
    processes = sorted(set(labels), key=int)
    header = 30
    canvas = Image.new(
        "RGB",
        (columns_per_process * 512, len(processes) * (512 + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, process in enumerate(processes):
        row_images = [
            images[index] for index, label in enumerate(labels) if label == process
        ]
        y = row * (512 + header)
        draw.text((8, y + 7), f"Process {process}", fill="black")
        for column, image in enumerate(row_images[:columns_per_process]):
            canvas.paste(tensor_to_pil(image), (column * 512, y + header))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


@torch.no_grad()
def compute_condition_statistics(loader, device):
    conditions = []
    labels = []
    for images, paths in loader:
        images = images.to(device, non_blocking=True)
        descriptors = morphology_descriptors(images)
        process = process_features(paths, device)
        conditions.append(torch.cat((descriptors, process), dim=1).cpu())
        labels.extend(process_id(path) for path in paths)

    raw = torch.cat(conditions)
    mean = raw.mean(dim=0)
    std = raw.std(dim=0).clamp_min(1e-6)
    descriptor_centroids = {}
    for process in sorted(set(labels), key=int):
        indices = [index for index, label in enumerate(labels) if label == process]
        descriptor_centroids[process] = raw[indices, : len(DESCRIPTOR_NAMES)].mean(
            dim=0
        )
    return raw, mean, std, descriptor_centroids


def normalize_condition(raw, mean, std):
    return (raw - mean[None]) / std[None]


def build_process_conditions(processes, descriptor_centroids, mean, std, device):
    rows = []
    for process in processes:
        descriptor = descriptor_centroids[process].to(device)
        process_row = process_features([f"{process}.placeholder.png"], device)[0]
        rows.append(torch.cat((descriptor, process_row)))
    raw = torch.stack(rows)
    return normalize_condition(raw, mean, std)


@torch.no_grad()
def evaluate_denoising(
    cae,
    model,
    loader,
    scheduler,
    latent_mean,
    latent_std,
    condition_mean,
    condition_std,
    device,
    max_images,
    seed,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    count = 0
    for images, paths in loader:
        images = images.to(device, non_blocking=True)
        if count + images.shape[0] > max_images:
            keep = max_images - count
            images = images[:keep]
            paths = paths[:keep]
        latent = normalize_latent(cae.encode(images).float(), latent_mean, latent_std)
        raw_condition = torch.cat(
            (morphology_descriptors(images), process_features(paths, device)),
            dim=1,
        )
        condition = normalize_condition(
            raw_condition,
            condition_mean,
            condition_std,
        )
        noise = torch.randn(
            latent.shape,
            generator=generator,
            device=device,
            dtype=latent.dtype,
        )
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (latent.shape[0],),
            generator=generator,
            device=device,
        ).long()
        noisy = scheduler.add_noise(latent, noise, timesteps)
        prediction = model(
            noisy,
            timesteps,
            condition,
            torch.ones(latent.shape[0], device=device),
        )
        loss = F.mse_loss(prediction.float(), noise.float())
        total += float(loss.item()) * latent.shape[0]
        count += latent.shape[0]
        if count >= max_images:
            break
    model.train()
    return total / count


@torch.no_grad()
def generate_conditioned(
    cae,
    model,
    scheduler,
    latent_mean,
    latent_std,
    conditions,
    device,
    infer_steps,
    guidance_scale,
    seed,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        (conditions.shape[0], latent_mean.numel(), 64, 64),
        generator=generator,
        device=device,
    )
    sampler = DDIMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(infer_steps, device=device)
    present = torch.ones(conditions.shape[0], device=device)
    absent = torch.zeros(conditions.shape[0], device=device)
    null_condition = torch.zeros_like(conditions)

    for timestep in sampler.timesteps:
        unconditional = model(latent, timestep, null_condition, absent)
        conditional = model(latent, timestep, conditions, present)
        prediction = unconditional + guidance_scale * (conditional - unconditional)
        latent = sampler.step(
            prediction,
            timestep,
            latent,
            eta=0.0,
            generator=generator,
        ).prev_sample

    latent = denormalize_latent(latent, latent_mean, latent_std)
    decoded = []
    for start in range(0, latent.shape[0], 4):
        decoded.append(cae.decode(latent[start : start + 4]).cpu())
    model.train()
    return torch.cat(decoded)


def conditional_sample_metrics(images, labels, descriptor_centroids, descriptor_std):
    generated = []
    for start in range(0, images.shape[0], 4):
        generated.append(morphology_descriptors(images[start : start + 4].cuda()).cpu())
    generated = torch.cat(generated)
    per_process = {}
    all_absolute_z = []
    for process in sorted(set(labels), key=int):
        indices = [index for index, label in enumerate(labels) if label == process]
        target = descriptor_centroids[process]
        absolute_z = torch.abs(
            (generated[indices] - target[None]) / descriptor_std[None]
        )
        all_absolute_z.append(absolute_z)
        per_process[process] = {
            "mean_absolute_descriptor_z": float(absolute_z.mean().item()),
            "generated_descriptor_mean": {
                name: float(generated[indices, column].mean().item())
                for column, name in enumerate(DESCRIPTOR_NAMES)
            },
            "target_descriptor_mean": {
                name: float(target[column].item())
                for column, name in enumerate(DESCRIPTOR_NAMES)
            },
        }
    all_absolute_z = torch.cat(all_absolute_z)
    return {
        "mean_absolute_descriptor_z": float(all_absolute_z.mean().item()),
        "fraction_absolute_z_gt_2": float(
            (all_absolute_z > 2).float().mean().item()
        ),
        "per_process": per_process,
    }


def save_checkpoint(
    path,
    model,
    ema_model,
    optimizer,
    scaler,
    step,
    best_val,
    latent_mean,
    latent_std,
    condition_mean,
    condition_std,
    descriptor_centroids,
    args,
):
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val": best_val,
            "latent_mean": latent_mean.detach().cpu(),
            "latent_std": latent_std.detach().cpu(),
            "condition_mean": condition_mean.detach().cpu(),
            "condition_std": condition_std.detach().cpu(),
            "descriptor_centroids": {
                key: value.cpu() for key, value in descriptor_centroids.items()
            },
            "descriptor_names": DESCRIPTOR_NAMES,
            "process_feature_names": PROCESS_FEATURE_NAMES,
            "process_parameters": PROCESS_PARAMETERS,
            "args": vars(args),
        },
        path,
    )


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E3 training requires CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)

    paths = list_images(args.data_root)
    train_paths, val_paths, ignored_paths, source_counts = build_spatial_split(
        paths,
        train_max=args.train_max_x,
        val_min=args.val_min_x,
    )
    train_dataset = ConditionalPatchDataset(train_paths)
    process_counts = Counter(process_id(path) for path in train_paths)
    sample_weights = [
        1.0 / process_counts[process_id(path)] for path in train_paths
    ]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_paths),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    stats_loader = DataLoader(
        ConditionalPatchDataset(train_paths),
        batch_size=args.stats_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ConditionalPatchDataset(val_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    e2_checkpoint = torch.load(args.e2_checkpoint, map_location="cpu")
    latent_mean = e2_checkpoint["latent_mean"].float().to(device)
    latent_std = e2_checkpoint["latent_std"].float().to(device)

    _, condition_mean_cpu, condition_std_cpu, descriptor_centroids = (
        compute_condition_statistics(stats_loader, device)
    )
    condition_mean = condition_mean_cpu.to(device)
    condition_std = condition_std_cpu.to(device)
    condition_dim = len(DESCRIPTOR_NAMES) + len(PROCESS_FEATURE_NAMES)
    model = ConditionedUNet(latent_channels, condition_dim).to(device)
    model.backbone.load_state_dict(e2_checkpoint["ema"], strict=True)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)

    scheduler = build_noise_scheduler()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    metrics_path = os.path.join(args.out_root, "metrics.jsonl")
    last_path = os.path.join(args.out_root, "conditional_last.pt")
    best_path = os.path.join(args.out_root, "conditional_best.pt")
    step = 0
    best_val = float("inf")

    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema_model.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        best_val = float(checkpoint["best_val"])
        print(f"[E3] resumed from step={step}", flush=True)

    with open(os.path.join(args.out_root, "condition_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "descriptor_names": DESCRIPTOR_NAMES,
                "process_feature_names": PROCESS_FEATURE_NAMES,
                "process_parameters": PROCESS_PARAMETERS,
                "condition_mean": condition_mean_cpu.tolist(),
                "condition_std": condition_std_cpu.tolist(),
                "descriptor_centroids": {
                    key: value.tolist()
                    for key, value in descriptor_centroids.items()
                },
                "process_counts": process_counts,
                "source_counts": source_counts,
                "ignored": len(ignored_paths),
            },
            handle,
            indent=2,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[E3] train={len(train_paths)} val={len(val_paths)} ignored={len(ignored_paths)} "
        f"process_counts={dict(process_counts)} balanced_sampling=True",
        flush=True,
    )
    print(
        f"[E3] parameters={parameter_count:,} condition_dim={condition_dim} "
        f"cfg_dropout={args.condition_dropout} initialized_from={args.e2_checkpoint}",
        flush=True,
    )

    iterator = iter(train_loader)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()

    while step < args.max_steps:
        try:
            images, batch_paths = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, batch_paths = next(iterator)
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            latent = normalize_latent(
                cae.encode(images).float(),
                latent_mean,
                latent_std,
            )
            raw_condition = torch.cat(
                (
                    morphology_descriptors(images),
                    process_features(batch_paths, device),
                ),
                dim=1,
            )
            condition = normalize_condition(
                raw_condition,
                condition_mean,
                condition_std,
            )
            present = (
                torch.rand(images.shape[0], device=device)
                >= args.condition_dropout
            ).float()
            condition = condition * present[:, None]

        noise = torch.randn_like(latent)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (latent.shape[0],),
            device=device,
        ).long()
        noisy = scheduler.add_noise(latent, noise, timesteps)
        with torch.amp.autocast("cuda", enabled=args.amp):
            prediction = model(noisy, timesteps, condition, present)
            loss = F.mse_loss(prediction.float(), noise.float()) / args.grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema_model, model, args.ema_decay)

        if step % args.log_every == 0:
            print(
                f"[E3] step={step}/{args.max_steps} "
                f"noise_mse={loss.item() * args.grad_accum:.6f} "
                f"condition_present={present.mean().item():.3f} "
                f"time={(time.time() - started) / 60:.1f}m",
                flush=True,
            )

        if step > 0 and step % args.val_every == 0:
            val_loss = evaluate_denoising(
                cae,
                ema_model,
                val_loader,
                scheduler,
                latent_mean,
                latent_std,
                condition_mean,
                condition_std,
                device,
                args.val_images,
                args.seed + 809,
            )
            record = {"step": step, "val_conditional_noise_mse": val_loss}
            append_jsonl(metrics_path, record)
            print(f"[E3][VAL] {json.dumps(record)}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_path,
                    model,
                    ema_model,
                    optimizer,
                    scaler,
                    step,
                    best_val,
                    latent_mean,
                    latent_std,
                    condition_mean,
                    condition_std,
                    descriptor_centroids,
                    args,
                )
                print(f"[E3] saved best checkpoint val={best_val:.6f}", flush=True)

        if step > 0 and step % args.sample_every == 0:
            labels = []
            for process in sorted(PROCESS_PARAMETERS, key=int):
                labels.extend([process] * args.samples_per_process)
            conditions = build_process_conditions(
                labels,
                descriptor_centroids,
                condition_mean,
                condition_std,
                device,
            )
            images = generate_conditioned(
                cae,
                ema_model,
                scheduler,
                latent_mean,
                latent_std,
                conditions,
                device,
                args.infer_steps,
                args.guidance_scale,
                args.sample_seed,
            )
            path = os.path.join(args.out_root, f"conditional_step{step:06d}.png")
            save_process_grid(images, labels, path, args.samples_per_process)
            sample_metrics = conditional_sample_metrics(
                images,
                labels,
                descriptor_centroids,
                condition_std_cpu[: len(DESCRIPTOR_NAMES)],
            )
            sample_metrics["step"] = step
            append_jsonl(
                os.path.join(args.out_root, "sample_metrics.jsonl"),
                sample_metrics,
            )
            print(
                f"[E3][SAMPLE] mean_abs_descriptor_z="
                f"{sample_metrics['mean_absolute_descriptor_z']:.4f}",
                flush=True,
            )

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(
                last_path,
                model,
                ema_model,
                optimizer,
                scaler,
                step,
                best_val,
                latent_mean,
                latent_std,
                condition_mean,
                condition_std,
                descriptor_centroids,
                args,
            )
            print(f"[E3] saved {last_path}", flush=True)
        step += 1

    labels = []
    for process in sorted(PROCESS_PARAMETERS, key=int):
        labels.extend([process] * args.samples_per_process)
    conditions = build_process_conditions(
        labels,
        descriptor_centroids,
        condition_mean,
        condition_std,
        device,
    )
    images = generate_conditioned(
        cae,
        ema_model,
        scheduler,
        latent_mean,
        latent_std,
        conditions,
        device,
        args.infer_steps,
        args.guidance_scale,
        args.sample_seed,
    )
    save_process_grid(
        images,
        labels,
        os.path.join(args.out_root, "conditional_final.png"),
        args.samples_per_process,
    )
    final_sample_metrics = conditional_sample_metrics(
        images,
        labels,
        descriptor_centroids,
        condition_std_cpu[: len(DESCRIPTOR_NAMES)],
    )
    final_sample_metrics.update({"step": step, "final": True})
    append_jsonl(
        os.path.join(args.out_root, "sample_metrics.jsonl"),
        final_sample_metrics,
    )
    save_checkpoint(
        last_path,
        model,
        ema_model,
        optimizer,
        scaler,
        step,
        best_val,
        latent_mean,
        latent_std,
        condition_mean,
        condition_std,
        descriptor_centroids,
        args,
    )
    print(f"[E3] completed step={step} best_val={best_val:.6f}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--cae_checkpoint", required=True)
    parser.add_argument("--e2_checkpoint", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--train_max_x", type=float, default=0.60)
    parser.add_argument("--val_min_x", type=float, default=0.80)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--stats_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_images", type=int, default=128)
    parser.add_argument("--sample_every", type=int, default=2000)
    parser.add_argument("--samples_per_process", type=int, default=4)
    parser.add_argument("--sample_seed", type=int, default=3030)
    parser.add_argument("--infer_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
