#!/usr/bin/env python3

import argparse
import copy
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diffusers import DDIMScheduler

from microstructure_e1_cae import list_images
from microstructure_e2_ldm import (
    build_noise_scheduler,
    denormalize_latent,
    load_cae,
    normalize_latent,
    seed_all,
    update_ema,
)
from microstructure_e3_conditional_ldm import (
    ConditionedUNet,
    PROCESS_FEATURE_NAMES,
    PROCESS_PARAMETERS,
    process_features,
    process_id,
    save_process_grid,
)


PROCESS_IDS = tuple(sorted(PROCESS_PARAMETERS, key=int))


def parse_csv_numbers(value, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def validate_crop_configuration(scales, weights):
    if not scales:
        raise ValueError("At least one crop scale is required.")
    if len(scales) != len(weights):
        raise ValueError("--crop_scales and --crop_weights must have equal length.")
    if any(scale < 512 for scale in scales):
        raise ValueError("Crop scales must be at least 512 pixels.")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Crop weights must be positive.")


def discover_originals(root):
    paths = list_images(os.path.expanduser(root))
    groups = defaultdict(list)
    for path in paths:
        try:
            groups[process_id(path)].append(path)
        except ValueError:
            continue
    missing = [process for process in PROCESS_IDS if not groups[process]]
    if missing:
        raise RuntimeError(f"Missing original images for processes: {missing}")
    return {process: sorted(groups[process]) for process in PROCESS_IDS}


class CachedOriginals:
    def __init__(self):
        self.images = {}

    def get(self, path):
        if path not in self.images:
            with Image.open(path) as image:
                self.images[path] = image.convert("RGB").copy()
        return self.images[path]


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous() / 127.5 - 1.0


class AuthenticMultiScaleDataset(Dataset):
    """Balanced online crops from the untouched microscopy originals."""

    def __init__(
        self,
        groups,
        epoch_size,
        crop_scales,
        crop_weights,
        train_max_x,
        horizontal_flip_prob,
    ):
        self.groups = groups
        self.epoch_size = epoch_size
        self.crop_scales = crop_scales
        weight_sum = sum(crop_weights)
        self.crop_weights = tuple(weight / weight_sum for weight in crop_weights)
        self.train_max_x = train_max_x
        self.horizontal_flip_prob = horizontal_flip_prob
        self.cache = CachedOriginals()

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        process = PROCESS_IDS[index % len(PROCESS_IDS)]
        path = random.choice(self.groups[process])
        image = self.cache.get(path)
        width, height = image.size
        train_width = int(width * self.train_max_x)
        valid_scales = [
            (scale, weight)
            for scale, weight in zip(self.crop_scales, self.crop_weights)
            if scale <= train_width and scale <= height
        ]
        if not valid_scales:
            raise RuntimeError(
                f"No crop scale fits training region for {path}: "
                f"image={image.size}, train_width={train_width}"
            )
        scales, weights = zip(*valid_scales)
        scale = random.choices(scales, weights=weights, k=1)[0]
        x = random.randint(0, train_width - scale)
        y = random.randint(0, height - scale)
        crop = image.crop((x, y, x + scale, y + scale))
        if scale != 512:
            crop = crop.resize((512, 512), Image.Resampling.LANCZOS)
        if random.random() < self.horizontal_flip_prob:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image_to_tensor(crop), process, scale, Path(path).name


class FixedRightHoldoutDataset(Dataset):
    """Fixed right-side crops that never overlap the left training region."""

    def __init__(self, groups, crops_per_source, train_max_x, val_min_center_x):
        self.groups = groups
        self.records = []
        self.cache = CachedOriginals()
        for process in PROCESS_IDS:
            for path in groups[process]:
                with Image.open(path) as image:
                    width, height = image.size
                x = width - 512
                center_fraction = (x + 256) / width
                if x < int(width * train_max_x):
                    raise RuntimeError(f"Train/validation overlap for {path}")
                if center_fraction < val_min_center_x:
                    raise RuntimeError(
                        f"Validation crop center {center_fraction:.3f} is left of "
                        f"{val_min_center_x:.3f} for {path}"
                    )
                max_y = height - 512
                if crops_per_source == 1:
                    positions = [max_y // 2]
                else:
                    positions = [
                        round(index * max_y / (crops_per_source - 1))
                        for index in range(crops_per_source)
                    ]
                for y in sorted(set(positions)):
                    self.records.append((path, process, x, y))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, process, x, y = self.records[index]
        image = self.cache.get(path)
        crop = image.crop((x, y, x + 512, y + 512))
        return image_to_tensor(crop), process, 512, Path(path).name


def process_statistics(device):
    placeholders = [f"{process}.png" for process in PROCESS_IDS]
    raw = process_features(placeholders, device)
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def normalized_process_features(labels, mean, std, device):
    raw = process_features([f"{label}.png" for label in labels], device)
    return (raw - mean[None]) / std[None]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def save_reference_grid(dataset, path, samples_per_process):
    buckets = {process: [] for process in PROCESS_IDS}
    attempts = 0
    while any(len(bucket) < samples_per_process for bucket in buckets.values()):
        image, process, _, _ = dataset[attempts % len(dataset)]
        if len(buckets[process]) < samples_per_process:
            buckets[process].append(image)
        attempts += 1
        if attempts > len(dataset) * 2:
            raise RuntimeError("Could not assemble a balanced reference grid.")
    labels = []
    images = []
    for process in PROCESS_IDS:
        labels.extend([process] * samples_per_process)
        images.extend(buckets[process])
    save_process_grid(torch.stack(images), labels, path, samples_per_process)


@torch.no_grad()
def evaluate_denoising(
    cae,
    model,
    loader,
    scheduler,
    latent_mean,
    latent_std,
    process_mean,
    process_std,
    device,
    max_images,
    seed,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = defaultdict(float)
    counts = defaultdict(int)
    total_images = 0

    for images, labels, _, _ in loader:
        if total_images >= max_images:
            break
        keep = min(images.shape[0], max_images - total_images)
        images = images[:keep].to(device, non_blocking=True)
        labels = list(labels[:keep])
        latent = normalize_latent(cae.encode(images).float(), latent_mean, latent_std)
        condition = normalized_process_features(
            labels, process_mean, process_std, device
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
            (keep,),
            generator=generator,
            device=device,
        ).long()
        noisy = scheduler.add_noise(latent, noise, timesteps)
        prediction = model(
            noisy,
            timesteps,
            condition,
            torch.ones(keep, device=device),
        )
        per_image = F.mse_loss(
            prediction.float(), noise.float(), reduction="none"
        ).flatten(1).mean(dim=1)
        for label, value in zip(labels, per_image):
            totals[label] += float(value.item())
            counts[label] += 1
        total_images += keep

    model.train()
    aggregate = sum(totals.values()) / sum(counts.values())
    return {
        "val_noise_mse": aggregate,
        "images": sum(counts.values()),
        "per_process": {
            process: totals[process] / counts[process]
            for process in PROCESS_IDS
            if counts[process]
        },
    }


@torch.no_grad()
def generate_process_grid(
    cae,
    model,
    scheduler,
    latent_mean,
    latent_std,
    process_mean,
    process_std,
    device,
    samples_per_process,
    infer_steps,
    guidance_scale,
    seed,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    base_noise = torch.randn(
        (samples_per_process, latent_mean.numel(), 64, 64),
        generator=generator,
        device=device,
    )
    latent = base_noise.repeat(len(PROCESS_IDS), 1, 1, 1)
    labels = [
        process
        for process in PROCESS_IDS
        for _ in range(samples_per_process)
    ]
    condition = normalized_process_features(
        labels, process_mean, process_std, device
    )
    present = torch.ones(len(labels), device=device)
    absent = torch.zeros(len(labels), device=device)
    null_condition = torch.zeros_like(condition)
    sampler = DDIMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(infer_steps, device=device)

    for timestep in sampler.timesteps:
        unconditional = model(latent, timestep, null_condition, absent)
        conditional = model(latent, timestep, condition, present)
        prediction = unconditional + guidance_scale * (
            conditional - unconditional
        )
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
    return torch.cat(decoded), labels


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
    process_mean,
    process_std,
    args,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
            "process_mean": process_mean.detach().cpu(),
            "process_std": process_std.detach().cpu(),
            "process_feature_names": PROCESS_FEATURE_NAMES,
            "process_parameters": PROCESS_PARAMETERS,
            "args": vars(args),
        },
        path,
    )


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E6 training requires CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)

    scales = parse_csv_numbers(args.crop_scales, int)
    weights = parse_csv_numbers(args.crop_weights, float)
    validate_crop_configuration(scales, weights)
    groups = discover_originals(args.data_root)
    train_dataset = AuthenticMultiScaleDataset(
        groups,
        args.epoch_size,
        scales,
        weights,
        args.train_max_x,
        args.horizontal_flip_prob,
    )
    val_dataset = FixedRightHoldoutDataset(
        groups,
        args.val_crops_per_source,
        args.train_max_x,
        args.val_min_center_x,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    save_reference_grid(
        train_dataset,
        os.path.join(args.out_root, "real_train_multiscale_reference.png"),
        args.samples_per_process,
    )
    save_reference_grid(
        val_dataset,
        os.path.join(args.out_root, "real_validation_reference.png"),
        args.samples_per_process,
    )

    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    e2_checkpoint = torch.load(args.e2_checkpoint, map_location="cpu")
    latent_mean = e2_checkpoint["latent_mean"].float().to(device)
    latent_std = e2_checkpoint["latent_std"].float().to(device)
    process_mean, process_std = process_statistics(device)

    model = ConditionedUNet(
        latent_channels,
        condition_dim=len(PROCESS_FEATURE_NAMES),
    ).to(device)
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

    last_path = os.path.join(args.out_root, "e6_last.pt")
    best_path = os.path.join(args.out_root, "e6_best.pt")
    metrics_path = os.path.join(args.out_root, "metrics.jsonl")
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
        if not torch.allclose(
            checkpoint["latent_mean"], latent_mean.cpu(), rtol=1e-4, atol=1e-5
        ):
            raise RuntimeError("E2 latent mean differs from the resume checkpoint.")
        if not torch.allclose(
            checkpoint["latent_std"], latent_std.cpu(), rtol=1e-4, atol=1e-5
        ):
            raise RuntimeError("E2 latent std differs from the resume checkpoint.")
        print(f"[E6] resumed from step={step}", flush=True)

    schema = {
        "sources": groups,
        "process_parameters": PROCESS_PARAMETERS,
        "process_feature_names": PROCESS_FEATURE_NAMES,
        "process_mean": process_mean.cpu().tolist(),
        "process_std": process_std.cpu().tolist(),
        "crop_scales": scales,
        "crop_weights": weights,
        "train_max_x": args.train_max_x,
        "validation": "fixed 512 px right-edge crops",
        "validation_crops": len(val_dataset),
        "horizontal_flip_prob": args.horizontal_flip_prob,
        "latent_normalization_source": args.e2_checkpoint,
    }
    with open(
        os.path.join(args.out_root, "data_condition_schema.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(schema, handle, indent=2)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[E6] originals={sum(len(paths) for paths in groups.values())} "
        f"per_process={dict((key, len(value)) for key, value in groups.items())} "
        f"virtual_train={len(train_dataset)} val={len(val_dataset)}",
        flush=True,
    )
    print(
        f"[E6] crop_scales={scales} weights={weights} train_x<=0.."
        f"{args.train_max_x:.2f} val_center_x>={args.val_min_center_x:.2f} "
        f"flip={args.horizontal_flip_prob:.2f}",
        flush=True,
    )
    print(
        f"[E6] parameters={parameter_count:,} process_only_condition=True "
        f"cfg_dropout={args.condition_dropout} initialized_from={args.e2_checkpoint}",
        flush=True,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    started = time.time()

    while step < args.max_steps:
        try:
            images, labels, scales_in_batch, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels, scales_in_batch, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        labels = list(labels)

        with torch.no_grad():
            latent = normalize_latent(
                cae.encode(images).float(),
                latent_mean,
                latent_std,
            )
            condition = normalized_process_features(
                labels, process_mean, process_std, device
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
                f"[E6] step={step}/{args.max_steps} "
                f"noise_mse={loss.item() * args.grad_accum:.6f} "
                f"condition_present={present.mean().item():.3f} "
                f"crop_mean={scales_in_batch.float().mean().item():.1f} "
                f"time={(time.time() - started) / 60:.1f}m",
                flush=True,
            )

        if step > 0 and step % args.val_every == 0:
            metrics = evaluate_denoising(
                cae,
                ema_model,
                val_loader,
                scheduler,
                latent_mean,
                latent_std,
                process_mean,
                process_std,
                device,
                args.val_images,
                args.seed + 1601,
            )
            metrics["step"] = step
            append_jsonl(metrics_path, metrics)
            print(f"[E6][VAL] {json.dumps(metrics)}", flush=True)
            if metrics["val_noise_mse"] < best_val:
                best_val = metrics["val_noise_mse"]
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
                    process_mean,
                    process_std,
                    args,
                )
                print(f"[E6] saved best val={best_val:.6f}", flush=True)

        if step > 0 and step % args.sample_every == 0:
            generated, sample_labels = generate_process_grid(
                cae,
                ema_model,
                scheduler,
                latent_mean,
                latent_std,
                process_mean,
                process_std,
                device,
                args.samples_per_process,
                args.infer_steps,
                args.guidance_scale,
                args.sample_seed,
            )
            sample_path = os.path.join(
                args.out_root, f"process_samples_step{step:06d}.png"
            )
            save_process_grid(
                generated,
                sample_labels,
                sample_path,
                args.samples_per_process,
            )
            print(f"[E6] saved sample {sample_path}", flush=True)

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
                process_mean,
                process_std,
                args,
            )
            print(f"[E6] saved {last_path}", flush=True)
        step += 1

    generated, sample_labels = generate_process_grid(
        cae,
        ema_model,
        scheduler,
        latent_mean,
        latent_std,
        process_mean,
        process_std,
        device,
        args.samples_per_process,
        args.infer_steps,
        args.guidance_scale,
        args.sample_seed,
    )
    save_process_grid(
        generated,
        sample_labels,
        os.path.join(args.out_root, "process_samples_final.png"),
        args.samples_per_process,
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
        process_mean,
        process_std,
        args,
    )
    print(f"[E6] completed step={step} best_val={best_val:.6f}", flush=True)


def sample(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E6 sampling requires CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    model = ConditionedUNet(
        latent_channels,
        condition_dim=len(PROCESS_FEATURE_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    generated, labels = generate_process_grid(
        cae,
        model,
        build_noise_scheduler(),
        checkpoint["latent_mean"].float().to(device),
        checkpoint["latent_std"].float().to(device),
        checkpoint["process_mean"].float().to(device),
        checkpoint["process_std"].float().to(device),
        device,
        args.samples_per_process,
        args.infer_steps,
        args.guidance_scale,
        args.seed,
    )
    save_process_grid(
        generated,
        labels,
        args.out_png,
        args.samples_per_process,
    )
    print(f"[E6] saved sample {args.out_png}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_root", required=True)
    train_parser.add_argument("--cae_checkpoint", required=True)
    train_parser.add_argument("--e2_checkpoint", required=True)
    train_parser.add_argument("--out_root", required=True)
    train_parser.add_argument("--epoch_size", type=int, default=4000)
    train_parser.add_argument("--crop_scales", default="512,768,1024")
    train_parser.add_argument("--crop_weights", default="0.40,0.35,0.25")
    train_parser.add_argument("--train_max_x", type=float, default=0.60)
    train_parser.add_argument("--val_min_center_x", type=float, default=0.80)
    train_parser.add_argument("--val_crops_per_source", type=int, default=4)
    train_parser.add_argument("--horizontal_flip_prob", type=float, default=0.50)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--grad_accum", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--max_steps", type=int, default=20000)
    train_parser.add_argument("--condition_dropout", type=float, default=0.10)
    train_parser.add_argument("--ema_decay", type=float, default=0.999)
    train_parser.add_argument("--grad_clip", type=float, default=1.0)
    train_parser.add_argument("--val_every", type=int, default=500)
    train_parser.add_argument("--val_images", type=int, default=32)
    train_parser.add_argument("--sample_every", type=int, default=2000)
    train_parser.add_argument("--samples_per_process", type=int, default=4)
    train_parser.add_argument("--sample_seed", type=int, default=6060)
    train_parser.add_argument("--infer_steps", type=int, default=100)
    train_parser.add_argument("--guidance_scale", type=float, default=1.5)
    train_parser.add_argument("--save_every", type=int, default=1000)
    train_parser.add_argument("--log_every", type=int, default=25)
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--cae_checkpoint", required=True)
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--out_png", required=True)
    sample_parser.add_argument("--samples_per_process", type=int, default=4)
    sample_parser.add_argument("--infer_steps", type=int, default=150)
    sample_parser.add_argument("--guidance_scale", type=float, default=1.5)
    sample_parser.add_argument("--seed", type=int, default=6060)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "sample":
        sample(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
