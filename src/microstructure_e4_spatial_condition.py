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
from microstructure_e2_ldm import (
    build_noise_scheduler,
    denormalize_latent,
    load_cae,
    normalize_latent,
    seed_all,
    update_ema,
)
from microstructure_e3_conditional_ldm import PROCESS_PARAMETERS


STRUCTURE_NAMES = ("low_frequency", "dark_ridge", "arc_edge")
PROCESS_NAMES = ("laser_power", "scan_speed", "time", "linear_energy")


def process_id(path):
    value = Path(path).stem.split(".", 1)[0]
    if value not in PROCESS_PARAMETERS:
        raise ValueError(f"Unknown process for {path}")
    return value


def process_features(paths, device):
    values = []
    for path in paths:
        power, speed, duration = PROCESS_PARAMETERS[process_id(path)]
        values.append((power, speed, duration, power / speed))
    return torch.tensor(values, dtype=torch.float32, device=device)


class PathDataset(Dataset):
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


def gaussian_kernel(sigma, device, dtype):
    radius = max(1, int(math.ceil(3 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    return kernel / kernel.sum()


def gaussian_blur(images, sigma):
    kernel = gaussian_kernel(sigma, images.device, images.dtype)
    radius = kernel.numel() // 2
    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    images = F.conv2d(images, horizontal, padding=(0, radius), groups=channels)
    return F.conv2d(images, vertical, padding=(radius, 0), groups=channels)


def sobel_magnitude(gray):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def structure_map(images):
    gray = ((images + 1.0) * 0.5).mean(dim=1, keepdim=True)
    gray = F.interpolate(gray, size=(64, 64), mode="area")
    fine = gaussian_blur(gray, sigma=0.65)
    low = gaussian_blur(gray, sigma=2.0)
    dark_ridge = F.relu(low - fine)
    edge = sobel_magnitude(fine)
    return torch.cat((low, dark_ridge, edge), dim=1)


class SpatialConditionedUNet(nn.Module):
    def __init__(self, latent_channels, structure_channels, process_dim):
        super().__init__()
        self.latent_channels = latent_channels
        self.backbone = UNet2DModel(
            sample_size=64,
            in_channels=latent_channels + structure_channels,
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
        embedding_dim = 128 * 4
        self.process_projection = nn.Sequential(
            nn.Linear(process_dim + 1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.zeros_(self.process_projection[-1].weight)
        nn.init.zeros_(self.process_projection[-1].bias)

    def forward(self, noisy_latent, timestep, structure, process, present):
        model_input = torch.cat((noisy_latent, structure), dim=1)
        projection_input = torch.cat((process, present[:, None]), dim=1)
        embedding = self.process_projection(projection_input)
        return self.backbone(
            model_input,
            timestep,
            class_labels=embedding,
        ).sample


def initialize_from_e2(model, e2_state, latent_channels):
    state = model.backbone.state_dict()
    for key, value in e2_state.items():
        if key == "conv_in.weight":
            state[key].zero_()
            state[key][:, :latent_channels].copy_(value)
        else:
            state[key].copy_(value)
    model.backbone.load_state_dict(state, strict=True)


@torch.no_grad()
def compute_statistics(loader, device):
    structure_sum = torch.zeros(len(STRUCTURE_NAMES), dtype=torch.float64)
    structure_sum_sq = torch.zeros_like(structure_sum)
    structure_count = 0
    process_rows = []
    for images, paths in loader:
        images = images.to(device, non_blocking=True)
        maps = structure_map(images).double().permute(1, 0, 2, 3).reshape(
            len(STRUCTURE_NAMES), -1
        )
        structure_sum += maps.sum(dim=1).cpu()
        structure_sum_sq += maps.square().sum(dim=1).cpu()
        structure_count += maps.shape[1]
        process_rows.append(process_features(paths, device).cpu())
    structure_mean = structure_sum / structure_count
    structure_var = structure_sum_sq / structure_count - structure_mean.square()
    structure_std = torch.sqrt(structure_var.clamp_min(1e-12))
    process = torch.cat(process_rows)
    return (
        structure_mean.float(),
        structure_std.float(),
        process.mean(dim=0),
        process.std(dim=0).clamp_min(1e-6),
    )


def normalize_channels(values, mean, std):
    return (values - mean[None, :, None, None]) / std[None, :, None, None]


def normalize_rows(values, mean, std):
    return (values - mean[None]) / std[None]


@torch.no_grad()
def evaluate_noise(
    cae,
    model,
    loader,
    scheduler,
    latent_mean,
    latent_std,
    structure_mean,
    structure_std,
    process_mean,
    process_std,
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
        structure = normalize_channels(
            structure_map(images),
            structure_mean,
            structure_std,
        )
        process = normalize_rows(
            process_features(paths, device),
            process_mean,
            process_std,
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
            structure,
            process,
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
def generate(
    cae,
    model,
    noise_scheduler,
    latent_mean,
    latent_std,
    structure,
    process,
    device,
    infer_steps,
    guidance_scale,
    seed,
):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        (structure.shape[0], latent_mean.numel(), 64, 64),
        generator=generator,
        device=device,
    )
    sampler = DDIMScheduler.from_config(noise_scheduler.config)
    sampler.set_timesteps(infer_steps, device=device)
    present = torch.ones(structure.shape[0], device=device)
    absent = torch.zeros_like(present)
    null_structure = torch.zeros_like(structure)
    null_process = torch.zeros_like(process)
    for timestep in sampler.timesteps:
        unconditional = model(
            latent,
            timestep,
            null_structure,
            null_process,
            absent,
        )
        conditional = model(latent, timestep, structure, process, present)
        prediction = unconditional + guidance_scale * (conditional - unconditional)
        latent = sampler.step(
            prediction,
            timestep,
            latent,
            eta=0.0,
            generator=generator,
        ).prev_sample
    latent = denormalize_latent(latent, latent_mean, latent_std)
    images = []
    for start in range(0, latent.shape[0], 4):
        images.append(cae.decode(latent[start : start + 4]).cpu())
    model.train()
    return torch.cat(images)


def channel_correlation(first, second):
    first = first.flatten(2)
    second = second.flatten(2)
    first = first - first.mean(dim=2, keepdim=True)
    second = second - second.mean(dim=2, keepdim=True)
    numerator = (first * second).sum(dim=2)
    denominator = torch.sqrt(
        first.square().sum(dim=2) * second.square().sum(dim=2)
    ).clamp_min(1e-8)
    return numerator / denominator


def adherence_metrics(target_structure, generated_images):
    generated_structure = structure_map(generated_images.cuda()).cpu()
    correlation = channel_correlation(target_structure, generated_structure)
    l1 = (target_structure - generated_structure).abs().mean(dim=(0, 2, 3))
    return {
        "correlation": {
            name: float(correlation[:, index].mean().item())
            for index, name in enumerate(STRUCTURE_NAMES)
        },
        "l1": {
            name: float(l1[index].item())
            for index, name in enumerate(STRUCTURE_NAMES)
        },
    }


def tensor_to_pil(image):
    array = ((image.detach().clamp(-1, 1) + 1.0) * 127.5)
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def map_to_pil(channel):
    channel = channel.detach().float().cpu()
    minimum = torch.quantile(channel, 0.02)
    maximum = torch.quantile(channel, 0.98)
    channel = ((channel - minimum) / (maximum - minimum).clamp_min(1e-8)).clamp(0, 1)
    array = (channel * 255).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L").convert("RGB").resize((512, 512))


def save_oracle_grid(originals, raw_structure, generated, paths, output):
    count = originals.shape[0]
    header = 24
    canvas = Image.new("RGB", (count * 512, 3 * 512 + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        label = f"P{process_id(paths[index])}: original / ridge condition / generated"
        draw.text((index * 512 + 4, 5), label, fill="black")
        canvas.paste(tensor_to_pil(originals[index]), (index * 512, header))
        canvas.paste(
            map_to_pil(raw_structure[index, 1]),
            (index * 512, header + 512),
        )
        canvas.paste(
            tensor_to_pil(generated[index]),
            (index * 512, header + 1024),
        )
    canvas.save(output)


def select_oracle_paths(validation_paths):
    selected = {}
    for path in validation_paths:
        selected.setdefault(process_id(path), path)
    missing = set(PROCESS_PARAMETERS) - set(selected)
    if missing:
        raise RuntimeError(f"No validation patch for processes: {sorted(missing)}")
    return [selected[key] for key in sorted(selected, key=int)]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def save_checkpoint(
    path,
    model,
    ema,
    optimizer,
    scaler,
    step,
    best_val,
    statistics,
    latent_mean,
    latent_std,
    args,
):
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val": best_val,
            "statistics": {key: value.cpu() for key, value in statistics.items()},
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
            "structure_names": STRUCTURE_NAMES,
            "process_names": PROCESS_NAMES,
            "args": vars(args),
        },
        path,
    )


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E4 training requires CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)

    paths = list_images(args.data_root)
    train_paths, validation_paths, ignored_paths, source_counts = build_spatial_split(
        paths,
        train_max=args.train_max_x,
        val_min=args.val_min_x,
    )
    process_counts = Counter(process_id(path) for path in train_paths)
    weights = [1.0 / process_counts[process_id(path)] for path in train_paths]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_paths),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        PathDataset(train_paths),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    stats_loader = DataLoader(
        PathDataset(train_paths),
        batch_size=args.stats_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        PathDataset(validation_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    oracle_paths = select_oracle_paths(validation_paths)
    oracle_loader = DataLoader(
        PathDataset(oracle_paths),
        batch_size=len(oracle_paths),
        shuffle=False,
        num_workers=0,
    )
    oracle_images_cpu, oracle_paths_batch = next(iter(oracle_loader))
    oracle_images = oracle_images_cpu.to(device)
    oracle_paths_batch = list(oracle_paths_batch)

    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    e2 = torch.load(args.e2_checkpoint, map_location="cpu")
    latent_mean = e2["latent_mean"].float().to(device)
    latent_std = e2["latent_std"].float().to(device)
    (
        structure_mean_cpu,
        structure_std_cpu,
        process_mean_cpu,
        process_std_cpu,
    ) = compute_statistics(stats_loader, device)
    statistics = {
        "structure_mean": structure_mean_cpu,
        "structure_std": structure_std_cpu,
        "process_mean": process_mean_cpu,
        "process_std": process_std_cpu,
    }
    structure_mean = structure_mean_cpu.to(device)
    structure_std = structure_std_cpu.to(device)
    process_mean = process_mean_cpu.to(device)
    process_std = process_std_cpu.to(device)

    model = SpatialConditionedUNet(
        latent_channels,
        len(STRUCTURE_NAMES),
        len(PROCESS_NAMES),
    ).to(device)
    initialize_from_e2(model, e2["ema"], latent_channels)
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    scheduler = build_noise_scheduler()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    last_path = os.path.join(args.out_root, "spatial_last.pt")
    best_path = os.path.join(args.out_root, "spatial_best.pt")
    metrics_path = os.path.join(args.out_root, "metrics.jsonl")
    step = 0
    best_val = float("inf")

    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        best_val = float(checkpoint["best_val"])
        print(f"[E4] resumed step={step}", flush=True)

    with open(os.path.join(args.out_root, "experiment.json"), "w") as handle:
        json.dump(
            {
                "train": len(train_paths),
                "validation": len(validation_paths),
                "ignored": len(ignored_paths),
                "process_counts": process_counts,
                "source_counts": source_counts,
                "oracle_paths": oracle_paths,
                "structure_names": STRUCTURE_NAMES,
                "process_names": PROCESS_NAMES,
                "statistics": {
                    key: value.tolist() for key, value in statistics.items()
                },
            },
            handle,
            indent=2,
        )
    print(
        f"[E4] train={len(train_paths)} val={len(validation_paths)} "
        f"process_counts={dict(process_counts)} structure={STRUCTURE_NAMES}",
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
            latent = normalize_latent(cae.encode(images).float(), latent_mean, latent_std)
            structure = normalize_channels(
                structure_map(images),
                structure_mean,
                structure_std,
            )
            process = normalize_rows(
                process_features(batch_paths, device),
                process_mean,
                process_std,
            )
            present = (
                torch.rand(images.shape[0], device=device) >= args.condition_dropout
            ).float()
            structure = structure * present[:, None, None, None]
            process = process * present[:, None]

        noise = torch.randn_like(latent)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (latent.shape[0],),
            device=device,
        ).long()
        noisy = scheduler.add_noise(latent, noise, timesteps)
        with torch.amp.autocast("cuda", enabled=args.amp):
            prediction = model(noisy, timesteps, structure, process, present)
            loss = F.mse_loss(prediction.float(), noise.float()) / args.grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model, args.ema_decay)

        if step % args.log_every == 0:
            print(
                f"[E4] step={step}/{args.max_steps} "
                f"noise_mse={loss.item() * args.grad_accum:.6f} "
                f"present={present.mean().item():.3f} "
                f"time={(time.time() - started) / 60:.1f}m",
                flush=True,
            )

        if step > 0 and step % args.val_every == 0:
            val_loss = evaluate_noise(
                cae,
                ema,
                validation_loader,
                scheduler,
                latent_mean,
                latent_std,
                structure_mean,
                structure_std,
                process_mean,
                process_std,
                device,
                args.val_images,
                args.seed + 919,
            )
            record = {"step": step, "val_noise_mse": val_loss}
            append_jsonl(metrics_path, record)
            print(f"[E4][VAL] {json.dumps(record)}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_path,
                    model,
                    ema,
                    optimizer,
                    scaler,
                    step,
                    best_val,
                    statistics,
                    latent_mean,
                    latent_std,
                    args,
                )

        if step > 0 and step % args.sample_every == 0:
            raw_oracle_structure = structure_map(oracle_images)
            oracle_structure = normalize_channels(
                raw_oracle_structure,
                structure_mean,
                structure_std,
            )
            oracle_process = normalize_rows(
                process_features(oracle_paths_batch, device),
                process_mean,
                process_std,
            )
            generated = generate(
                cae,
                ema,
                scheduler,
                latent_mean,
                latent_std,
                oracle_structure,
                oracle_process,
                device,
                args.infer_steps,
                args.guidance_scale,
                args.sample_seed,
            )
            adherence = adherence_metrics(raw_oracle_structure.cpu(), generated)
            adherence["step"] = step
            append_jsonl(
                os.path.join(args.out_root, "adherence.jsonl"),
                adherence,
            )
            save_oracle_grid(
                oracle_images_cpu,
                raw_oracle_structure.cpu(),
                generated,
                oracle_paths_batch,
                os.path.join(args.out_root, f"oracle_step{step:06d}.png"),
            )
            print(f"[E4][ADHERENCE] {json.dumps(adherence)}", flush=True)

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(
                last_path,
                model,
                ema,
                optimizer,
                scaler,
                step,
                best_val,
                statistics,
                latent_mean,
                latent_std,
                args,
            )
        step += 1

    raw_oracle_structure = structure_map(oracle_images)
    generated = generate(
        cae,
        ema,
        scheduler,
        latent_mean,
        latent_std,
        normalize_channels(raw_oracle_structure, structure_mean, structure_std),
        normalize_rows(
            process_features(oracle_paths_batch, device),
            process_mean,
            process_std,
        ),
        device,
        args.infer_steps,
        args.guidance_scale,
        args.sample_seed,
    )
    adherence = adherence_metrics(raw_oracle_structure.cpu(), generated)
    adherence.update({"step": step, "final": True})
    append_jsonl(os.path.join(args.out_root, "adherence.jsonl"), adherence)
    save_oracle_grid(
        oracle_images_cpu,
        raw_oracle_structure.cpu(),
        generated,
        oracle_paths_batch,
        os.path.join(args.out_root, "oracle_final.png"),
    )
    save_checkpoint(
        last_path,
        model,
        ema,
        optimizer,
        scaler,
        step,
        best_val,
        statistics,
        latent_mean,
        latent_std,
        args,
    )
    print(f"[E4] completed step={step} adherence={json.dumps(adherence)}", flush=True)


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
    parser.add_argument("--max_steps", type=int, default=8000)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_images", type=int, default=128)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=4040)
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
