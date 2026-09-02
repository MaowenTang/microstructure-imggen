#!/usr/bin/env python3

import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from microstructure_e1_cae import (
    MorphologyCAE,
    PatchDataset,
    build_spatial_split,
    list_images,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_pil(x):
    array = ((x.detach().clamp(-1, 1) + 1.0) * 127.5)
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_grid(images, path):
    count = images.shape[0]
    columns = max(1, int(math.ceil(math.sqrt(count))))
    rows = int(math.ceil(count / columns))
    canvas = Image.new("RGB", (columns * 512, rows * 512))
    for index in range(count):
        row, column = divmod(index, columns)
        canvas.paste(tensor_to_pil(images[index]), (column * 512, row * 512))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_cae(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    latent_channels = int(checkpoint_args.get("latent_channels", 8))
    base_channels = int(checkpoint_args.get("base_channels", 64))
    model = MorphologyCAE(
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, latent_channels


def build_unet(latent_channels):
    return UNet2DModel(
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
    )


def build_noise_scheduler():
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )


@torch.no_grad()
def compute_latent_stats(cae, loader, device, latent_channels):
    sums = torch.zeros(latent_channels, dtype=torch.float64)
    sums_squared = torch.zeros(latent_channels, dtype=torch.float64)
    count = 0

    for images in loader:
        images = images.to(device, non_blocking=True)
        latent = cae.encode(images).double()
        values = latent.permute(1, 0, 2, 3).reshape(latent_channels, -1).cpu()
        sums += values.sum(dim=1)
        sums_squared += values.square().sum(dim=1)
        count += values.shape[1]

    mean = sums / count
    variance = torch.clamp(sums_squared / count - mean.square(), min=1e-12)
    return mean.float(), torch.sqrt(variance).float(), count


def normalize_latent(latent, mean, std):
    return (latent - mean[None, :, None, None]) / std[None, :, None, None]


def denormalize_latent(latent, mean, std):
    return latent * std[None, :, None, None] + mean[None, :, None, None]


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def evaluate_denoising(
    cae,
    unet,
    loader,
    scheduler,
    latent_mean,
    latent_std,
    device,
    max_images,
    seed,
):
    unet.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total_loss = 0.0
    image_count = 0

    for images in loader:
        images = images.to(device, non_blocking=True)
        if image_count + images.shape[0] > max_images:
            images = images[: max_images - image_count]

        latent = normalize_latent(cae.encode(images).float(), latent_mean, latent_std)
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
        noisy_latent = scheduler.add_noise(latent, noise, timesteps)
        prediction = unet(noisy_latent, timesteps).sample
        loss = F.mse_loss(prediction.float(), noise.float())
        total_loss += float(loss.item()) * latent.shape[0]
        image_count += latent.shape[0]
        if image_count >= max_images:
            break

    unet.train()
    return total_loss / image_count


@torch.no_grad()
def generate(
    cae,
    unet,
    noise_scheduler,
    latent_mean,
    latent_std,
    device,
    count,
    infer_steps,
    seed,
    decode_batch=4,
):
    unet.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        (count, latent_mean.numel(), 64, 64),
        generator=generator,
        device=device,
    )
    scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    scheduler.set_timesteps(infer_steps, device=device)

    for timestep in scheduler.timesteps:
        prediction = unet(latent, timestep).sample
        latent = scheduler.step(
            prediction,
            timestep,
            latent,
            eta=0.0,
            generator=generator,
        ).prev_sample

    latent = denormalize_latent(latent, latent_mean, latent_std)
    decoded = []
    for start in range(0, count, decode_batch):
        decoded.append(cae.decode(latent[start : start + decode_batch]).cpu())
    unet.train()
    return torch.cat(decoded, dim=0)


def save_checkpoint(
    path,
    unet,
    ema_model,
    optimizer,
    scaler,
    step,
    best_val,
    latent_mean,
    latent_std,
    args,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "unet": unet.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best_val": best_val,
            "latent_mean": latent_mean.detach().cpu(),
            "latent_std": latent_std.detach().cpu(),
            "args": vars(args),
        },
        path,
    )


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E2 latent diffusion training requires CUDA.")

    seed_all(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.out_root, exist_ok=True)

    paths = list_images(os.path.expanduser(args.data_root))
    train_paths, val_paths, ignored_paths, source_counts = build_spatial_split(
        paths,
        train_max=args.train_max_x,
        val_min=args.val_min_x,
    )
    train_loader = DataLoader(
        PatchDataset(train_paths, flip=True, grayscale_rgb=args.grayscale_rgb),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    stats_loader = DataLoader(
        PatchDataset(train_paths, flip=False, grayscale_rgb=args.grayscale_rgb),
        batch_size=args.stats_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        PatchDataset(val_paths, flip=False, grayscale_rgb=args.grayscale_rgb),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    latent_mean_cpu, latent_std_cpu, stats_count = compute_latent_stats(
        cae,
        stats_loader,
        device,
        latent_channels,
    )
    latent_mean = latent_mean_cpu.to(device)
    latent_std = latent_std_cpu.to(device)
    with open(os.path.join(args.out_root, "latent_stats_train.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "mean": latent_mean_cpu.tolist(),
                "std": latent_std_cpu.tolist(),
                "values_per_channel": stats_count,
                "patches": len(train_paths),
                "split": "training spatial region only",
            },
            handle,
            indent=2,
        )
    with open(os.path.join(args.out_root, "split.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train": len(train_paths),
                "validation": len(val_paths),
                "ignored_buffer": len(ignored_paths),
                "source_counts": source_counts,
            },
            handle,
            indent=2,
        )

    unet = build_unet(latent_channels).to(device)
    ema_model = copy.deepcopy(unet).eval()
    ema_model.requires_grad_(False)
    scheduler = build_noise_scheduler()
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    last_path = os.path.join(args.out_root, "ldm_last.pt")
    best_path = os.path.join(args.out_root, "ldm_best.pt")
    metrics_path = os.path.join(args.out_root, "metrics.jsonl")
    step = 0
    best_val = float("inf")
    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location="cpu")
        unet.load_state_dict(checkpoint["unet"], strict=True)
        ema_model.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_val", best_val))
        saved_mean = checkpoint["latent_mean"].float()
        saved_std = checkpoint["latent_std"].float()
        if not torch.allclose(saved_mean, latent_mean_cpu, rtol=1e-4, atol=1e-5):
            raise RuntimeError("Training latent mean changed since the checkpoint was created.")
        if not torch.allclose(saved_std, latent_std_cpu, rtol=1e-4, atol=1e-5):
            raise RuntimeError("Training latent std changed since the checkpoint was created.")
        print(f"[E2] resumed from step={step}", flush=True)

    parameters = sum(parameter.numel() for parameter in unet.parameters())
    print(
        f"[E2] train={len(train_paths)} val={len(val_paths)} ignored={len(ignored_paths)} "
        f"batch={args.batch_size} grad_accum={args.grad_accum} parameters={parameters:,}",
        flush=True,
    )
    print(
        f"[E2] latent_mean={latent_mean_cpu.tolist()} "
        f"latent_std={latent_std_cpu.tolist()}",
        flush=True,
    )
    print(
        "[E2] scheduler=squaredcos_cap_v2 prediction=epsilon clip_sample=False",
        flush=True,
    )
    print(f"[E2] grayscale_rgb={args.grayscale_rgb}", flush=True)

    unet.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    started = time.time()

    while step < args.max_steps:
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images = next(iterator)
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            latent = cae.encode(images).float()
            latent = normalize_latent(latent, latent_mean, latent_std)
        noise = torch.randn_like(latent)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (latent.shape[0],),
            device=device,
        ).long()
        noisy_latent = scheduler.add_noise(latent, noise, timesteps)

        with torch.amp.autocast("cuda", enabled=args.amp):
            prediction = unet(noisy_latent, timesteps).sample
            loss = F.mse_loss(prediction.float(), noise.float()) / args.grad_accum

        scaler.scale(loss).backward()
        optimizer_step = (step + 1) % args.grad_accum == 0
        if optimizer_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema_model, unet, args.ema_decay)

        if step % args.log_every == 0:
            minutes = (time.time() - started) / 60.0
            print(
                f"[E2] step={step}/{args.max_steps} "
                f"noise_mse={loss.item() * args.grad_accum:.6f} "
                f"z_mean={latent.mean().item():.4f} z_std={latent.std().item():.4f} "
                f"time={minutes:.1f}m",
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
                device,
                args.val_images,
                args.seed + 701,
            )
            record = {"step": step, "val_noise_mse": val_loss}
            append_jsonl(metrics_path, record)
            print(f"[E2][VAL] {json.dumps(record)}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_path,
                    unet,
                    ema_model,
                    optimizer,
                    scaler,
                    step,
                    best_val,
                    latent_mean,
                    latent_std,
                    args,
                )
                print(f"[E2] saved best checkpoint val={best_val:.6f}", flush=True)

        if step > 0 and step % args.sample_every == 0:
            images = generate(
                cae,
                ema_model,
                scheduler,
                latent_mean,
                latent_std,
                device,
                args.sample_count,
                args.infer_steps,
                args.sample_seed,
            )
            sample_path = os.path.join(args.out_root, f"samples_step{step:06d}.png")
            save_grid(images, sample_path)
            print(f"[E2] saved sample {sample_path}", flush=True)

        if step > 0 and step % args.save_every == 0:
            save_checkpoint(
                last_path,
                unet,
                ema_model,
                optimizer,
                scaler,
                step,
                best_val,
                latent_mean,
                latent_std,
                args,
            )
            print(f"[E2] saved {last_path}", flush=True)

        step += 1

    final_images = generate(
        cae,
        ema_model,
        scheduler,
        latent_mean,
        latent_std,
        device,
        args.sample_count,
        args.infer_steps,
        args.sample_seed,
    )
    save_grid(final_images, os.path.join(args.out_root, "samples_final.png"))
    save_checkpoint(
        last_path,
        unet,
        ema_model,
        optimizer,
        scaler,
        step,
        best_val,
        latent_mean,
        latent_std,
        args,
    )
    print(f"[E2] completed step={step} best_val={best_val:.6f}", flush=True)


def sample(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E2 sampling requires CUDA.")

    seed_all(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(args.ldm_checkpoint, map_location="cpu")
    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    unet = build_unet(latent_channels).to(device)
    unet.load_state_dict(checkpoint["ema"], strict=True)
    unet.eval()
    latent_mean = checkpoint["latent_mean"].float().to(device)
    latent_std = checkpoint["latent_std"].float().to(device)
    images = generate(
        cae,
        unet,
        build_noise_scheduler(),
        latent_mean,
        latent_std,
        device,
        args.count,
        args.infer_steps,
        args.seed,
    )
    save_grid(images, args.out_png)
    print(f"[E2] saved sample {args.out_png}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_root", required=True)
    train_parser.add_argument("--cae_checkpoint", required=True)
    train_parser.add_argument("--out_root", required=True)
    train_parser.add_argument("--train_max_x", type=float, default=0.60)
    train_parser.add_argument("--val_min_x", type=float, default=0.80)
    train_parser.add_argument("--grayscale_rgb", action="store_true")
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--stats_batch_size", type=int, default=8)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--grad_accum", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--max_steps", type=int, default=10000)
    train_parser.add_argument("--ema_decay", type=float, default=0.999)
    train_parser.add_argument("--grad_clip", type=float, default=1.0)
    train_parser.add_argument("--val_every", type=int, default=500)
    train_parser.add_argument("--val_images", type=int, default=128)
    train_parser.add_argument("--sample_every", type=int, default=1000)
    train_parser.add_argument("--sample_count", type=int, default=9)
    train_parser.add_argument("--sample_seed", type=int, default=2026)
    train_parser.add_argument("--infer_steps", type=int, default=100)
    train_parser.add_argument("--save_every", type=int, default=1000)
    train_parser.add_argument("--log_every", type=int, default=25)
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--cae_checkpoint", required=True)
    sample_parser.add_argument("--ldm_checkpoint", required=True)
    sample_parser.add_argument("--out_png", required=True)
    sample_parser.add_argument("--count", type=int, default=9)
    sample_parser.add_argument("--infer_steps", type=int, default=150)
    sample_parser.add_argument("--seed", type=int, default=0)

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
