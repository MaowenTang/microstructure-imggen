#!/usr/bin/env python3

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
COORD_RE = re.compile(r"^(?P<source>.+)_p\d+_x(?P<x>\d+)_y(?P<y>\d+)$")


def list_images(root):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, f"*{ext}")))
        files.extend(glob.glob(os.path.join(root, f"*{ext.upper()}")))
    return sorted(set(files))


def parse_patch(path):
    match = COORD_RE.match(Path(path).stem)
    if not match:
        return None
    source = match.group("source")
    return source, int(match.group("x")), int(match.group("y"))


def stable_fraction(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_spatial_split(paths, train_max=0.60, val_min=0.80):
    grouped = defaultdict(list)
    fallback = []
    for path in paths:
        parsed = parse_patch(path)
        if parsed is None:
            fallback.append(path)
        else:
            grouped[parsed[0]].append((path, parsed[1], parsed[2]))

    train_paths = []
    val_paths = []
    ignored_paths = []
    source_counts = {}

    for source, rows in sorted(grouped.items()):
        max_x = max(row[1] for row in rows)
        inferred_width = max_x + 512
        counts = {"train": 0, "val": 0, "ignored_buffer": 0}

        for path, x, _ in rows:
            normalized_center_x = (x + 256) / inferred_width
            if normalized_center_x <= train_max:
                train_paths.append(path)
                counts["train"] += 1
            elif normalized_center_x >= val_min:
                val_paths.append(path)
                counts["val"] += 1
            else:
                ignored_paths.append(path)
                counts["ignored_buffer"] += 1
        source_counts[source] = counts

    for path in fallback:
        if stable_fraction(path) < 0.8:
            train_paths.append(path)
        else:
            val_paths.append(path)

    if not train_paths or not val_paths:
        raise RuntimeError("Spatial split produced an empty train or validation set.")

    return train_paths, val_paths, ignored_paths, source_counts


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def rank():
    return int(os.environ.get("RANK", "0"))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main():
    return rank() == 0


def setup_ddp():
    if not is_ddp():
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank())
    return True


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def log(message):
    if is_main():
        print(message, flush=True)


class PatchDataset(Dataset):
    def __init__(self, paths, flip=False, grayscale_rgb=False):
        self.paths = list(paths)
        self.flip = flip
        self.grayscale_rgb = grayscale_rgb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.BICUBIC)
        if self.grayscale_rgb:
            image = image.convert("L").convert("RGB")
        if self.flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor / 127.5 - 1.0


def group_norm(channels):
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        return self.res2(self.res1(self.down(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.res1 = ResidualBlock(out_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.res2(self.res1(self.conv(x)))


class MorphologyCAE(nn.Module):
    def __init__(self, latent_channels=8, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            ResidualBlock(base_channels, base_channels),
            DownBlock(base_channels, base_channels),
            DownBlock(base_channels, base_channels * 2),
            DownBlock(base_channels * 2, base_channels * 4),
            ResidualBlock(base_channels * 4, base_channels * 4),
            group_norm(base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1),
            ResidualBlock(base_channels * 4, base_channels * 4),
            UpBlock(base_channels * 4, base_channels * 2),
            UpBlock(base_channels * 2, base_channels),
            UpBlock(base_channels, base_channels),
            group_norm(base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def charbonnier_loss(prediction, target, epsilon=1e-3):
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def ssim_loss(x, y, window=11):
    x = (x + 1.0) * 0.5
    y = (y + 1.0) * 0.5
    c1 = 0.01**2
    c2 = 0.03**2
    padding = window // 2

    mu_x = F.avg_pool2d(x, window, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=padding)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.avg_pool2d(x.square(), window, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y.square(), window, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=padding) - mu_xy

    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-12
    )
    return 1.0 - value.mean()


def sobel_magnitude(x):
    gray = x.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def edge_loss(prediction, target):
    return F.l1_loss(sobel_magnitude(prediction), sobel_magnitude(target))


def fft_loss(prediction, target):
    pred_fft = torch.log1p(torch.abs(torch.fft.rfft2(prediction.float(), norm="ortho")))
    target_fft = torch.log1p(torch.abs(torch.fft.rfft2(target.float(), norm="ortho")))
    return F.l1_loss(pred_fft, target_fft)


def auxiliary_scale(step, warmup_steps, ramp_steps):
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, (step - warmup_steps + 1) / ramp_steps)


def tensor_to_pil(x):
    array = ((x.detach().clamp(-1, 1) + 1.0) * 127.5)
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def save_grid(original, reconstruction, path, max_samples=6):
    count = min(max_samples, original.shape[0])
    canvas = Image.new("RGB", (count * 512, 2 * 512))
    for index in range(count):
        canvas.paste(tensor_to_pil(original[index]), (index * 512, 0))
        canvas.paste(tensor_to_pil(reconstruction[index]), (index * 512, 512))
    canvas.save(path)


@torch.no_grad()
def evaluate(model, loader, device, max_images, grid_path=None):
    model.eval()
    totals = defaultdict(float)
    latent_sum = None
    latent_sum_sq = None
    latent_count = 0
    image_count = 0
    grid_saved = False

    for x in loader:
        x = x.to(device, non_blocking=True)
        if image_count + x.shape[0] > max_images:
            x = x[: max_images - image_count]
        reconstruction, latent = model(x)

        mse = F.mse_loss(reconstruction, x)
        l1 = F.l1_loss(reconstruction, x)
        psnr = 10.0 * torch.log10(4.0 / torch.clamp(mse, min=1e-12))
        ssim = 1.0 - ssim_loss(reconstruction, x)
        edge = edge_loss(reconstruction, x)
        spectrum = fft_loss(reconstruction, x)

        batch = x.shape[0]
        totals["mse"] += float(mse.item()) * batch
        totals["l1"] += float(l1.item()) * batch
        totals["psnr"] += float(psnr.item()) * batch
        totals["ssim"] += float(ssim.item()) * batch
        totals["edge_mae"] += float(edge.item()) * batch
        totals["log_fft_mae"] += float(spectrum.item()) * batch

        values = latent.detach().double().permute(1, 0, 2, 3).reshape(latent.shape[1], -1).cpu()
        if latent_sum is None:
            latent_sum = torch.zeros(latent.shape[1], dtype=torch.float64)
            latent_sum_sq = torch.zeros(latent.shape[1], dtype=torch.float64)
        latent_sum += values.sum(dim=1)
        latent_sum_sq += values.square().sum(dim=1)
        latent_count += values.shape[1]

        if grid_path and not grid_saved:
            save_grid(x, reconstruction, grid_path)
            grid_saved = True

        image_count += batch
        if image_count >= max_images:
            break

    mean = latent_sum / latent_count
    variance = torch.clamp(latent_sum_sq / latent_count - mean.square(), min=0.0)
    metrics = {key: value / image_count for key, value in totals.items()}
    metrics["images"] = image_count
    metrics["latent_mean"] = mean.tolist()
    metrics["latent_std"] = torch.sqrt(variance).tolist()
    metrics["latent_std_mean"] = float(torch.sqrt(variance).mean().item())
    model.train()
    return metrics


def save_checkpoint(model, optimizer, scaler, step, args, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "args": vars(args),
        },
        path,
    )


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def train(args):
    ddp = setup_ddp()
    seed_all(args.seed + rank())

    device = torch.device(f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CAE training requires CUDA.")

    all_paths = list_images(os.path.expanduser(args.data_root))
    train_paths, val_paths, ignored_paths, source_counts = build_spatial_split(
        all_paths,
        train_max=args.train_max_x,
        val_min=args.val_min_x,
    )

    os.makedirs(args.out_root, exist_ok=True)
    if is_main():
        with open(os.path.join(args.out_root, "split.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "train": len(train_paths),
                    "validation": len(val_paths),
                    "ignored_buffer": len(ignored_paths),
                    "source_counts": source_counts,
                    "train_max_x": args.train_max_x,
                    "val_min_x": args.val_min_x,
                },
                handle,
                indent=2,
            )

    train_dataset = PatchDataset(train_paths, flip=True, grayscale_rgb=args.grayscale_rgb)
    val_dataset = PatchDataset(val_paths, flip=False, grayscale_rgb=args.grayscale_rgb)
    sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=args.seed, drop_last=True)
        if ddp
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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

    model = MorphologyCAE(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
    ).to(device)
    if ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank()],
            output_device=local_rank(),
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    checkpoint_path = os.path.join(args.out_root, "cae_last.pt")
    metrics_path = os.path.join(args.out_root, "metrics.jsonl")
    start_step = 0
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        (model.module if ddp else model).load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", 0))
        log(f"[E1] resumed from step={start_step}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log(
        f"[E1] train={len(train_dataset)} val={len(val_dataset)} ignored={len(ignored_paths)} "
        f"batch={args.batch_size} world={world_size()} parameters={parameter_count:,}"
    )
    log(
        f"[E1] latent={args.latent_channels}x64x64 lr={args.lr} "
        f"weights: charb=1 ssim={args.w_ssim} edge={args.w_edge} fft={args.w_fft}"
    )
    log(f"[E1] grayscale_rgb={args.grayscale_rgb}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    epoch = 0
    started = time.time()

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for x in train_loader:
            x = x.to(device, non_blocking=True)
            scale = auxiliary_scale(step, args.aux_warmup_steps, args.aux_ramp_steps)

            with torch.cuda.amp.autocast(enabled=args.amp):
                reconstruction, latent = model(x)
                loss_charb = charbonnier_loss(reconstruction, x)
                loss_ssim = ssim_loss(reconstruction, x)
                loss_edge = edge_loss(reconstruction, x)
                loss_fft = fft_loss(reconstruction, x)
                loss = (
                    loss_charb
                    + scale * args.w_ssim * loss_ssim
                    + scale * args.w_edge * loss_edge
                    + scale * args.w_fft * loss_fft
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if is_main() and step % args.log_every == 0:
                minutes = (time.time() - started) / 60
                log(
                    f"[E1] step={step}/{args.max_steps} loss={loss.item():.5f} "
                    f"charb={loss_charb.item():.5f} ssim={loss_ssim.item():.5f} "
                    f"edge={loss_edge.item():.5f} fft={loss_fft.item():.5f} "
                    f"aux_scale={scale:.3f} latent_std={latent.std().item():.4f} "
                    f"time={minutes:.1f}m"
                )

            if is_main() and step > 0 and step % args.val_every == 0:
                base_model = model.module if ddp else model
                metrics = evaluate(
                    base_model,
                    val_loader,
                    device,
                    max_images=args.val_images,
                    grid_path=os.path.join(args.out_root, f"recon_step{step:06d}.png"),
                )
                record = {"step": step, "aux_scale": scale, **metrics}
                append_jsonl(metrics_path, record)
                log(f"[E1][VAL] {json.dumps(record)}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base_model = model.module if ddp else model
                save_checkpoint(base_model, optimizer, scaler, step, args, checkpoint_path)
                log(f"[E1] saved {checkpoint_path}")

            step += 1
            if step >= args.max_steps:
                break
        epoch += 1

    if is_main():
        base_model = model.module if ddp else model
        final_metrics = evaluate(
            base_model,
            val_loader,
            device,
            max_images=args.val_images,
            grid_path=os.path.join(args.out_root, "recon_final.png"),
        )
        append_jsonl(metrics_path, {"step": step, "final": True, **final_metrics})
        save_checkpoint(base_model, optimizer, scaler, step, args, checkpoint_path)
        with open(os.path.join(args.out_root, "latent_stats.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mean": final_metrics["latent_mean"],
                    "std": final_metrics["latent_std"],
                    "images": final_metrics["images"],
                },
                handle,
                indent=2,
            )
        log(f"[E1] completed step={step} metrics={json.dumps(final_metrics)}")

    if ddp:
        dist.barrier()
    cleanup_ddp()


def evaluate_checkpoint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    latent_channels = int(checkpoint_args.get("latent_channels", args.latent_channels))
    base_channels = int(checkpoint_args.get("base_channels", args.base_channels))

    model = MorphologyCAE(latent_channels=latent_channels, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    paths = list_images(os.path.expanduser(args.data_root))
    _, val_paths, _, _ = build_spatial_split(
        paths,
        train_max=args.train_max_x,
        val_min=args.val_min_x,
    )
    loader = DataLoader(
        PatchDataset(val_paths, flip=False, grayscale_rgb=args.grayscale_rgb),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    os.makedirs(args.out_root, exist_ok=True)
    metrics = evaluate(
        model,
        loader,
        device,
        max_images=args.val_images,
        grid_path=os.path.join(args.out_root, "recon_eval.png"),
    )
    with open(os.path.join(args.out_root, "eval_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_root", required=True)
    common.add_argument("--out_root", required=True)
    common.add_argument("--latent_channels", type=int, default=8)
    common.add_argument("--base_channels", type=int, default=64)
    common.add_argument("--batch_size", type=int, default=4)
    common.add_argument("--num_workers", type=int, default=4)
    common.add_argument("--train_max_x", type=float, default=0.60)
    common.add_argument("--val_min_x", type=float, default=0.80)
    common.add_argument("--val_images", type=int, default=128)
    common.add_argument("--grayscale_rgb", action="store_true")

    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--max_steps", type=int, default=5000)
    train_parser.add_argument("--w_ssim", type=float, default=0.10)
    train_parser.add_argument("--w_edge", type=float, default=0.05)
    train_parser.add_argument("--w_fft", type=float, default=0.05)
    train_parser.add_argument("--aux_warmup_steps", type=int, default=1000)
    train_parser.add_argument("--aux_ramp_steps", type=int, default=2000)
    train_parser.add_argument("--grad_clip", type=float, default=1.0)
    train_parser.add_argument("--log_every", type=int, default=25)
    train_parser.add_argument("--val_every", type=int, default=500)
    train_parser.add_argument("--save_every", type=int, default=1000)
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--seed", type=int, default=0)

    eval_parser = subparsers.add_parser("eval", parents=[common])
    eval_parser.add_argument("--checkpoint", required=True)

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate_checkpoint(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
