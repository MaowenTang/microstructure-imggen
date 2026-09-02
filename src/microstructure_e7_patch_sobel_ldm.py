#!/usr/bin/env python3

import argparse
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from microstructure_e1_cae import list_images
from microstructure_e2_ldm import (
    build_noise_scheduler,
    load_cae,
    normalize_latent,
    seed_all,
    update_ema,
)
from microstructure_e3_conditional_ldm import (
    ConditionedUNet,
    PROCESS_FEATURE_NAMES,
    PROCESS_PARAMETERS,
    save_process_grid,
)
from microstructure_e6_multiscale_ldm import (
    append_jsonl,
    evaluate_denoising,
    normalized_process_features,
    process_statistics,
    save_checkpoint,
)


PROCESS_IDS = tuple(sorted(PROCESS_PARAMETERS, key=int))
PATCH_RE = re.compile(r"^(?P<source>(?P<process>[0-9])\.[0-9]+)_p[0-9]+_x(?P<x>[0-9]+)_y(?P<y>[0-9]+)")


def parse_processes(value):
    if value.lower() == "all":
        return PROCESS_IDS
    processes = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}, key=int))
    unknown = [process for process in processes if process not in PROCESS_IDS]
    if unknown:
        raise ValueError(f"Unknown process ids: {unknown}")
    if not processes:
        raise ValueError("At least one process id is required.")
    return processes


def patch_metadata(path):
    match = PATCH_RE.match(Path(path).stem)
    if not match:
        raise ValueError(f"Cannot parse patch metadata from {path}")
    return {
        "source": match.group("source"),
        "process": match.group("process"),
        "x": int(match.group("x")),
        "y": int(match.group("y")),
    }


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous() / 127.5 - 1.0


def load_patch(path, grayscale_rgb=False):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (512, 512):
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((512, 512), Image.Resampling.BICUBIC)
        if grayscale_rgb:
            image = image.convert("L").convert("RGB")
        return image.copy()


def sobel_energy_score(image):
    gray = image.convert("L")
    x = torch.from_numpy(np.asarray(gray, dtype=np.float32)).view(1, 1, 512, 512) / 255.0
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12).flatten()
    topk = max(1, int(0.10 * mag.numel()))
    return float(torch.topk(mag, topk).values.mean().item())


def build_sobel_manifest(data_root, manifest_path, top_ratio):
    data_root = os.path.expanduser(data_root)
    paths = list_images(data_root)
    if not paths:
        raise RuntimeError(f"No patch images found under {data_root}")

    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (
            manifest.get("data_root") == data_root
            and math.isclose(float(manifest.get("top_ratio")), float(top_ratio))
        ):
            print(f"[E7] loaded manifest {manifest_path}", flush=True)
            return manifest["records"], manifest

    by_source = defaultdict(list)
    t0 = time.time()
    for index, path in enumerate(paths):
        meta = patch_metadata(path)
        score = sobel_energy_score(load_patch(path))
        record = {
            "path": path,
            "score": score,
            **meta,
        }
        by_source[meta["source"]].append(record)
        if (index + 1) % 500 == 0:
            print(
                f"[E7] scored {index + 1}/{len(paths)} patches "
                f"time={(time.time() - t0):.1f}s",
                flush=True,
            )

    kept = []
    source_summary = {}
    for source, records in sorted(by_source.items()):
        records = sorted(records, key=lambda item: item["score"], reverse=True)
        keep_n = max(1, int(round(len(records) * top_ratio)))
        source_kept = records[:keep_n]
        kept.extend(source_kept)
        source_summary[source] = {
            "total": len(records),
            "kept": len(source_kept),
            "threshold": source_kept[-1]["score"],
        }

    kept = sorted(kept, key=lambda item: (item["process"], item["source"], item["path"]))
    manifest = {
        "data_root": data_root,
        "top_ratio": top_ratio,
        "total_patches": len(paths),
        "kept_patches": len(kept),
        "source_summary": source_summary,
        "records": kept,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[E7] wrote manifest {manifest_path}", flush=True)
    return kept, manifest


class PatchSobelDataset(Dataset):
    def __init__(
        self,
        records,
        epoch_size,
        horizontal_flip_prob,
        active_processes,
        process_balanced=True,
        grayscale_rgb=False,
        sobel_weight_power=0.0,
        sobel_weight_floor=0.05,
    ):
        self.records = records
        self.epoch_size = epoch_size
        self.horizontal_flip_prob = horizontal_flip_prob
        self.process_balanced = process_balanced
        self.active_processes = active_processes
        self.grayscale_rgb = grayscale_rgb
        self.sobel_weight_power = sobel_weight_power
        self.sobel_weight_floor = sobel_weight_floor
        self.by_process = defaultdict(list)
        for record in records:
            self.by_process[record["process"]].append(record)
        missing = [process for process in active_processes if not self.by_process[process]]
        if missing:
            raise RuntimeError(f"Missing selected patches for processes: {missing}")
        self.record_weights = self._build_weights(self.records)
        self.by_process_weights = {
            process: self._build_weights(records)
            for process, records in self.by_process.items()
        }

    def _build_weights(self, records):
        if self.sobel_weight_power <= 0:
            return None
        scores = np.asarray([max(0.0, float(record["score"])) for record in records], dtype=np.float64)
        if scores.size == 0:
            return None
        span = float(scores.max() - scores.min())
        if span <= 1e-12:
            return [1.0 for _ in records]
        normalized = (scores - scores.min()) / span
        floor = max(0.0, float(self.sobel_weight_floor))
        weights = floor + np.power(normalized, float(self.sobel_weight_power))
        return weights.tolist()

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        if self.process_balanced:
            process = self.active_processes[index % len(self.active_processes)]
            records = self.by_process[process]
            weights = self.by_process_weights[process]
            if weights is None:
                record = random.choice(records)
            else:
                record = random.choices(records, weights=weights, k=1)[0]
        else:
            if self.record_weights is None:
                record = random.choice(self.records)
            else:
                record = random.choices(self.records, weights=self.record_weights, k=1)[0]
        image = load_patch(record["path"], grayscale_rgb=self.grayscale_rgb)
        if random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image_to_tensor(image), record["process"], 512, Path(record["path"]).name


class FixedPatchMonitorDataset(Dataset):
    def __init__(self, records, samples_per_process, active_processes, grayscale_rgb=False):
        self.records = []
        self.grayscale_rgb = grayscale_rgb
        by_process = defaultdict(list)
        for record in records:
            by_process[record["process"]].append(record)
        for process in active_processes:
            ranked = sorted(
                by_process[process],
                key=lambda item: (-item["score"], item["source"], item["path"]),
            )
            self.records.extend(ranked[:samples_per_process])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return (
            image_to_tensor(load_patch(record["path"], grayscale_rgb=self.grayscale_rgb)),
            record["process"],
            512,
            Path(record["path"]).name,
        )


def summarize_records(records):
    by_process = defaultdict(int)
    by_source = defaultdict(int)
    for record in records:
        by_process[record["process"]] += 1
        by_source[record["source"]] += 1
    return {
        "by_process": dict(sorted(by_process.items())),
        "by_source": dict(sorted(by_source.items())),
    }


def save_selected_reference_grid(dataset, path, samples_per_process, active_processes):
    buckets = {process: [] for process in active_processes}
    attempts = 0
    while any(len(bucket) < samples_per_process for bucket in buckets.values()):
        image, process, _, _ = dataset[attempts % len(dataset)]
        if process in buckets and len(buckets[process]) < samples_per_process:
            buckets[process].append(image)
        attempts += 1
        if attempts > len(dataset) * 2:
            raise RuntimeError("Could not assemble a balanced reference grid.")
    labels = []
    images = []
    for process in active_processes:
        labels.extend([process] * samples_per_process)
        images.extend(buckets[process])
    save_process_grid(torch.stack(images), labels, path, samples_per_process)


@torch.no_grad()
def generate_selected_process_grid(
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
    active_processes,
):
    from diffusers import DDIMScheduler
    from microstructure_e2_ldm import denormalize_latent

    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    base_noise = torch.randn(
        (samples_per_process, latent_mean.numel(), 64, 64),
        generator=generator,
        device=device,
    )
    latent = base_noise.repeat(len(active_processes), 1, 1, 1)
    labels = [
        process
        for process in active_processes
        for _ in range(samples_per_process)
    ]
    condition = normalized_process_features(labels, process_mean, process_std, device)
    present = torch.ones(len(labels), device=device)
    absent = torch.zeros(len(labels), device=device)
    null_condition = torch.zeros_like(condition)
    sampler = DDIMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(infer_steps, device=device)

    for timestep in sampler.timesteps:
        unconditional = model(latent, timestep, null_condition, absent)
        conditional = model(latent, timestep, condition, present)
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
    return torch.cat(decoded), labels


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E7 training requires CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    out_root = Path(os.path.expanduser(args.out_root))
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "e7_sobel_manifest.json"
    active_processes = parse_processes(args.processes)
    records, manifest = build_sobel_manifest(args.data_root, manifest_path, args.top_ratio)
    records = [record for record in records if record["process"] in active_processes]
    if not records:
        raise RuntimeError(f"No records remain for processes={active_processes}")
    summary = summarize_records(records)
    print(
        f"[E7] selected={len(records)}/{manifest['total_patches']} "
        f"top_ratio={args.top_ratio} processes={active_processes} summary={summary}",
        flush=True,
    )

    train_dataset = PatchSobelDataset(
        records,
        epoch_size=args.epoch_size,
        horizontal_flip_prob=args.horizontal_flip_prob,
        active_processes=active_processes,
        process_balanced=not args.no_process_balance,
        grayscale_rgb=args.grayscale_rgb,
        sobel_weight_power=args.sobel_weight_power,
        sobel_weight_floor=args.sobel_weight_floor,
    )
    monitor_dataset = FixedPatchMonitorDataset(
        records,
        args.samples_per_process,
        active_processes,
        grayscale_rgb=args.grayscale_rgb,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    monitor_loader = DataLoader(
        monitor_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    save_selected_reference_grid(
        monitor_dataset,
        str(out_root / "e7_real_sobel_reference.png"),
        args.samples_per_process,
        active_processes,
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
    ema_model = ConditionedUNet(
        latent_channels,
        condition_dim=len(PROCESS_FEATURE_NAMES),
    ).to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = build_noise_scheduler()

    last_path = out_root / "e7_last.pt"
    best_path = out_root / "e7_best.pt"
    step = 0
    best_monitor = float("inf")
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema_model.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
        best_monitor = float(checkpoint["best_val"])
        print(f"[E7] resumed {last_path} at step={step}", flush=True)

    with open(out_root / "e7_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "record_summary": summary,
                "manifest": {
                    key: value for key, value in manifest.items() if key != "records"
                },
            },
            handle,
            indent=2,
        )

    print(
        f"[E7] batch={args.batch_size} grad_accum={args.grad_accum} "
        f"lr={args.lr} max_steps={args.max_steps} amp={args.amp}",
        flush=True,
    )
    print(
        f"[E7] grayscale_rgb={args.grayscale_rgb} "
        f"sobel_weight_power={args.sobel_weight_power} "
        f"sobel_weight_floor={args.sobel_weight_floor}",
        flush=True,
    )

    model.train()
    t0 = time.time()
    log_path = out_root / "e7_train_log.jsonl"
    while step < args.max_steps:
        for images, labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = list(labels)
            with torch.no_grad():
                latent = normalize_latent(cae.encode(images).float(), latent_mean, latent_std)
            condition = normalized_process_features(labels, process_mean, process_std, device)
            if args.condition_dropout > 0:
                present = (
                    torch.rand(condition.shape[0], device=device)
                    >= args.condition_dropout
                ).float()
                condition = condition * present[:, None]
            else:
                present = torch.ones(condition.shape[0], device=device)

            batch = latent.shape[0]
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (batch,),
                device=device,
            ).long()
            noise = torch.randn_like(latent)
            noisy = scheduler.add_noise(latent, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=args.amp):
                prediction = model(noisy, timesteps, condition, present)
                loss = F.mse_loss(prediction, noise) / args.grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema_model, model, args.ema_decay)

            if step % args.log_every == 0:
                print(
                    f"[E7] step={step}/{args.max_steps} "
                    f"loss={loss.item() * args.grad_accum:.5f} "
                    f"condition_present={present.mean().item():.3f} "
                    f"time={(time.time() - t0) / 60.0:.1f}m",
                    flush=True,
                )

            if step > 0 and step % args.val_every == 0:
                eval_metrics = evaluate_denoising(
                    cae,
                    ema_model,
                    monitor_loader,
                    scheduler,
                    latent_mean,
                    latent_std,
                    process_mean,
                    process_std,
                    device,
                    args.val_images,
                    args.seed + step,
                )
                monitor = eval_metrics["val_noise_mse"]
                append_jsonl(
                    log_path,
                    {
                        "step": step,
                        "train_loss": loss.item() * args.grad_accum,
                        "monitor_mse": monitor,
                        "per_process": eval_metrics["per_process"],
                        "elapsed_minutes": (time.time() - t0) / 60.0,
                    },
                )
                print(f"[E7] step={step} monitor_mse={monitor:.6f}", flush=True)
                if monitor < best_monitor:
                    best_monitor = monitor
                    save_checkpoint(
                        str(best_path),
                        model,
                        ema_model,
                        optimizer,
                        scaler,
                        step,
                        best_monitor,
                        latent_mean,
                        latent_std,
                        process_mean,
                        process_std,
                        args,
                    )
                    print(f"[E7] saved best {best_path}", flush=True)

            if step > 0 and step % args.sample_every == 0:
                generated, labels_out = generate_selected_process_grid(
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
                    active_processes,
                )
                out_png = out_root / f"process_samples_step{step:06d}.png"
                save_process_grid(
                    generated,
                    labels_out,
                    str(out_png),
                    args.samples_per_process,
                )
                print(f"[E7] saved sample {out_png}", flush=True)

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(
                    str(last_path),
                    model,
                    ema_model,
                    optimizer,
                    scaler,
                    step,
                    best_monitor,
                    latent_mean,
                    latent_std,
                    process_mean,
                    process_std,
                    args,
                )
                print(f"[E7] saved last {last_path}", flush=True)

            step += 1
            if step >= args.max_steps:
                break

    save_checkpoint(
        str(last_path),
        model,
        ema_model,
        optimizer,
        scaler,
        step,
        best_monitor,
        latent_mean,
        latent_std,
        process_mean,
        process_std,
        args,
    )
    print(f"[E7] completed step={step} best_monitor={best_monitor:.6f}", flush=True)


def sample(args):
    if not torch.cuda.is_available():
        raise RuntimeError("E7 sampling requires CUDA.")
    seed_all(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    process_arg = args.processes or checkpoint_args.get("processes", "all")
    active_processes = parse_processes(process_arg)
    cae, latent_channels = load_cae(args.cae_checkpoint, device)
    model = ConditionedUNet(
        latent_channels,
        condition_dim=len(PROCESS_FEATURE_NAMES),
    ).to(device)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.eval()
    generated, labels = generate_selected_process_grid(
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
        active_processes,
    )
    save_process_grid(generated, labels, args.out_png, args.samples_per_process)
    print(f"[E7] saved sample {args.out_png}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_root", required=True)
    train_parser.add_argument("--cae_checkpoint", required=True)
    train_parser.add_argument("--e2_checkpoint", required=True)
    train_parser.add_argument("--out_root", required=True)
    train_parser.add_argument("--top_ratio", type=float, default=0.70)
    train_parser.add_argument("--grayscale_rgb", action="store_true")
    train_parser.add_argument("--sobel_weight_power", type=float, default=0.0)
    train_parser.add_argument("--sobel_weight_floor", type=float, default=0.05)
    train_parser.add_argument("--epoch_size", type=int, default=2800)
    train_parser.add_argument("--horizontal_flip_prob", type=float, default=0.50)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--grad_accum", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=5e-5)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--max_steps", type=int, default=40000)
    train_parser.add_argument("--condition_dropout", type=float, default=0.10)
    train_parser.add_argument("--ema_decay", type=float, default=0.999)
    train_parser.add_argument("--grad_clip", type=float, default=1.0)
    train_parser.add_argument("--val_every", type=int, default=500)
    train_parser.add_argument("--val_images", type=int, default=16)
    train_parser.add_argument("--sample_every", type=int, default=2000)
    train_parser.add_argument("--samples_per_process", type=int, default=4)
    train_parser.add_argument("--sample_seed", type=int, default=7070)
    train_parser.add_argument("--infer_steps", type=int, default=100)
    train_parser.add_argument("--guidance_scale", type=float, default=1.5)
    train_parser.add_argument("--save_every", type=int, default=1000)
    train_parser.add_argument("--log_every", type=int, default=25)
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--no_process_balance", action="store_true")
    train_parser.add_argument("--processes", default="all")
    train_parser.add_argument("--seed", type=int, default=0)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--cae_checkpoint", required=True)
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--out_png", required=True)
    sample_parser.add_argument("--samples_per_process", type=int, default=4)
    sample_parser.add_argument("--infer_steps", type=int, default=250)
    sample_parser.add_argument("--guidance_scale", type=float, default=1.5)
    sample_parser.add_argument("--processes", default=None)
    sample_parser.add_argument("--seed", type=int, default=7070)
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
