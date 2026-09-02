#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ddpm_microstructure_pipeline_attn1632.py

Two subcommands:
1) make_patches: sample 512x512 patches from N source images
2) train_ddpm  : train unconditional DDPM (diffusers UNet2DModel) with optional DDP (multi-GPU)

Designed for:
- NVIDIA V100 32GB x4 (original baseline)
- Driver 470 / CUDA 11.4
- torch 1.12.1+cu113
- diffusers 0.19.3

Update in this version:
- Use attention at 16x16 + 32x32 by setting: --attn_level 32
  (i.e., "32" means {32,16} attention; "16" means {16} only)

Run:
  # 1) make patches
  python ddpm_microstructure_pipeline.py make_patches \
    --src_dir  ~/projects/microstructure_images/500um_Ripples \
    --out_dir  ~/projects/microstructure_images/patches_512_from8 \
    --num_images 8 --patch_size 512 --patches_per_image 500 --border 10 --seed 0

  # 2) single GPU train
  python ddpm_microstructure_pipeline.py train_ddpm \
    --data_root ~/projects/microstructure_images/patches_512_from8 \
    --out_root  ~/projects/microstructure_images/diffusion_runs/ddpm_uncond_512_from8_struct \
    --resolution 512 --batch_size 1 --grad_accum 4 \
    --max_steps 20000 --sample_every 1000 --save_every 5000 --amp --seed 0 \
    --attn_level 32

  # 3) multi-GPU DDP train (recommended)
  torchrun --standalone --nproc_per_node=4 ddpm_microstructure_pipeline.py train_ddpm \
    --data_root ~/projects/microstructure_images/patches_512_from8 \
    --out_root  ~/projects/microstructure_images/diffusion_runs/ddpm_uncond_512_from8_struct_ddp \
    --resolution 512 --batch_size 1 --grad_accum 2 \
    --max_steps 20000 --sample_every 1000 --save_every 5000 --amp --seed 0 \
    --attn_level 32

Notes:
- Global batch = world_size * batch_size * grad_accum
- OOM avoidance: attention only at low-res (16x16) or mid+low (32x32 + 16x16)
"""

import os
import math
import time
import glob
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Utilities
# -------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(root: str):
    files = []
    root = os.path.expanduser(root)
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    files = sorted(set(files))
    return files


def ensure_rgb(im: Image.Image) -> Image.Image:
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def ddp_setup():
    """
    torchrun sets:
      LOCAL_RANK, RANK, WORLD_SIZE
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    return False, 0, 0, 1


def ddp_cleanup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist

        dist.destroy_process_group()


# -------------------------
# make_patches
# -------------------------
def sample_patches_from_image(im: Image.Image, patch: int, n_patches: int, border: int):
    W, H = im.size
    if W < patch or H < patch:
        scale = patch / float(min(W, H))
        im = im.resize((int(W * scale) + 1, int(H * scale) + 1), Image.BICUBIC)
        W, H = im.size

    patches = []
    max_x = max(border, W - patch - border)
    max_y = max(border, H - patch - border)
    for _ in range(n_patches):
        x0 = random.randint(border, max_x)
        y0 = random.randint(border, max_y)
        crop = im.crop((x0, y0, x0 + patch, y0 + patch))
        patches.append(crop)
    return patches


def run_make_patches(args):
    seed_all(args.seed)
    src_dir = os.path.expanduser(args.src_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    imgs = list_images(src_dir)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found under: {src_dir}")

    if args.shuffle:
        random.shuffle(imgs)
    selected = imgs[: args.num_images]

    print(f"[MAKE_PATCHES] src_dir={src_dir}")
    print(f"[MAKE_PATCHES] found_images={len(imgs)}  selected={len(selected)}")
    print(f"[MAKE_PATCHES] out_dir={out_dir}")
    print(
        f"[MAKE_PATCHES] patch_size={args.patch_size} patches_per_image={args.patches_per_image} "
        f"border={args.border} seed={args.seed}"
    )

    total = 0
    for i, p in enumerate(selected, 1):
        im = ensure_rgb(Image.open(p))
        patches = sample_patches_from_image(
            im, patch=args.patch_size, n_patches=args.patches_per_image, border=args.border
        )
        stem = Path(p).stem
        for j, patch_im in enumerate(patches):
            out = os.path.join(out_dir, f"{stem}_p{j:04d}.png")
            patch_im.save(out, "PNG")
        total += len(patches)
        print(f"[{i}/{len(selected)}] {os.path.basename(p)} -> {len(patches)} patches")

    print(f"[DONE] total_patches={total}")
    print(f"[DONE] patch_dir={out_dir}")


# -------------------------
# Dataset
# -------------------------
class ImageFolderSquare(Dataset):
    def __init__(self, root: str, resolution: int):
        self.root = os.path.expanduser(root)
        self.paths = list_images(self.root)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {self.root}")
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = ensure_rgb(Image.open(p))

        # center-crop to square then resize
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        if s != self.resolution:
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        # basic augmentation
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        x = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # [0,255] -> [-1,1]
        x = x.permute(2, 0, 1)  # HWC -> CHW
        return x


# -------------------------
# Sampling / checkpoint
# -------------------------
@torch.no_grad()
def sample_images(model, noise_scheduler, device, out_path, resolution: int, n: int, num_inference_steps: int, amp: bool):
    model.eval()

    x = torch.randn(n, 3, resolution, resolution, device=device)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    for t in noise_scheduler.timesteps:
        with torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda"))):
            eps = model(x, t).sample
        x = noise_scheduler.step(eps, t, x).prev_sample

    x = (x.clamp(-1, 1) + 1) * 127.5
    x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    grid = Image.new("RGB", (cols * resolution, rows * resolution))
    for i in range(n):
        r, c = divmod(i, cols)
        grid.paste(Image.fromarray(x[i]), (c * resolution, r * resolution))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    model.train()


def save_ckpt(model, noise_scheduler, optimizer, lr_sched, scaler, out_dir, step: int):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, f"ckpt_step{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # unwrap DDP
    raw_model = model.module if hasattr(model, "module") else model

    raw_model.save_pretrained(ckpt_dir, safe_serialization=True)
    noise_scheduler.save_pretrained(ckpt_dir)

    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "lr_sched": lr_sched.state_dict() if lr_sched is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        os.path.join(ckpt_dir, "train_state.pt"),
    )
    print(f"[CKPT] saved: {ckpt_dir}")


# -------------------------
# Model
# -------------------------
def build_unet(resolution: int, attn_level: str = "16"):
    """
    attn_level:
      - "none": no attention
      - "16":  attention only at 16x16 (safest for 512)
      - "32":  attention at 32x32 AND 16x16 (recommended for better edge/ripple learning)
    """
    from diffusers import UNet2DModel

    # smaller channels for stability on 512 (you can scale up later on H100 if desired)
    block_out_channels = (64, 128, 128, 256, 256)

    if attn_level == "none":
        down = ("DownBlock2D",) * 5
        up = ("UpBlock2D",) * 5
    elif attn_level == "16":
        # attention only at the smallest scale (16x16)
        down = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D")
        up = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    elif attn_level == "32":
        # attention at 32x32 + 16x16  (this is your requested "16+32")
        down = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    else:
        raise ValueError(f"Unknown attn_level: {attn_level}")

    model = UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=down,
        up_block_types=up,
    )
    return model


# -------------------------
# Training
# -------------------------
def run_train_ddpm(args):
    seed_all(args.seed)

    is_ddp, local_rank, rank, world_size = ddp_setup()
    is_main = (rank == 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_ddp:
        device = f"cuda:{local_rank}"

    from diffusers import DDPMScheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup

    ds = ImageFolderSquare(args.data_root, args.resolution)

    sampler = None
    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if is_main:
        print(f"[DATA] images={len(ds)} resolution={args.resolution} batch={args.batch_size} device={device} ddp={is_ddp} world={world_size}")
        print(f"[CFG] attn_level={args.attn_level} grad_accum={args.grad_accum} amp={args.amp}")

    model = build_unet(args.resolution, attn_level=args.attn_level).to(device)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    lr_sched = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    amp_enabled = bool(args.amp) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    os.makedirs(os.path.expanduser(args.out_root), exist_ok=True)

    step = 0
    t0 = time.time()
    model.train()

    dl_iter = iter(dl)

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                x0 = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                x0 = next(dl_iter)

            x0 = x0.to(device, non_blocking=True)
            b = x0.shape[0]

            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device).long()
            noise = torch.randn_like(x0)
            xt = noise_scheduler.add_noise(x0, noise, t)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(xt, t).sample
                loss = F.mse_loss(pred, noise) / float(args.grad_accum)

            scaler.scale(loss).backward()
            total_loss += float(loss.item())

        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()

        if is_main and (step % args.log_every == 0):
            dt = time.time() - t0
            lr = lr_sched.get_last_lr()[0]
            loss_report = total_loss * float(args.grad_accum)  # per-(effective)batch loss
            print(f"[TRAIN] step={step}/{args.max_steps} loss={loss_report:.4f} lr={lr:.2e} time={dt/60:.1f}m")

        if is_main and (step > 0) and (step % args.sample_every == 0):
            out_path = os.path.join(os.path.expanduser(args.out_root), f"samples_step{step:06d}.png")
            sample_images(
                model.module if hasattr(model, "module") else model,
                noise_scheduler,
                device,
                out_path,
                args.resolution,
                n=args.sample_n,
                num_inference_steps=args.sample_steps,
                amp=amp_enabled,
            )
            print(f"[SAMPLE] saved: {out_path}")

        if is_main and (step > 0) and (step % args.save_every == 0):
            save_ckpt(model, noise_scheduler, optimizer, lr_sched, scaler, os.path.expanduser(args.out_root), step)

        step += 1

    if is_main:
        save_ckpt(model, noise_scheduler, optimizer, lr_sched, scaler, os.path.expanduser(args.out_root), step)
        out_path = os.path.join(os.path.expanduser(args.out_root), f"samples_step{step:06d}.png")
        sample_images(
            model.module if hasattr(model, "module") else model,
            noise_scheduler,
            device,
            out_path,
            args.resolution,
            n=args.sample_n,
            num_inference_steps=max(args.sample_steps, 300),
            amp=amp_enabled,
        )
        print("[DONE] training finished.")

    ddp_cleanup()


# -------------------------
# CLI
# -------------------------
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_mk = sub.add_parser("make_patches", help="Sample patches from N images into a patch dataset")
    p_mk.add_argument("--src_dir", type=str, required=True)
    p_mk.add_argument("--out_dir", type=str, required=True)
    p_mk.add_argument("--num_images", type=int, default=8)
    p_mk.add_argument("--patch_size", type=int, default=512)
    p_mk.add_argument("--patches_per_image", type=int, default=500)
    p_mk.add_argument("--border", type=int, default=10)
    p_mk.add_argument("--shuffle", action="store_true", help="Shuffle image list before selecting num_images")
    p_mk.add_argument("--seed", type=int, default=0)

    p_tr = sub.add_parser("train_ddpm", help="Train unconditional DDPM on a folder of images")
    p_tr.add_argument("--data_root", type=str, required=True)
    p_tr.add_argument("--out_root", type=str, required=True)
    p_tr.add_argument("--resolution", type=int, default=512)
    p_tr.add_argument("--batch_size", type=int, default=1)
    p_tr.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (per rank)")
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--weight_decay", type=float, default=1e-4)
    p_tr.add_argument("--max_steps", type=int, default=20000)
    p_tr.add_argument("--warmup_steps", type=int, default=500)
    p_tr.add_argument("--num_workers", type=int, default=2)
    p_tr.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp autocast + GradScaler")
    p_tr.add_argument("--seed", type=int, default=0)

    # logging / saving
    p_tr.add_argument("--log_every", type=int, default=50)
    p_tr.add_argument("--save_every", type=int, default=5000)
    p_tr.add_argument("--sample_every", type=int, default=1000)
    p_tr.add_argument("--sample_n", type=int, default=9)
    p_tr.add_argument("--sample_steps", type=int, default=200)

    # model controls
    p_tr.add_argument(
        "--attn_level",
        type=str,
        default="32",  # default to 16+32, as requested
        choices=["none", "16", "32"],
        help="Where to use self-attention. '16' = 16x16 only. '32' = 32x32 + 16x16 (recommended).",
    )

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "make_patches":
        run_make_patches(args)
    elif args.cmd == "train_ddpm":
        run_train_ddpm(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()