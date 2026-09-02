#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ddpm_microstructure_pipeline_attn643216_vpred_ema.py

Upgrades (as requested):
1) True grayscale (1-channel) training (in/out_channels=1, dataset reads "L")
2) Wider UNet capacity (default block_out_channels=(128, 256, 256, 512, 512))
3) v-prediction training (prediction_type="v_prediction")
   + optional SNR weighting via --snr_gamma (0 disables)
4) EMA for sampling/checkpointing (sampling uses EMA weights)

Attention:
--attn_level 32 : attention at {32,16}
--attn_level 64 : attention at {64,32,16}  (recommended if memory allows)

Notes:
- For 512 resolution, down stages are: 512->256->128->64->32->16 (5 blocks)
- "64 attention" means AttnDownBlock2D at the 64x64 stage (block index 2)

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

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


# -------------------------
# Utils
# -------------------------
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
    return sorted(set(files))


def ddp_setup():
    """
    torchrun sets: LOCAL_RANK, RANK, WORLD_SIZE
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


def is_main_process(rank: int) -> bool:
    return rank == 0


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
        # keep color in source crop stage; you can convert later in augmentation pipeline if desired
        im = Image.open(p).convert("RGB")
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
# Dataset (TRUE GRAYSCALE)
# -------------------------
class GrayImageFolderSquare(Dataset):
    """
    Reads patches as grayscale 'L' -> tensor shape [1, H, W] in [-1, 1]
    Adds lightweight on-the-fly geometric aug (optional)
    """
    def __init__(self, root: str, resolution: int, aug: bool = True):
        self.root = os.path.expanduser(root)
        self.paths = list_images(self.root)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {self.root}")
        self.resolution = resolution
        self.aug = aug

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("L")  # 1-channel

        # center-crop to square then resize
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        if s != self.resolution:
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)

        if self.aug:
            # flips
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.2:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # random rotate by 0/90/180/270
            if random.random() < 0.3:
                k = random.randint(0, 3)
                if k == 1:
                    img = img.transpose(Image.ROTATE_90)
                elif k == 2:
                    img = img.transpose(Image.ROTATE_180)
                elif k == 3:
                    img = img.transpose(Image.ROTATE_270)

        x = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0  # [H,W] in [-1,1]
        x = x.unsqueeze(0)  # [1,H,W]
        return x


# -------------------------
# EMA
# -------------------------
@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    """
    ema = decay * ema + (1-decay) * model
    """
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if k in msd:
            esd[k].mul_(decay).add_(msd[k], alpha=1.0 - decay)
    ema_model.load_state_dict(esd, strict=True)


def copy_model(model: torch.nn.Module) -> torch.nn.Module:
    import copy
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m


# -------------------------
# Sampling / checkpoint
# -------------------------
@torch.no_grad()
def sample_images_1ch(
    model,
    noise_scheduler,
    device,
    out_path,
    resolution: int,
    n: int,
    num_inference_steps: int,
    amp: bool,
):
    """
    1-channel sampling -> saves as grayscale grid (mode 'L')
    """
    model.eval()
    x = torch.randn(n, 1, resolution, resolution, device=device)

    noise_scheduler.set_timesteps(num_inference_steps, device=device)

    for t in noise_scheduler.timesteps:
        with torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda"))):
            out = model(x, t)
            pred = out.sample
        x = noise_scheduler.step(pred, t, x).prev_sample

    # [-1,1] -> [0,255]
    x = (x.clamp(-1, 1) + 1) * 127.5
    x = x.to(torch.uint8).squeeze(1).cpu().numpy()  # [n,H,W]

    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    grid = Image.new("L", (cols * resolution, rows * resolution))
    for i in range(n):
        r, c = divmod(i, cols)
        grid.paste(Image.fromarray(x[i], mode="L"), (c * resolution, r * resolution))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    model.train()


def save_ckpt(
    model,
    ema_model,
    noise_scheduler,
    optimizer,
    lr_sched,
    scaler,
    out_dir,
    step: int,
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, f"ckpt_step{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # unwrap DDP
    raw_model = model.module if hasattr(model, "module") else model

    # save model + scheduler (diffusers format)
    raw_model.save_pretrained(ckpt_dir, safe_serialization=True)
    noise_scheduler.save_pretrained(ckpt_dir)

    # save EMA weights
    if ema_model is not None:
        torch.save(ema_model.state_dict(), os.path.join(ckpt_dir, "ema_state.pt"))

    # training state
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
def build_unet_1ch(resolution: int, attn_level: str = "64"):
    """
    attn_level:
      - "none": no attention
      - "32"  : attention at 32x32 + 16x16
      - "64"  : attention at 64x64 + 32x32 + 16x16 (recommended)
    """
    from diffusers import UNet2DModel

    # Wider capacity (requested)
    block_out_channels = (128, 256, 256, 512, 512)

    if attn_level == "none":
        down = ("DownBlock2D",) * 5
        up = ("UpBlock2D",) * 5
    elif attn_level == "32":
        # attention at {32,16}: blocks 3 and 4 in down path
        down = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up = ("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    elif attn_level == "64":
        # attention at {64,32,16}: blocks 2,3,4 in down path
        down = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")
        up = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
    else:
        raise ValueError(f"Unknown attn_level: {attn_level}")

    model = UNet2DModel(
        sample_size=resolution,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=block_out_channels,
        down_block_types=down,
        up_block_types=up,
    )
    return model


# -------------------------
# v-pred + optional SNR weighting
# -------------------------
def compute_snr(noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """
    Returns SNR(t) for given timesteps.
    SNR = alpha_cumprod / (1 - alpha_cumprod)
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    a = alphas_cumprod[timesteps]
    snr = a / (1.0 - a)
    return snr


# -------------------------
# Training
# -------------------------
def run_train_ddpm(args):
    seed_all(args.seed)

    is_ddp, local_rank, rank, world_size = ddp_setup()
    main = is_main_process(rank)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if is_ddp:
        device = f"cuda:{local_rank}"

    from diffusers import DDPMScheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup

    # dataset: grayscale
    ds = GrayImageFolderSquare(args.data_root, args.resolution, aug=args.on_the_fly_aug)

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

    if main:
        print(f"[DATA] images={len(ds)} resolution={args.resolution} batch={args.batch_size} device={device} ddp={is_ddp} world={world_size}")
        print(f"[CFG] attn_level={args.attn_level} grad_accum={args.grad_accum} amp={args.amp} ema={args.ema_decay}")
        print(f"[CFG] v_pred=True snr_gamma={args.snr_gamma} on_the_fly_aug={args.on_the_fly_aug}")

    # model (1ch, wider, attention)
    model = build_unet_1ch(args.resolution, attn_level=args.attn_level).to(device)

    # EMA model kept on main process only
    ema_model = None
    if main:
        ema_model = copy_model(model)

    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    # Scheduler with v-pred
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        prediction_type="v_prediction",
    )

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

    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

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

            x0 = x0.to(device, non_blocking=True)  # [B,1,H,W]
            b = x0.shape[0]

            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (b,), device=device).long()
            noise = torch.randn_like(x0)
            xt = noise_scheduler.add_noise(x0, noise, t)

            # v-target
            target_v = noise_scheduler.get_velocity(x0, noise, t)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(xt, t).sample
                loss = F.mse_loss(pred, target_v, reduction="none")

                # optional SNR weighting (gamma=0 disables)
                if args.snr_gamma > 0:
                    snr = compute_snr(noise_scheduler, t)  # [B]
                    # common weighting: min(snr, gamma) / snr
                    w = torch.minimum(snr, torch.full_like(snr, float(args.snr_gamma))) / snr
                    w = w.view(b, 1, 1, 1)
                    loss = loss * w

                loss = loss.mean() / float(args.grad_accum)

            scaler.scale(loss).backward()
            total_loss += float(loss.item())

        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()

        # EMA update after optimizer step (main process only)
        if main and ema_model is not None:
            raw = model.module if hasattr(model, "module") else model
            ema_update(ema_model, raw, decay=float(args.ema_decay))

        if main and (step % args.log_every == 0):
            dt = time.time() - t0
            lr = lr_sched.get_last_lr()[0]
            loss_report = total_loss * float(args.grad_accum)
            print(f"[TRAIN] step={step}/{args.max_steps} loss={loss_report:.4f} lr={lr:.2e} time={dt/60:.1f}m")

        if main and (step > 0) and (step % args.sample_every == 0):
            out_path = os.path.join(out_root, f"samples_step{step:06d}.png")
            # sample with EMA if available; otherwise raw
            sampler_model = ema_model if ema_model is not None else (model.module if hasattr(model, "module") else model)
            sample_images_1ch(
                sampler_model,
                noise_scheduler,
                device,
                out_path,
                args.resolution,
                n=args.sample_n,
                num_inference_steps=args.sample_steps,
                amp=amp_enabled,
            )
            print(f"[SAMPLE] saved: {out_path}")

        if main and (step > 0) and (step % args.save_every == 0):
            raw = model.module if hasattr(model, "module") else model
            save_ckpt(raw, ema_model, noise_scheduler, optimizer, lr_sched, scaler, out_root, step)

        step += 1

    if main:
        raw = model.module if hasattr(model, "module") else model
        save_ckpt(raw, ema_model, noise_scheduler, optimizer, lr_sched, scaler, out_root, step)

        out_path = os.path.join(out_root, f"samples_step{step:06d}.png")
        sampler_model = ema_model if ema_model is not None else raw
        sample_images_1ch(
            sampler_model,
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
    p_mk.add_argument("--shuffle", action="store_true")
    p_mk.add_argument("--seed", type=int, default=0)

    p_tr = sub.add_parser("train_ddpm", help="Train unconditional DDPM on a folder of images (1ch grayscale)")
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
    p_tr.add_argument("--amp", action="store_true")
    p_tr.add_argument("--seed", type=int, default=0)

    # logging / saving
    p_tr.add_argument("--log_every", type=int, default=50)
    p_tr.add_argument("--save_every", type=int, default=5000)
    p_tr.add_argument("--sample_every", type=int, default=1000)
    p_tr.add_argument("--sample_n", type=int, default=9)
    p_tr.add_argument("--sample_steps", type=int, default=200)

    # requested upgrades
    p_tr.add_argument(
        "--attn_level",
        type=str,
        default="64",
        choices=["none", "32", "64"],
        help="'32' = attn at 32+16. '64' = attn at 64+32+16 (recommended).",
    )
    p_tr.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay. Typical: 0.999~0.9999. (bigger=slower, smoother)",
    )
    p_tr.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma. Set 0 to disable. Typical: 5 or 10.",
    )
    p_tr.add_argument(
        "--on_the_fly_aug",
        action="store_true",
        help="Enable light on-the-fly geometric augmentation (flip/rotate).",
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