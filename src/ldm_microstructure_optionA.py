#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ldm_microstructure_optionA.py

Option A:
Structured-CAE/VAE + latent-first residual diffusion for microstructure patches.

Pipeline:
  1) filter_patches
  2) train_vae
  3) train_optionA
  4) sample_optionA

Core idea:
  - structure latent   z_struct = E(blur(x))
  - full latent        z_full   = E(x)
  - residual latent    z_res    = z_full - z_struct

Diffusion:
  Stage A: unconditional DDPM on z_struct
  Stage B: conditional DDPM on z_res given z_struct

Sampling:
  sample z_struct -> sample z_res conditioned on z_struct -> z = z_struct + z_res -> decode
"""

import os
import math
import time
import glob
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusers import UNet2DModel, DDPMScheduler


# =========================================================
# Utilities
# =========================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def list_images(root: str):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(files))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_ddp():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def ddp_rank():
    return int(os.environ.get("RANK", "0"))


def ddp_world():
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_setup():
    if not is_ddp():
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(ddp_local_rank())
    return True


def ddp_cleanup():
    if is_ddp() and dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return (not is_ddp()) or ddp_rank() == 0


def barrier():
    if is_ddp() and dist.is_initialized():
        dist.barrier()


def log0(msg: str):
    if is_main():
        print(msg, flush=True)


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def pil_load_rgb(p: str):
    im = Image.open(p)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def pil_center_crop_resize(im: Image.Image, resolution: int):
    w, h = im.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    im = im.crop((left, top, left + s, top + s))
    if s != resolution:
        im = im.resize((resolution, resolution), Image.BICUBIC)
    return im


def to_tensor_m11(im: Image.Image):
    x = torch.from_numpy(np.array(im)).float() / 127.5 - 1.0
    x = x.permute(2, 0, 1).contiguous()
    return x


def tensor_to_pil_m11(x: torch.Tensor):
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


# =========================================================
# Dataset
# =========================================================
class PatchFolder(Dataset):
    def __init__(self, root: str, resolution: int = 512, flip: bool = True):
        self.paths = list_images(root)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {root}")
        self.res = resolution
        self.flip = flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = pil_load_rgb(p)
        img = pil_center_crop_resize(img, self.res)
        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = to_tensor_m11(img)
        return x


# =========================================================
# Sobel filter scoring
# =========================================================
def sobel_energy_score_rgb_pil(im: Image.Image) -> float:
    g = im.convert("L")
    x = torch.from_numpy(np.array(g)).float().unsqueeze(0).unsqueeze(0) / 255.0

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)

    flat = mag.flatten()
    k = max(1, int(0.10 * flat.numel()))
    topk = torch.topk(flat, k, largest=True).values
    return float(topk.mean().item())


def cmd_filter_patches(args):
    seed_all(args.seed)

    in_root = os.path.expanduser(args.in_root)
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    paths = list_images(in_root)
    if len(paths) == 0:
        raise RuntimeError(f"No patches found under: {in_root}")

    log0(f"[FILTER] in_root={in_root}")
    log0(f"[FILTER] found={len(paths)}  top_ratio={args.top_ratio}  out_root={out_root}")

    scored = []
    t0 = time.time()
    for i, p in enumerate(paths):
        im = pil_load_rgb(p)
        im = pil_center_crop_resize(im, args.resolution)
        s = sobel_energy_score_rgb_pil(im)
        scored.append((s, p))
        if (i + 1) % 200 == 0:
            log0(f"[FILTER] scored {i+1}/{len(paths)}  time={(time.time() - t0):.1f}s")

    scored.sort(key=lambda x: x[0], reverse=True)
    keep_n = int(round(args.top_ratio * len(scored)))
    keep_n = max(1, min(len(scored), keep_n))

    kept = scored[:keep_n]
    thr = kept[-1][0]
    log0(f"[FILTER] keep_n={keep_n}/{len(scored)}  score_threshold≈{thr:.6f}")

    meta = []
    for rank, (s, p) in enumerate(kept):
        stem = Path(p).stem
        dst = os.path.join(out_root, f"{rank:06d}_{stem}.png")
        im = pil_load_rgb(p)
        im = pil_center_crop_resize(im, args.resolution)
        im.save(dst, "PNG")
        meta.append({"rank": rank, "score": float(s), "src": p, "dst": dst})

    save_json(os.path.join(out_root, "filter_meta.json"), {
        "in_root": in_root,
        "out_root": out_root,
        "resolution": args.resolution,
        "top_ratio": args.top_ratio,
        "keep_n": keep_n,
        "threshold_score": float(thr),
        "seed": args.seed,
    })
    save_json(os.path.join(out_root, "filter_list.json"), meta)
    log0("[FILTER] done.")


# =========================================================
# Image ops / losses
# =========================================================
def gaussian_kernel2d(kernel_size=21, sigma=3.0, channels=3, device="cpu", dtype=torch.float32):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur(x, kernel_size=21, sigma=3.0):
    c = x.shape[1]
    k = gaussian_kernel2d(kernel_size, sigma, c, x.device, x.dtype)
    pad = kernel_size // 2
    return F.conv2d(x, k, padding=pad, groups=c)


def sobel_edges(x):
    # x in [-1,1], [B,C,H,W]
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag


def edge_loss(x_rec, x):
    return F.l1_loss(sobel_edges(x_rec), sobel_edges(x))


def fft_mag(x):
    # use grayscale
    g = x.mean(dim=1, keepdim=True)
    f = torch.fft.rfft2(g, norm="ortho")
    m = torch.abs(f)
    return m


def fft_loss(x_rec, x):
    return F.l1_loss(fft_mag(x_rec), fft_mag(x))


def ssim_loss(x, y, window=11):
    # x,y in [-1,1], map to [0,1]
    x = (x + 1.0) * 0.5
    y = (y + 1.0) * 0.5

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pad = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return 1.0 - ssim_map.mean()


def latent_decorrelation_loss(mu):
    # mu: [B,C,H,W] -> global pooled [B,C]
    z = mu.mean(dim=(2, 3))
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    offdiag = cov - torch.diag(torch.diag(cov))
    return (offdiag ** 2).mean()


# =========================================================
# Structured VAE
# =========================================================
class SimpleVAE(nn.Module):
    def __init__(self, z_channels=4):
        super().__init__()
        ch = 64
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.SiLU(),         # 256
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),    # 128
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.SiLU(),# 64
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(ch * 4, z_channels, 1)
        self.to_logvar = nn.Conv2d(ch * 4, z_channels, 1)

        self.dec = nn.Sequential(
            nn.Conv2d(z_channels, ch * 4, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.SiLU(), # 128
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU(),     # 256
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.SiLU(),         # 512
            nn.Conv2d(ch, 3, 3, 1, 1),
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar, z


def vae_kl(mu, logvar):
    return 0.5 * torch.mean(torch.exp(logvar) + mu * mu - 1.0 - logvar)


@torch.no_grad()
def vae_recon_grid(vae, dl, device, out_png, n=9, blur_kernel=21, blur_sigma=3.0):
    vae.eval()
    xs = []
    for x in dl:
        xs.append(x)
        if len(xs) * x.size(0) >= n:
            break
    x = torch.cat(xs, dim=0)[:n].to(device)
    x_blur = gaussian_blur(x, blur_kernel, blur_sigma)
    mu_full, _ = vae.encode(x)
    mu_struct, _ = vae.encode(x_blur)
    x_rec = vae.decode(mu_full)
    x_struct = vae.decode(mu_struct)

    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols * 512, rows * 512 * 3))
    for i in range(n):
        r, c = divmod(i, cols)
        canvas.paste(tensor_to_pil_m11(x[i]),        (c * 512, r * 512 * 3 + 0))
        canvas.paste(tensor_to_pil_m11(x_struct[i]), (c * 512, r * 512 * 3 + 512))
        canvas.paste(tensor_to_pil_m11(x_rec[i]),    (c * 512, r * 512 * 3 + 1024))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)
    vae.train()


def save_ckpt(model, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"model": model.state_dict()}
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_ckpt(model, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def cmd_train_vae(args):
    ddp = ddp_setup()
    seed_all(args.seed + ddp_rank())

    data_root = os.path.expanduser(args.data_root)
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    device = f"cuda:{ddp_local_rank()}" if torch.cuda.is_available() else "cpu"

    ds = PatchFolder(data_root, resolution=512, flip=True)
    sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True) if ddp else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    vae = SimpleVAE(z_channels=args.z_channels).to(device)
    if ddp:
        vae = torch.nn.parallel.DistributedDataParallel(
            vae,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
        )

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    ckpt_path = os.path.join(out_root, "vae_last.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        log0(f"[VAE] resume from {ckpt_path}")
        ckpt = load_ckpt(vae.module if ddp else vae, ckpt_path, map_location="cpu")
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    log0(f"[VAE] data={len(ds)} batch={args.batch_size} ddp={ddp} world={ddp_world()} device={device}")
    log0(f"[VAE] z_channels={args.z_channels} amp={args.amp}")
    log0(f"[VAE] weights: l1={args.w_l1} ssim={args.w_ssim} edge={args.w_edge} fft={args.w_fft} kl={args.w_kl} decor={args.w_decor}")

    t0 = time.time()
    step = start_step
    vae.train()

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x in dl:
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                x_rec, mu, logvar, z = vae(x)
                l1 = F.l1_loss(x_rec, x)
                l_ssim = ssim_loss(x_rec, x)
                l_edge = edge_loss(x_rec, x)
                l_fft = fft_loss(x_rec, x)
                l_kl = vae_kl(mu, logvar)
                l_decor = latent_decorrelation_loss(mu)

                loss = (
                    args.w_l1 * l1
                    + args.w_ssim * l_ssim
                    + args.w_edge * l_edge
                    + args.w_fft * l_fft
                    + args.w_kl * l_kl
                    + args.w_decor * l_decor
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                log0(
                    f"[VAE] step={step}/{args.max_steps} "
                    f"loss={loss.item():.4f} l1={l1.item():.4f} ssim={l_ssim.item():.4f} "
                    f"edge={l_edge.item():.4f} fft={l_fft.item():.4f} kl={l_kl.item():.4f} decor={l_decor.item():.4f} "
                    f"time={dt:.1f}m"
                )

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"vae_recon_step{step:06d}.png")
                base_vae = vae.module if ddp else vae
                vae_recon_grid(base_vae, dl, device, out_png, n=9, blur_kernel=args.blur_kernel, blur_sigma=args.blur_sigma)
                log0(f"[VAE] saved recon: {out_png}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base_vae = vae.module if ddp else vae
                save_ckpt(base_vae, ckpt_path, extra={
                    "step": step,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                })
                log0(f"[VAE] saved: {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break

    if is_main():
        base_vae = vae.module if ddp else vae
        save_ckpt(base_vae, ckpt_path, extra={
            "step": step,
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        })
        log0(f"[VAE] done. saved: {ckpt_path}")

    barrier()
    ddp_cleanup()


# =========================================================
# Latent helpers
# =========================================================
@torch.no_grad()
def encode_latents(vae: SimpleVAE, x: torch.Tensor):
    mu, logvar = vae.encode(x)
    return mu


@torch.no_grad()
def decode_latents(vae: SimpleVAE, z: torch.Tensor):
    return vae.decode(z)


# =========================================================
# Option A diffusion models
# =========================================================
def build_struct_unet():
    return UNet2DModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    )


def build_res_unet():
    # input = [z_res_t, z_struct_cond] => 8 channels
    return UNet2DModel(
        sample_size=64,
        in_channels=8,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    )


@torch.no_grad()
def sample_optionA(
    vae,
    unet_struct,
    unet_res,
    sched_struct,
    sched_res,
    device,
    out_png,
    n=9,
    infer_steps=250,
    cond_noise_std=0.0,
):
    vae.eval()
    unet_struct.eval()
    unet_res.eval()

    # Stage 1: sample structure latent
    z_struct = torch.randn(n, 4, 64, 64, device=device)
    sched_struct.set_timesteps(infer_steps, device=device)
    for t in sched_struct.timesteps:
        eps = unet_struct(z_struct, t).sample
        z_struct = sched_struct.step(eps, t, z_struct).prev_sample

    # Stage 2: sample residual latent conditioned on structure latent
    z_res = torch.randn(n, 4, 64, 64, device=device)
    sched_res.set_timesteps(infer_steps, device=device)

    cond = z_struct
    if cond_noise_std > 0:
        cond = cond + cond_noise_std * torch.randn_like(cond)

    for t in sched_res.timesteps:
        inp = torch.cat([z_res, cond], dim=1)
        eps = unet_res(inp, t).sample
        z_res = sched_res.step(eps, t, z_res).prev_sample

    z_final = z_struct + z_res
    x_struct = decode_latents(vae, z_struct)
    x_final = decode_latents(vae, z_final)

    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols * 512, rows * 512 * 2))

    for i in range(n):
        r, c = divmod(i, cols)
        canvas.paste(tensor_to_pil_m11(x_struct[i]), (c * 512, r * 512 * 2 + 0))
        canvas.paste(tensor_to_pil_m11(x_final[i]),  (c * 512, r * 512 * 2 + 512))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)

    unet_struct.train()
    unet_res.train()
    vae.train()


def cmd_train_optionA(args):
    ddp = ddp_setup()
    seed_all(args.seed + ddp_rank())

    data_root = os.path.expanduser(args.data_root)
    out_root = os.path.expanduser(args.out_root)
    vae_ckpt = os.path.expanduser(args.vae_ckpt)

    os.makedirs(out_root, exist_ok=True)
    device = f"cuda:{ddp_local_rank()}" if torch.cuda.is_available() else "cpu"

    ds = PatchFolder(data_root, resolution=512, flip=True)
    sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True) if ddp else None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    vae = SimpleVAE(z_channels=4).to(device)
    load_ckpt(vae, vae_ckpt, map_location="cpu")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    unet_struct = build_struct_unet().to(device)
    unet_res = build_res_unet().to(device)

    if ddp:
        unet_struct = torch.nn.parallel.DistributedDataParallel(
            unet_struct,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
        )
        unet_res = torch.nn.parallel.DistributedDataParallel(
            unet_res,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
        )

    sched_struct = DDPMScheduler(num_train_timesteps=1000)
    sched_res = DDPMScheduler(num_train_timesteps=1000)

    params = list(unet_struct.parameters()) + list(unet_res.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    ckpt_path = os.path.join(out_root, "optionA_last.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        log0(f"[OPT-A] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        (unet_struct.module if ddp else unet_struct).load_state_dict(ckpt["unet_struct"], strict=True)
        (unet_res.module if ddp else unet_res).load_state_dict(ckpt["unet_res"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    accum = max(1, args.grad_accum)

    log0(f"[OPT-A] data={len(ds)} batch={args.batch_size} ddp={ddp} world={ddp_world()} device={device}")
    log0(f"[OPT-A] amp={args.amp} grad_accum={accum} steps={args.max_steps}")
    log0(f"[OPT-A] blur_kernel={args.blur_kernel} blur_sigma={args.blur_sigma} cond_noise_std={args.cond_noise_std}")
    log0(f"[OPT-A] using VAE ckpt: {vae_ckpt}")

    t0 = time.time()
    step = start_step
    unet_struct.train()
    unet_res.train()
    torch.backends.cudnn.benchmark = True

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x0 in dl:
            x0 = x0.to(device, non_blocking=True)
            x_blur = gaussian_blur(x0, kernel_size=args.blur_kernel, sigma=args.blur_sigma)

            with torch.no_grad():
                z_full = encode_latents(vae, x0)
                z_struct = encode_latents(vae, x_blur)
                z_res = z_full - z_struct

            b = z0_batch = z_full.size(0)

            # ----- Stage 1: structure latent diffusion -----
            t_struct = torch.randint(0, sched_struct.config.num_train_timesteps, (b,), device=device).long()
            noise_struct = torch.randn_like(z_struct)
            z_struct_t = sched_struct.add_noise(z_struct, noise_struct, t_struct)

            # ----- Stage 2: residual latent diffusion conditioned on structure latent -----
            t_res = torch.randint(0, sched_res.config.num_train_timesteps, (b,), device=device).long()
            noise_res = torch.randn_like(z_res)
            z_res_t = sched_res.add_noise(z_res, noise_res, t_res)

            cond = z_struct
            if args.cond_noise_std > 0:
                cond = cond + args.cond_noise_std * torch.randn_like(cond)

            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                pred_struct = unet_struct(z_struct_t, t_struct).sample
                loss_struct = F.mse_loss(pred_struct, noise_struct)

                inp_res = torch.cat([z_res_t, cond], dim=1)
                pred_res = unet_res(inp_res, t_res).sample
                loss_res = F.mse_loss(pred_res, noise_res)

                loss = (args.w_struct * loss_struct + args.w_res * loss_res) / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                lr = opt.param_groups[0]["lr"]
                log0(
                    f"[OPT-A] step={step}/{args.max_steps} "
                    f"loss={(loss.item() * accum):.4f} struct={loss_struct.item():.4f} res={loss_res.item():.4f} "
                    f"lr={lr:.2e} time={dt:.1f}m"
                )

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"samples_step{step:06d}.png")
                base_struct = unet_struct.module if ddp else unet_struct
                base_res = unet_res.module if ddp else unet_res
                sample_optionA(
                    vae=vae,
                    unet_struct=base_struct,
                    unet_res=base_res,
                    sched_struct=sched_struct,
                    sched_res=sched_res,
                    device=device,
                    out_png=out_png,
                    n=9,
                    infer_steps=args.infer_steps,
                    cond_noise_std=0.0,
                )
                log0(f"[OPT-A] saved sample: {out_png}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base_struct = unet_struct.module if ddp else unet_struct
                base_res = unet_res.module if ddp else unet_res
                torch.save({
                    "step": step,
                    "unet_struct": base_struct.state_dict(),
                    "unet_res": base_res.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                log0(f"[OPT-A] saved: {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break

    if is_main():
        base_struct = unet_struct.module if ddp else unet_struct
        base_res = unet_res.module if ddp else unet_res
        torch.save({
            "step": step,
            "unet_struct": base_struct.state_dict(),
            "unet_res": base_res.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        log0(f"[OPT-A] done. saved: {ckpt_path}")

    barrier()
    ddp_cleanup()


def cmd_sample_optionA(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    vae = SimpleVAE(z_channels=4).to(device)
    load_ckpt(vae, os.path.expanduser(args.vae_ckpt), map_location="cpu")
    vae.eval()

    unet_struct = build_struct_unet().to(device)
    unet_res = build_res_unet().to(device)

    ckpt = torch.load(os.path.expanduser(args.optionA_ckpt), map_location="cpu")
    unet_struct.load_state_dict(ckpt["unet_struct"], strict=True)
    unet_res.load_state_dict(ckpt["unet_res"], strict=True)
    unet_struct.eval()
    unet_res.eval()

    sched_struct = DDPMScheduler(num_train_timesteps=1000)
    sched_res = DDPMScheduler(num_train_timesteps=1000)

    seed_all(args.seed)
    out_png = os.path.join(out_root, f"samples_seed{args.seed}_steps{args.infer_steps}.png")
    sample_optionA(
        vae=vae,
        unet_struct=unet_struct,
        unet_res=unet_res,
        sched_struct=sched_struct,
        sched_res=sched_res,
        device=device,
        out_png=out_png,
        n=args.n,
        infer_steps=args.infer_steps,
        cond_noise_std=0.0,
    )
    print(f"[SAMPLE-OPT-A] saved: {out_png}")


# =========================================================
# Main
# =========================================================
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # -----------------------------------------------------
    # filter
    # -----------------------------------------------------
    p_f = sub.add_parser("filter_patches")
    p_f.add_argument("--in_root", required=True)
    p_f.add_argument("--out_root", required=True)
    p_f.add_argument("--resolution", type=int, default=512)
    p_f.add_argument("--top_ratio", type=float, default=0.7)
    p_f.add_argument("--seed", type=int, default=0)

    # -----------------------------------------------------
    # train_vae
    # -----------------------------------------------------
    p_v = sub.add_parser("train_vae")
    p_v.add_argument("--data_root", required=True)
    p_v.add_argument("--out_root", required=True)
    p_v.add_argument("--z_channels", type=int, default=4)
    p_v.add_argument("--batch_size", type=int, default=4)
    p_v.add_argument("--lr", type=float, default=2e-4)
    p_v.add_argument("--max_steps", type=int, default=100000)
    p_v.add_argument("--num_workers", type=int, default=4)
    p_v.add_argument("--amp", action="store_true")
    p_v.add_argument("--resume", action="store_true")
    p_v.add_argument("--save_every", type=int, default=2000)
    p_v.add_argument("--sample_every", type=int, default=2000)
    p_v.add_argument("--log_every", type=int, default=50)
    p_v.add_argument("--seed", type=int, default=0)

    p_v.add_argument("--w_l1", type=float, default=1.0)
    p_v.add_argument("--w_ssim", type=float, default=0.2)
    p_v.add_argument("--w_edge", type=float, default=0.1)
    p_v.add_argument("--w_fft", type=float, default=0.1)
    p_v.add_argument("--w_kl", type=float, default=1e-3)
    p_v.add_argument("--w_decor", type=float, default=1e-3)
    p_v.add_argument("--blur_kernel", type=int, default=21)
    p_v.add_argument("--blur_sigma", type=float, default=3.0)

    # -----------------------------------------------------
    # train_optionA
    # -----------------------------------------------------
    p_a = sub.add_parser("train_optionA")
    p_a.add_argument("--data_root", required=True)
    p_a.add_argument("--out_root", required=True)
    p_a.add_argument("--vae_ckpt", required=True)
    p_a.add_argument("--batch_size", type=int, default=8)
    p_a.add_argument("--lr", type=float, default=1e-4)
    p_a.add_argument("--max_steps", type=int, default=200000)
    p_a.add_argument("--num_workers", type=int, default=4)
    p_a.add_argument("--grad_accum", type=int, default=1)
    p_a.add_argument("--amp", action="store_true")
    p_a.add_argument("--resume", action="store_true")
    p_a.add_argument("--save_every", type=int, default=5000)
    p_a.add_argument("--sample_every", type=int, default=5000)
    p_a.add_argument("--log_every", type=int, default=50)
    p_a.add_argument("--infer_steps", type=int, default=250)
    p_a.add_argument("--seed", type=int, default=0)

    p_a.add_argument("--blur_kernel", type=int, default=21)
    p_a.add_argument("--blur_sigma", type=float, default=3.0)
    p_a.add_argument("--cond_noise_std", type=float, default=0.02)
    p_a.add_argument("--w_struct", type=float, default=1.0)
    p_a.add_argument("--w_res", type=float, default=1.0)

    # -----------------------------------------------------
    # sample_optionA
    # -----------------------------------------------------
    p_s = sub.add_parser("sample_optionA")
    p_s.add_argument("--vae_ckpt", required=True)
    p_s.add_argument("--optionA_ckpt", required=True)
    p_s.add_argument("--out_root", required=True)
    p_s.add_argument("--n", type=int, default=9)
    p_s.add_argument("--infer_steps", type=int, default=250)
    p_s.add_argument("--seed", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "filter_patches":
        cmd_filter_patches(args)
    elif args.cmd == "train_vae":
        cmd_train_vae(args)
    elif args.cmd == "train_optionA":
        cmd_train_optionA(args)
    elif args.cmd == "sample_optionA":
        cmd_sample_optionA(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()