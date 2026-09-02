#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ldm_microstructure_optionA_plus.py

Option A+:
Frequency-aware dual-branch VAE + two-stage latent diffusion for microstructure patches.

Pipeline:
  1) filter_patches
  2) train_vae_plus
  3) train_optionA_plus
  4) sample_optionA_plus

Core idea:
  x -> decompose into (x_low, x_high)
  x_low  -> encoder_low  -> z_low
  x_high -> encoder_high -> z_high

  decode_low(z_low)   -> x_low_hat
  decode_high(z_high) -> x_high_hat

  x_hat = x_low_hat + x_high_hat

Diffusion:
  Stage 1: unconditional DDPM on z_low
  Stage 2: conditional DDPM on z_high given z_low
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


def vis_signed_map(x: torch.Tensor):
    """
    For visualizing residual / high-frequency maps.
    Input: [-?, ?], shape [3,H,W]
    Output: [3,H,W] in [-1,1] for visualization
    """
    m = x.abs().amax().clamp_min(1e-6)
    y = (x / m).clamp(-1, 1)
    return y


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
            log0(f"[FILTER] scored {i+1}/{len(paths)}  time={(time.time()-t0):.1f}s")

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
# Frequency decomposition + losses
# =========================================================
def gaussian_kernel2d(kernel_size=9, sigma=1.0, channels=3, device="cpu", dtype=torch.float32):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def gaussian_blur(x, kernel_size=9, sigma=1.0):
    c = x.shape[1]
    k = gaussian_kernel2d(kernel_size, sigma, c, x.device, x.dtype)
    pad = kernel_size // 2
    return F.conv2d(x, k, padding=pad, groups=c)


def decompose_freq(x, kernel_size=9, sigma=1.0):
    x_low = gaussian_blur(x, kernel_size=kernel_size, sigma=sigma)
    x_high = x - x_low
    return x_low, x_high


def sobel_edges(x):
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
    g = x.mean(dim=1, keepdim=True)
    f = torch.fft.rfft2(g, norm="ortho")
    return torch.abs(f)


def fft_loss(x_rec, x):
    return F.l1_loss(fft_mag(x_rec), fft_mag(x))


def ssim_loss(x, y, window=11):
    x = (x + 1.0) * 0.5
    y = (y + 1.0) * 0.5

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window // 2

    mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return 1.0 - ssim_map.mean()


def vae_kl(mu, logvar):
    return 0.5 * torch.mean(torch.exp(logvar) + mu * mu - 1.0 - logvar)


def latent_decorrelation_loss(mu):
    z = mu.mean(dim=(2, 3))
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    offdiag = cov - torch.diag(torch.diag(cov))
    return (offdiag ** 2).mean()


# =========================================================
# Dual-branch VAE+
# =========================================================
class BranchEncoder(nn.Module):
    def __init__(self, in_ch=3, z_channels=4, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.SiLU(),             # 256
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.SiLU(),       # 128
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.SiLU(),   # 64
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1), nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(base_ch * 4, z_channels, 1)
        self.to_logvar = nn.Conv2d(base_ch * 4, z_channels, 1)

    def forward(self, x):
        h = self.net(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar


class BranchDecoder(nn.Module):
    def __init__(self, z_channels=4, out_ch=3, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, base_ch * 4, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1), nn.SiLU(),  # 128
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1), nn.SiLU(),      # 256
            nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1), nn.SiLU(),          # 512
            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
        )

    def forward(self, z):
        return self.net(z)


class DualBranchVAEPlus(nn.Module):
    def __init__(self, z_low_channels=4, z_high_channels=4):
        super().__init__()
        self.enc_low = BranchEncoder(in_ch=3, z_channels=z_low_channels, base_ch=64)
        self.enc_high = BranchEncoder(in_ch=3, z_channels=z_high_channels, base_ch=64)
        self.dec_low = BranchDecoder(z_channels=z_low_channels, out_ch=3, base_ch=64)
        self.dec_high = BranchDecoder(z_channels=z_high_channels, out_ch=3, base_ch=64)

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_low(self, x_low):
        return self.enc_low(x_low)

    def encode_high(self, x_high):
        return self.enc_high(x_high)

    def decode_low(self, z_low):
        return self.dec_low(z_low)

    def decode_high(self, z_high):
        return self.dec_high(z_high)

    def forward(self, x_low, x_high):
        mu_low, logvar_low = self.encode_low(x_low)
        mu_high, logvar_high = self.encode_high(x_high)

        z_low = self.reparam(mu_low, logvar_low)
        z_high = self.reparam(mu_high, logvar_high)

        x_low_hat = self.decode_low(z_low)
        x_high_hat = self.decode_high(z_high)
        x_hat = x_low_hat + x_high_hat

        return {
            "mu_low": mu_low,
            "logvar_low": logvar_low,
            "mu_high": mu_high,
            "logvar_high": logvar_high,
            "z_low": z_low,
            "z_high": z_high,
            "x_low_hat": x_low_hat,
            "x_high_hat": x_high_hat,
            "x_hat": x_hat,
        }


@torch.no_grad()
def vae_plus_recon_grid(model, dl, device, out_png, n=9, kernel_size=9, sigma=1.0):
    model.eval()
    xs = []
    for x in dl:
        xs.append(x)
        if len(xs) * x.size(0) >= n:
            break
    x = torch.cat(xs, dim=0)[:n].to(device)

    x_low, x_high = decompose_freq(x, kernel_size=kernel_size, sigma=sigma)

    mu_low, _ = model.encode_low(x_low)
    mu_high, _ = model.encode_high(x_high)

    x_low_hat = model.decode_low(mu_low)
    x_high_hat = model.decode_high(mu_high)
    x_hat = x_low_hat + x_high_hat

    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols * 512, rows * 512 * 4))

    for i in range(n):
        r, c = divmod(i, cols)
        canvas.paste(tensor_to_pil_m11(x[i]), (c * 512, r * 512 * 4 + 0))
        canvas.paste(tensor_to_pil_m11(x_low_hat[i]), (c * 512, r * 512 * 4 + 512))
        canvas.paste(tensor_to_pil_m11(vis_signed_map(x_high_hat[i])), (c * 512, r * 512 * 4 + 1024))
        canvas.paste(tensor_to_pil_m11(x_hat[i]), (c * 512, r * 512 * 4 + 1536))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)
    model.train()


def cmd_train_vae_plus(args):
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

    model = DualBranchVAEPlus(
        z_low_channels=args.z_low_channels,
        z_high_channels=args.z_high_channels,
    ).to(device)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    ckpt_path = os.path.join(out_root, "vae_plus_last.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        log0(f"[VAE+] resume from {ckpt_path}")
        ckpt = load_ckpt(model.module if ddp else model, ckpt_path, map_location="cpu")
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    log0(f"[VAE+] data={len(ds)} batch={args.batch_size} ddp={ddp} world={ddp_world()} device={device}")
    log0(
        f"[VAE+] z_low={args.z_low_channels} z_high={args.z_high_channels} "
        f"k={args.decomp_kernel} sigma={args.decomp_sigma}"
    )

    t0 = time.time()
    step = start_step
    model.train()

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x in dl:
            x = x.to(device, non_blocking=True)
            x_low, x_high = decompose_freq(x, kernel_size=args.decomp_kernel, sigma=args.decomp_sigma)

            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                out = model(x_low, x_high)

                x_low_hat = out["x_low_hat"]
                x_high_hat = out["x_high_hat"]
                x_hat = out["x_hat"]

                mu_low = out["mu_low"]
                logvar_low = out["logvar_low"]
                mu_high = out["mu_high"]
                logvar_high = out["logvar_high"]

                l_rec = F.l1_loss(x_hat, x)
                l_low = F.l1_loss(x_low_hat, x_low)
                l_high = F.l1_loss(x_high_hat, x_high)
                l_ssim = ssim_loss(x_hat, x)
                l_edge = edge_loss(x_hat, x)
                l_fft = fft_loss(x_hat, x)

                l_kl_low = vae_kl(mu_low, logvar_low)
                l_kl_high = vae_kl(mu_high, logvar_high)

                l_decor_low = latent_decorrelation_loss(mu_low)
                l_decor_high = latent_decorrelation_loss(mu_high)

                loss = (
                    args.w_rec * l_rec +
                    args.w_low * l_low +
                    args.w_high * l_high +
                    args.w_ssim * l_ssim +
                    args.w_edge * l_edge +
                    args.w_fft * l_fft +
                    args.w_kl_low * l_kl_low +
                    args.w_kl_high * l_kl_high +
                    args.w_decor_low * l_decor_low +
                    args.w_decor_high * l_decor_high
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                log0(
                    f"[VAE+] step={step}/{args.max_steps} "
                    f"loss={loss.item():.4f} rec={l_rec.item():.4f} low={l_low.item():.4f} high={l_high.item():.4f} "
                    f"ssim={l_ssim.item():.4f} edge={l_edge.item():.4f} fft={l_fft.item():.4f} "
                    f"kl_low={l_kl_low.item():.4f} kl_high={l_kl_high.item():.4f} "
                    f"time={dt:.1f}m"
                )

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"vae_plus_recon_step{step:06d}.png")
                base = model.module if ddp else model
                vae_plus_recon_grid(
                    base, dl, device, out_png, n=9,
                    kernel_size=args.decomp_kernel, sigma=args.decomp_sigma
                )
                log0(f"[VAE+] saved recon: {out_png}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base = model.module if ddp else model
                save_ckpt(base, ckpt_path, extra={
                    "step": step,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                })
                log0(f"[VAE+] saved: {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break

    if is_main():
        base = model.module if ddp else model
        save_ckpt(base, ckpt_path, extra={
            "step": step,
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        })
        log0(f"[VAE+] done. saved: {ckpt_path}")

    barrier()
    ddp_cleanup()


# =========================================================
# Latent helpers
# =========================================================
@torch.no_grad()
def encode_low_mu(model, x_low):
    mu, _ = model.encode_low(x_low)
    return mu


@torch.no_grad()
def encode_high_mu(model, x_high):
    mu, _ = model.encode_high(x_high)
    return mu


@torch.no_grad()
def decode_dual(model, z_low, z_high):
    x_low_hat = model.decode_low(z_low)
    x_high_hat = model.decode_high(z_high)
    return x_low_hat, x_high_hat, x_low_hat + x_high_hat


# =========================================================
# Diffusion models
# =========================================================
def build_low_unet(z_low_channels=4):
    return UNet2DModel(
        sample_size=64,
        in_channels=z_low_channels,
        out_channels=z_low_channels,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    )


def build_high_unet(z_high_channels=4, z_low_channels=4):
    # conditional input = [z_high_t, z_low_cond]
    return UNet2DModel(
        sample_size=64,
        in_channels=z_high_channels + z_low_channels,
        out_channels=z_high_channels,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    )


@torch.no_grad()
def sample_optionA_plus(
    model,
    unet_low,
    unet_high,
    sched_low,
    sched_high,
    device,
    out_png,
    z_low_channels=4,
    z_high_channels=4,
    n=9,
    infer_steps=250,
    cond_noise_std=0.0,
):
    model.eval()
    unet_low.eval()
    unet_high.eval()

    # stage 1: low latent
    z_low = torch.randn(n, z_low_channels, 64, 64, device=device)
    sched_low.set_timesteps(infer_steps, device=device)
    for t in sched_low.timesteps:
        eps = unet_low(z_low, t).sample
        z_low = sched_low.step(eps, t, z_low).prev_sample

    # stage 2: high latent conditioned on low
    cond = z_low
    if cond_noise_std > 0:
        cond = cond + cond_noise_std * torch.randn_like(cond)

    z_high = torch.randn(n, z_high_channels, 64, 64, device=device)
    sched_high.set_timesteps(infer_steps, device=device)
    for t in sched_high.timesteps:
        inp = torch.cat([z_high, cond], dim=1)
        eps = unet_high(inp, t).sample
        z_high = sched_high.step(eps, t, z_high).prev_sample

    x_low_hat, x_high_hat, x_hat = decode_dual(model, z_low, z_high)

    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols * 512, rows * 512 * 3))

    for i in range(n):
        r, c = divmod(i, cols)
        canvas.paste(tensor_to_pil_m11(x_low_hat[i]), (c * 512, r * 512 * 3 + 0))
        canvas.paste(tensor_to_pil_m11(vis_signed_map(x_high_hat[i])), (c * 512, r * 512 * 3 + 512))
        canvas.paste(tensor_to_pil_m11(x_hat[i]), (c * 512, r * 512 * 3 + 1024))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)

    unet_low.train()
    unet_high.train()
    model.train()


def cmd_train_optionA_plus(args):
    ddp = ddp_setup()
    seed_all(args.seed + ddp_rank())

    data_root = os.path.expanduser(args.data_root)
    out_root = os.path.expanduser(args.out_root)
    vae_ckpt = os.path.expanduser(args.vae_plus_ckpt)

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

    model = DualBranchVAEPlus(
        z_low_channels=args.z_low_channels,
        z_high_channels=args.z_high_channels,
    ).to(device)
    load_ckpt(model, vae_ckpt, map_location="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    unet_low = build_low_unet(args.z_low_channels).to(device)
    unet_high = build_high_unet(args.z_high_channels, args.z_low_channels).to(device)

    if ddp:
        unet_low = torch.nn.parallel.DistributedDataParallel(
            unet_low,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
        )
        unet_high = torch.nn.parallel.DistributedDataParallel(
            unet_high,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            find_unused_parameters=False,
        )

    sched_low = DDPMScheduler(num_train_timesteps=1000)
    sched_high = DDPMScheduler(num_train_timesteps=1000)

    params = list(unet_low.parameters()) + list(unet_high.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    ckpt_path = os.path.join(out_root, "optionA_plus_last.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        log0(f"[OPT-A+] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        (unet_low.module if ddp else unet_low).load_state_dict(ckpt["unet_low"], strict=True)
        (unet_high.module if ddp else unet_high).load_state_dict(ckpt["unet_high"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    accum = max(1, args.grad_accum)

    log0(f"[OPT-A+] data={len(ds)} batch={args.batch_size} ddp={ddp} world={ddp_world()} device={device}")
    log0(
        f"[OPT-A+] z_low={args.z_low_channels} z_high={args.z_high_channels} "
        f"k={args.decomp_kernel} sigma={args.decomp_sigma} cond_noise_std={args.cond_noise_std}"
    )
    log0(f"[OPT-A+] using VAE+ ckpt: {vae_ckpt}")

    t0 = time.time()
    step = start_step
    unet_low.train()
    unet_high.train()
    torch.backends.cudnn.benchmark = True

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x in dl:
            x = x.to(device, non_blocking=True)
            x_low, x_high = decompose_freq(x, kernel_size=args.decomp_kernel, sigma=args.decomp_sigma)

            with torch.no_grad():
                z_low = encode_low_mu(model, x_low)
                z_high = encode_high_mu(model, x_high)

            b = z_low.size(0)

            # stage 1: z_low diffusion
            t_low = torch.randint(0, sched_low.config.num_train_timesteps, (b,), device=device).long()
            noise_low = torch.randn_like(z_low)
            z_low_t = sched_low.add_noise(z_low, noise_low, t_low)

            # stage 2: z_high diffusion conditioned on z_low
            t_high = torch.randint(0, sched_high.config.num_train_timesteps, (b,), device=device).long()
            noise_high = torch.randn_like(z_high)
            z_high_t = sched_high.add_noise(z_high, noise_high, t_high)

            cond = z_low
            if args.cond_noise_std > 0:
                cond = cond + args.cond_noise_std * torch.randn_like(cond)

            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                pred_low = unet_low(z_low_t, t_low).sample
                loss_low = F.mse_loss(pred_low, noise_low)

                inp_high = torch.cat([z_high_t, cond], dim=1)
                pred_high = unet_high(inp_high, t_high).sample
                loss_high = F.mse_loss(pred_high, noise_high)

                loss = (args.w_low_stage * loss_low + args.w_high_stage * loss_high) / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                lr = opt.param_groups[0]["lr"]
                log0(
                    f"[OPT-A+] step={step}/{args.max_steps} "
                    f"loss={(loss.item() * accum):.4f} low_stage={loss_low.item():.4f} high_stage={loss_high.item():.4f} "
                    f"lr={lr:.2e} time={dt:.1f}m"
                )

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"samples_step{step:06d}.png")
                base_low = unet_low.module if ddp else unet_low
                base_high = unet_high.module if ddp else unet_high
                sample_optionA_plus(
                    model=model,
                    unet_low=base_low,
                    unet_high=base_high,
                    sched_low=sched_low,
                    sched_high=sched_high,
                    device=device,
                    out_png=out_png,
                    z_low_channels=args.z_low_channels,
                    z_high_channels=args.z_high_channels,
                    n=9,
                    infer_steps=args.infer_steps,
                    cond_noise_std=0.0,
                )
                log0(f"[OPT-A+] saved sample: {out_png}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base_low = unet_low.module if ddp else unet_low
                base_high = unet_high.module if ddp else unet_high
                torch.save({
                    "step": step,
                    "unet_low": base_low.state_dict(),
                    "unet_high": base_high.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                log0(f"[OPT-A+] saved: {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break

    if is_main():
        base_low = unet_low.module if ddp else unet_low
        base_high = unet_high.module if ddp else unet_high
        torch.save({
            "step": step,
            "unet_low": base_low.state_dict(),
            "unet_high": base_high.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        log0(f"[OPT-A+] done. saved: {ckpt_path}")

    barrier()
    ddp_cleanup()


def cmd_sample_optionA_plus(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    model = DualBranchVAEPlus(
        z_low_channels=args.z_low_channels,
        z_high_channels=args.z_high_channels,
    ).to(device)
    load_ckpt(model, os.path.expanduser(args.vae_plus_ckpt), map_location="cpu")
    model.eval()

    unet_low = build_low_unet(args.z_low_channels).to(device)
    unet_high = build_high_unet(args.z_high_channels, args.z_low_channels).to(device)

    ckpt = torch.load(os.path.expanduser(args.optionA_plus_ckpt), map_location="cpu")
    unet_low.load_state_dict(ckpt["unet_low"], strict=True)
    unet_high.load_state_dict(ckpt["unet_high"], strict=True)
    unet_low.eval()
    unet_high.eval()

    sched_low = DDPMScheduler(num_train_timesteps=1000)
    sched_high = DDPMScheduler(num_train_timesteps=1000)

    seed_all(args.seed)
    out_png = os.path.join(out_root, f"samples_seed{args.seed}_steps{args.infer_steps}.png")
    sample_optionA_plus(
        model=model,
        unet_low=unet_low,
        unet_high=unet_high,
        sched_low=sched_low,
        sched_high=sched_high,
        device=device,
        out_png=out_png,
        z_low_channels=args.z_low_channels,
        z_high_channels=args.z_high_channels,
        n=args.n,
        infer_steps=args.infer_steps,
        cond_noise_std=0.0,
    )
    print(f"[SAMPLE-OPT-A+] saved: {out_png}")


# =========================================================
# Parser
# =========================================================
def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # filter
    p_f = sub.add_parser("filter_patches")
    p_f.add_argument("--in_root", required=True)
    p_f.add_argument("--out_root", required=True)
    p_f.add_argument("--resolution", type=int, default=512)
    p_f.add_argument("--top_ratio", type=float, default=0.7)
    p_f.add_argument("--seed", type=int, default=0)

    # train_vae_plus
    p_v = sub.add_parser("train_vae_plus")
    p_v.add_argument("--data_root", required=True)
    p_v.add_argument("--out_root", required=True)
    p_v.add_argument("--z_low_channels", type=int, default=4)
    p_v.add_argument("--z_high_channels", type=int, default=4)
    p_v.add_argument("--batch_size", type=int, default=4)
    p_v.add_argument("--lr", type=float, default=2e-4)
    p_v.add_argument("--max_steps", type=int, default=100000)
    p_v.add_argument("--num_workers", type=int, default=4)
    p_v.add_argument("--amp", action="store_true")
    p_v.add_argument("--resume", action="store_true")
    p_v.add_argument("--save_every", type=int, default=2000)
    p_v.add_argument("--sample_every", type=int, default=1000)
    p_v.add_argument("--log_every", type=int, default=50)
    p_v.add_argument("--seed", type=int, default=0)

    p_v.add_argument("--decomp_kernel", type=int, default=9)
    p_v.add_argument("--decomp_sigma", type=float, default=1.0)

    p_v.add_argument("--w_rec", type=float, default=1.0)
    p_v.add_argument("--w_low", type=float, default=0.5)
    p_v.add_argument("--w_high", type=float, default=1.0)
    p_v.add_argument("--w_ssim", type=float, default=0.2)
    p_v.add_argument("--w_edge", type=float, default=0.2)
    p_v.add_argument("--w_fft", type=float, default=0.3)
    p_v.add_argument("--w_kl_low", type=float, default=1e-4)
    p_v.add_argument("--w_kl_high", type=float, default=5e-5)
    p_v.add_argument("--w_decor_low", type=float, default=1e-3)
    p_v.add_argument("--w_decor_high", type=float, default=1e-3)

    # train_optionA_plus
    p_a = sub.add_parser("train_optionA_plus")
    p_a.add_argument("--data_root", required=True)
    p_a.add_argument("--out_root", required=True)
    p_a.add_argument("--vae_plus_ckpt", required=True)
    p_a.add_argument("--z_low_channels", type=int, default=4)
    p_a.add_argument("--z_high_channels", type=int, default=4)
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

    p_a.add_argument("--decomp_kernel", type=int, default=9)
    p_a.add_argument("--decomp_sigma", type=float, default=1.0)
    p_a.add_argument("--cond_noise_std", type=float, default=0.02)
    p_a.add_argument("--w_low_stage", type=float, default=1.0)
    p_a.add_argument("--w_high_stage", type=float, default=1.0)

    # sample_optionA_plus
    p_s = sub.add_parser("sample_optionA_plus")
    p_s.add_argument("--vae_plus_ckpt", required=True)
    p_s.add_argument("--optionA_plus_ckpt", required=True)
    p_s.add_argument("--out_root", required=True)
    p_s.add_argument("--z_low_channels", type=int, default=4)
    p_s.add_argument("--z_high_channels", type=int, default=4)
    p_s.add_argument("--n", type=int, default=9)
    p_s.add_argument("--infer_steps", type=int, default=250)
    p_s.add_argument("--seed", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "filter_patches":
        cmd_filter_patches(args)
    elif args.cmd == "train_vae_plus":
        cmd_train_vae_plus(args)
    elif args.cmd == "train_optionA_plus":
        cmd_train_optionA_plus(args)
    elif args.cmd == "sample_optionA_plus":
        cmd_sample_optionA_plus(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()