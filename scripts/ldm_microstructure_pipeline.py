#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ldm_microstructure_pipeline.py
Latent Diffusion pipeline for microstructure patches (512x512)

Commands:
  1) filter_patches : Sobel gradient-energy ranking, keep top ratio
  2) train_vae      : Train a lightweight VAE -> latent (4, 64, 64)
  3) train_ldm      : Train DDPM UNet in latent space (64x64)
  4) sample         : Sample from latent DDPM and decode by VAE

DDP:
  torchrun --nproc_per_node=4 ldm_microstructure_pipeline.py train_vae ...
  torchrun --nproc_per_node=4 ldm_microstructure_pipeline.py train_ldm ...
"""

import os
import math
import time
import glob
import json
import random
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# diffusers (no accelerate needed)
from diffusers import UNet2DModel, DDPMScheduler


# -------------------------
# Utilities
# -------------------------
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

def to_tensor_01(im: Image.Image):
    x = torch.from_numpy(np.array(im)).float() / 255.0   # [0,1]
    x = x.permute(2, 0, 1).contiguous()                 # CHW
    return x

def to_tensor_m11(im: Image.Image):
    x = torch.from_numpy(np.array(im)).float() / 127.5 - 1.0  # [-1,1]
    x = x.permute(2, 0, 1).contiguous()
    return x

def tensor_to_pil_01(x: torch.Tensor):
    # x: [3,H,W] in [0,1]
    x = (x.clamp(0,1) * 255.0).to(torch.uint8).permute(1,2,0).cpu().numpy()
    return Image.fromarray(x)

def tensor_to_pil_m11(x: torch.Tensor):
    # x: [3,H,W] in [-1,1]
    x = ((x.clamp(-1,1) + 1) * 127.5).to(torch.uint8).permute(1,2,0).cpu().numpy()
    return Image.fromarray(x)


# -------------------------
# Dataset (patch folder)
# -------------------------
class PatchFolder(Dataset):
    def __init__(self, root: str, resolution: int = 512, flip: bool = True):
        self.paths = list_images(root)
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under: {root}")
        self.res = resolution
        self.flip = flip

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = pil_load_rgb(p)
        img = pil_center_crop_resize(img, self.res)
        if self.flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = to_tensor_m11(img)  # [-1,1]
        return x


# -------------------------
# Sobel filter scoring
# -------------------------
def sobel_energy_score_rgb_pil(im: Image.Image) -> float:
    """
    Score patch by Sobel gradient energy on grayscale.
    Implemented in torch conv2d on CPU for stability and no extra deps.
    """
    # grayscale [0,1]
    g = im.convert("L")
    x = torch.from_numpy(np.array(g)).float().unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
    x = x.contiguous()

    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-12)  # [1,1,H,W]

    # robust energy: mean of top-k pixels to emphasize “strong structure”
    flat = mag.flatten()
    k = max(1, int(0.10 * flat.numel()))  # top 10%
    topk = torch.topk(flat, k, largest=True).values
    score = float(topk.mean().item())
    return score

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

    # score all
    scored = []
    t0 = time.time()
    for i, p in enumerate(paths):
        im = pil_load_rgb(p)
        # ensure correct size (some may not be 512)
        im = pil_center_crop_resize(im, args.resolution)
        s = sobel_energy_score_rgb_pil(im)
        scored.append((s, p))
        if (i+1) % 200 == 0:
            log0(f"[FILTER] scored {i+1}/{len(paths)}  time={(time.time()-t0):.1f}s")

    scored.sort(key=lambda x: x[0], reverse=True)
    keep_n = int(round(args.top_ratio * len(scored)))
    keep_n = max(1, min(len(scored), keep_n))

    kept = scored[:keep_n]
    thr = kept[-1][0]
    log0(f"[FILTER] keep_n={keep_n}/{len(scored)}  score_threshold≈{thr:.6f}")

    # copy kept patches into out_root (flat)
    # NOTE: to avoid huge inode explosion, we keep flat filenames.
    meta = []
    for rank, (s, p) in enumerate(kept):
        src = p
        stem = Path(src).stem
        dst = os.path.join(out_root, f"{rank:06d}_{stem}.png")
        im = pil_load_rgb(src)
        im = pil_center_crop_resize(im, args.resolution)
        im.save(dst, "PNG")
        meta.append({"rank": rank, "score": float(s), "src": src, "dst": dst})

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


# -------------------------
# VAE (lightweight)
# latent: (z_channels=4, 64, 64) for 512 input
# -------------------------
class SimpleVAE(nn.Module):
    def __init__(self, z_channels=4):
        super().__init__()
        ch = 64
        # Encoder: 512 -> 256 -> 128 -> 64
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.SiLU(),            # 256
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.SiLU(),         # 128
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.SiLU(),       # 64
            nn.Conv2d(ch*4, ch*4, 3, 1, 1), nn.SiLU(),
        )
        self.to_mu = nn.Conv2d(ch*4, z_channels, 1)
        self.to_logvar = nn.Conv2d(ch*4, z_channels, 1)

        # Decoder: 64 -> 128 -> 256 -> 512
        self.dec = nn.Sequential(
            nn.Conv2d(z_channels, ch*4, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1), nn.SiLU(),  # 128
            nn.ConvTranspose2d(ch*2, ch, 4, 2, 1), nn.SiLU(),    # 256
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.SiLU(),      # 512
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
    # KL(q(z|x)||N(0,1)) per batch
    return 0.5 * torch.mean(torch.exp(logvar) + mu*mu - 1.0 - logvar)

@torch.no_grad()
def vae_recon_grid(vae, dl, device, out_png, n=9):
    vae.eval()
    xs = []
    for x in dl:
        xs.append(x)
        if len(xs) * x.size(0) >= n:
            break
    x = torch.cat(xs, dim=0)[:n].to(device)
    x_rec, mu, logvar, z = vae(x)
    # [-1,1] -> grid
    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    W = cols * 512
    H = rows * 512 * 2
    canvas = Image.new("RGB", (W, H))
    for i in range(n):
        r, c = divmod(i, cols)
        im0 = tensor_to_pil_m11(x[i])
        im1 = tensor_to_pil_m11(x_rec[i])
        canvas.paste(im0, (c*512, r*512*2))
        canvas.paste(im1, (c*512, r*512*2 + 512))
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
    out_root  = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    device = f"cuda:{ddp_local_rank()}" if torch.cuda.is_available() else "cpu"

    ds = PatchFolder(data_root, resolution=512, flip=True)
    if ddp:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True)
    else:
        sampler = None

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    vae = SimpleVAE(z_channels=args.z_channels).to(device)

    if ddp:
        vae = torch.nn.parallel.DistributedDataParallel(
            vae, device_ids=[ddp_local_rank()], output_device=ddp_local_rank(),
            find_unused_parameters=False
        )

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    start_step = 0
    ckpt_path = os.path.join(out_root, "vae_last.pt")
    if args.resume and os.path.exists(ckpt_path):
        log0(f"[VAE] resume from {ckpt_path}")
        ckpt = load_ckpt(vae.module if ddp else vae, ckpt_path, map_location="cpu")
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    log0(f"[VAE] data={len(ds)}  batch={args.batch_size}  ddp={ddp} world={ddp_world()} device={device}")
    log0(f"[VAE] z_channels={args.z_channels}  kl_weight={args.kl_weight}  amp={args.amp}")

    t0 = time.time()
    step = start_step
    vae.train()

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x in dl:
            x = x.to(device, non_blocking=True)  # [-1,1]
            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                x_rec, mu, logvar, z = (vae(x) if not ddp else vae(x))
                rec_loss = F.l1_loss(x_rec, x)
                kl = vae_kl(mu, logvar)
                loss = rec_loss + args.kl_weight * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                log0(f"[VAE] step={step}/{args.max_steps} loss={loss.item():.4f} rec={rec_loss.item():.4f} kl={kl.item():.4f} time={dt:.1f}m")

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"vae_recon_step{step:06d}.png")
                # use a non-distributed loader slice (reuse dl is OK; only rank0 runs)
                base_vae = vae.module if ddp else vae
                vae_recon_grid(base_vae, dl, device, out_png, n=9)
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


# -------------------------
# Latent Diffusion (DDPM in latent space)
# -------------------------
@torch.no_grad()
def encode_latents(vae: SimpleVAE, x: torch.Tensor):
    # x: [-1,1]
    mu, logvar = vae.encode(x)
    # use mean as deterministic encoding (more stable than sampling)
    z = mu
    return z

@torch.no_grad()
def decode_latents(vae: SimpleVAE, z: torch.Tensor):
    x = vae.decode(z)
    return x

@torch.no_grad()
def sample_ldm(step, vae, unet, scheduler, device, out_png, n=9, infer_steps=250):
    vae.eval()
    unet.eval()

    z = torch.randn(n, 4, 64, 64, device=device)
    scheduler.set_timesteps(infer_steps, device=device)

    for t in scheduler.timesteps:
        eps = unet(z, t).sample
        z = scheduler.step(eps, t, z).prev_sample

    x = decode_latents(vae, z)  # [-1,1]
    cols = int(math.sqrt(n))
    cols = max(1, cols)
    rows = int(math.ceil(n / cols))
    canvas = Image.new("RGB", (cols*512, rows*512))
    for i in range(n):
        r, c = divmod(i, cols)
        canvas.paste(tensor_to_pil_m11(x[i]), (c*512, r*512))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)
    unet.train()
    vae.train()

def cmd_train_ldm(args):
    ddp = ddp_setup()
    seed_all(args.seed + ddp_rank())

    data_root = os.path.expanduser(args.data_root)
    out_root  = os.path.expanduser(args.out_root)
    vae_ckpt  = os.path.expanduser(args.vae_ckpt)

    os.makedirs(out_root, exist_ok=True)
    device = f"cuda:{ddp_local_rank()}" if torch.cuda.is_available() else "cpu"

    # dataset (512)
    ds = PatchFolder(data_root, resolution=512, flip=True)
    if ddp:
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True)
    else:
        sampler = None
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # load VAE (frozen)
    vae = SimpleVAE(z_channels=4).to(device)
    load_ckpt(vae, vae_ckpt, map_location="cpu")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Latent UNet (64x64, 4ch)
    unet = UNet2DModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),  # bigger than your pixel UNet, but cheap in latent
        down_block_types=("DownBlock2D","AttnDownBlock2D","DownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","UpBlock2D","AttnUpBlock2D","UpBlock2D"),
        attention_head_dim=8,
    ).to(device)

    if ddp:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[ddp_local_rank()], output_device=ddp_local_rank(),
            find_unused_parameters=False
        )

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.startswith("cuda"))

    # resume
    ckpt_path = os.path.join(out_root, "ldm_last.pt")
    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        log0(f"[LDM] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        (unet.module if ddp else unet).load_state_dict(ckpt["unet"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))

    # grad accumulation
    accum = max(1, args.grad_accum)

    log0(f"[LDM] data={len(ds)}  batch={args.batch_size}  ddp={ddp} world={ddp_world()} device={device}")
    log0(f"[LDM] amp={args.amp} grad_accum={accum}  steps={args.max_steps}")
    log0(f"[LDM] using VAE ckpt: {vae_ckpt}")

    t0 = time.time()
    step = start_step
    unet.train()

    # optional: reduce DDP overhead / perf warnings
    torch.backends.cudnn.benchmark = True

    while step < args.max_steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // max(1, len(dl)))

        for x0 in dl:
            x0 = x0.to(device, non_blocking=True)  # [-1,1]

            with torch.no_grad():
                z0 = encode_latents(vae, x0)  # [B,4,64,64]
                # latent scaling is optional; keep as-is for simplicity

            b = z0.size(0)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device).long()
            noise = torch.randn_like(z0)
            zt = scheduler.add_noise(z0, noise, t)

            with torch.cuda.amp.autocast(enabled=args.amp and device.startswith("cuda")):
                pred = (unet(zt, t).sample if not ddp else unet(zt, t).sample)
                loss = F.mse_loss(pred, noise) / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            if is_main() and step % args.log_every == 0:
                dt = (time.time() - t0) / 60.0
                lr = opt.param_groups[0]["lr"]
                log0(f"[LDM] step={step}/{args.max_steps} loss={loss.item()*accum:.4f} lr={lr:.2e} time={dt:.1f}m")

            if is_main() and step > 0 and step % args.sample_every == 0:
                out_png = os.path.join(out_root, f"samples_step{step:06d}.png")
                base_unet = unet.module if ddp else unet
                sample_ldm(step, vae, base_unet, scheduler, device, out_png, n=9, infer_steps=args.infer_steps)
                log0(f"[LDM] saved sample: {out_png}")

            if is_main() and step > 0 and step % args.save_every == 0:
                base_unet = unet.module if ddp else unet
                torch.save({
                    "step": step,
                    "unet": base_unet.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                log0(f"[LDM] saved: {ckpt_path}")

            step += 1
            if step >= args.max_steps:
                break

    if is_main():
        base_unet = unet.module if ddp else unet
        torch.save({
            "step": step,
            "unet": base_unet.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }, ckpt_path)
        log0(f"[LDM] done. saved: {ckpt_path}")

    barrier()
    ddp_cleanup()


def cmd_sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    vae = SimpleVAE(z_channels=4).to(device)
    load_ckpt(vae, os.path.expanduser(args.vae_ckpt), map_location="cpu")
    vae.eval()

    unet = UNet2DModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(256, 512, 768, 768),
        down_block_types=("DownBlock2D","AttnDownBlock2D","DownBlock2D","AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D","UpBlock2D","AttnUpBlock2D","UpBlock2D"),
        attention_head_dim=8,
    ).to(device)

    ckpt = torch.load(os.path.expanduser(args.ldm_ckpt), map_location="cpu")
    unet.load_state_dict(ckpt["unet"], strict=True)
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    seed_all(args.seed)
    out_png = os.path.join(out_root, f"samples_seed{args.seed}_steps{args.infer_steps}.png")
    sample_ldm(0, vae, unet, scheduler, device, out_png, n=args.n, infer_steps=args.infer_steps)
    print(f"[SAMPLE] saved: {out_png}")


# -------------------------
# Main
# -------------------------
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

    # train_vae
    p_v = sub.add_parser("train_vae")
    p_v.add_argument("--data_root", required=True)
    p_v.add_argument("--out_root", required=True)
    p_v.add_argument("--z_channels", type=int, default=4)
    p_v.add_argument("--batch_size", type=int, default=4)
    p_v.add_argument("--lr", type=float, default=2e-4)
    p_v.add_argument("--kl_weight", type=float, default=1e-3)
    p_v.add_argument("--max_steps", type=int, default=100000)
    p_v.add_argument("--num_workers", type=int, default=4)
    p_v.add_argument("--amp", action="store_true")
    p_v.add_argument("--resume", action="store_true")
    p_v.add_argument("--save_every", type=int, default=2000)
    p_v.add_argument("--sample_every", type=int, default=2000)
    p_v.add_argument("--log_every", type=int, default=50)
    p_v.add_argument("--seed", type=int, default=0)

    # train_ldm
    p_l = sub.add_parser("train_ldm")
    p_l.add_argument("--data_root", required=True)
    p_l.add_argument("--out_root", required=True)
    p_l.add_argument("--vae_ckpt", required=True)
    p_l.add_argument("--batch_size", type=int, default=8)
    p_l.add_argument("--lr", type=float, default=1e-4)
    p_l.add_argument("--max_steps", type=int, default=200000)
    p_l.add_argument("--num_workers", type=int, default=4)
    p_l.add_argument("--grad_accum", type=int, default=1)
    p_l.add_argument("--amp", action="store_true")
    p_l.add_argument("--resume", action="store_true")
    p_l.add_argument("--save_every", type=int, default=5000)
    p_l.add_argument("--sample_every", type=int, default=5000)
    p_l.add_argument("--log_every", type=int, default=50)
    p_l.add_argument("--infer_steps", type=int, default=250)
    p_l.add_argument("--seed", type=int, default=0)

    # sample
    p_s = sub.add_parser("sample")
    p_s.add_argument("--vae_ckpt", required=True)
    p_s.add_argument("--ldm_ckpt", required=True)
    p_s.add_argument("--out_root", required=True)
    p_s.add_argument("--n", type=int, default=9)
    p_s.add_argument("--infer_steps", type=int, default=300)
    p_s.add_argument("--seed", type=int, default=0)

    return p

def main():
    args = build_parser().parse_args()

    if args.cmd == "filter_patches":
        cmd_filter_patches(args)
    elif args.cmd == "train_vae":
        cmd_train_vae(args)
    elif args.cmd == "train_ldm":
        cmd_train_ldm(args)
    elif args.cmd == "sample":
        cmd_sample(args)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()
